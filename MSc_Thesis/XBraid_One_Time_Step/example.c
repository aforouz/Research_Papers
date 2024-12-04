#include "example_lib.c"

/* --------------------------------------------------------------------
 * XBraid app struct 
 * -------------------------------------------------------------------- */
typedef struct _braid_App_struct {
   MPI_Comm                comm;             /* global communicator */
   MPI_Comm                comm_t;           /* communicator for parallelizing in time  */
   MPI_Comm                comm_x;           /* communicator for parallelizing in space  */
   int                     pt;               /* number of processors in time  */
   simulation_manager     *man;              /* user's simulation manager structure */
   HYPRE_SStructVector     e;                /* temporary vector used for error computations */
   int                     nA;               /* number of discr. matrices that have been created */
   int                     max_nA;           /* max nA value allowed */
   HYPRE_IJMatrix         *Phi;                /* nA sized array of discr. matrices (one per time level) */
   HYPRE_IJVector         *G;
   double                 *dt_A;             /* nA sized array of time step sizes for each  matrix  */
   int                     use_rand;         /* binary, use random initial guess (1) or zero initial guess (0) */
   int                    *runtime_max_iter; /* runtime info for the max number of spatial solve iterations at each level */
   int                    *max_iter_x;       /* length 2 array of expensive and cheap max PFMG iters for spatial solves*/
} my_App;

/* --------------------------------------------------------------------
 * XBraid vector 
 * Stores the state of the simulation for a given time step
 * -------------------------------------------------------------------- */
typedef struct _braid_Vector_struct {
   HYPRE_SStructVector   x;
} my_Vector;

/* --------------------------------------------------------------------
 * Time integrator routine that performs the update
 *   u_i = Phi_i(u_{i-1}) + g_i 
 * 
 * When Phi is called, u is u_{i-1}.
 * The return value is that u is set to u_i upon completion
 * -------------------------------------------------------------------- */
int
my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
   double tstart;             /* current time */
   double tstop;              /* evolve u to this time*/
   HYPRE_SStructVector  bstop;
   int level;
   int iters_taken = -1;
   
   /* Grab status of current time step */
   braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
   braid_StepStatusGetLevel(status, &level);

   /* Now, set up the discretization matrix.  Use the XBraid level to index
    * into the matrix lookup table */

   /* We need to "trick" the user's manager with the new dt */
   app->man->dt = tstop - tstart;

   /* Set up a new matrix */
   if( app->dt_A[level] == -1.0 )
   {
      app->nA++;
      app->dt_A[level] = tstop - tstart;
      
//     printf("Phi: ");
      setUpPhiMatrix(app->man);
      app->Phi[level] = app->man->Phi;
//      printf("Pass\n");
//      HYPRE_IJMatrixPrint(app->man->Phi, "txt/Phi");
//      HYPRE_IJMatrixPrint(app->man->A, "txt/A");

//      printf("G: ");
      setUpGVector(app->man);
      app->G[level] = app->man->G;
//      printf("Pass\n");
//      HYPRE_IJVectorPrint(app->man->G, "txt/G");
   }

   /* Time integration to next time point: Solve the system Ax = b.
    * First, "trick" the user's manager with the right matrix and solver */ 
   app->man->Phi = app->Phi[level];
   app->man->G = app->G[level];

   /* Take step */
   if (fstop == NULL)
   {
      bstop = NULL;
   }
   else
   {
      bstop = fstop->x;
   }

   take_step(app->man, ustop->x, bstop, u->x, tstart, tstop);

   /* Store iterations taken */
   app->runtime_max_iter[level] = max_i( (app->runtime_max_iter[level]), 100);

   /* Tell XBraid no refinement */
   braid_StepStatusSetRFactor(status, 1);

   return 0;
}

/* --------------------------------------------------------------------
 * -------------------------------------------------------------------- */
int
my_Residual(braid_App        app,
            braid_Vector     ustop,
            braid_Vector     r,
            braid_StepStatus status)
{
   double tstart;             /* current time */
   double tstop;              /* evolve u to this time*/
   int level;
   
   /* Grab status of current time step */
   braid_StepStatusGetTstartTstop(status, &tstart, &tstop);

   /* Grab level */
   braid_StepStatusGetLevel(status, &level);

   /* We need to "trick" the user's manager with the new dt */
   app->man->dt = tstop - tstart;

   /* Now, set up the discretization matrix.  Use the XBraid level to index
    * into the matrix lookup table */
   if( app->dt_A[level] == -1.0 )
   {
      app->nA++;
      app->dt_A[level] = tstop - tstart;

      setUpPhiMatrix( app->man);
      app->Phi[level] = app->man->Phi;

      // Transfer A from pass to here
      setUpGVector(app->man);
      app->G[level] = app->man->G;
   } 

   // Compute residual Ax
   app->man->G = app->G[level];
   app->man->Phi = app->Phi[level];

   comp_res(app->man, ustop->x, r->x, tstart, tstop);

   return 0;
}

/* --------------------------------------------------------------------
 * Create a vector object for a given time point.
 * This function is only called on the finest level.
 * -------------------------------------------------------------------- */
int
my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{
   
   my_Vector * u = (my_Vector *) malloc( sizeof(my_Vector) );
   
   if( t == app->man->tstart )
   {
      /* Sets u_ptr as the initial condition */
      t = 0.0;
   }
   else if (app->use_rand)
   {
      /* This t-value will tell set_initial_condition() below to make u_ptr uniformly random */
      t = -1.0;
   }
   else
   {
      /* Sets u_ptr as an all zero vector*/
      t = 1.0;
   }

   set_initial_condition(app->man, &(u->x), t);
   *u_ptr = u;
   return 0;
}

/* --------------------------------------------------------------------
 * Create a copy of a vector object.
 * -------------------------------------------------------------------- */

int
my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{
   my_Vector *v = (my_Vector *) malloc(sizeof(my_Vector));
   double    *values;
   initialize_vector(app->man, &(v->x));

   /* Set the values. */
   values = (double *) calloc( app->man->grid_size, sizeof(double) );
   HYPRE_SStructVectorGather( u->x );
   HYPRE_SStructVectorGetBoxValues( u->x, 0, app->man->ilower, app->man->iupper, 0, values );
   HYPRE_SStructVectorSetBoxValues( v->x, 0, app->man->ilower, app->man->iupper, 0, values );
   free( values );
   HYPRE_SStructVectorAssemble( v->x );

   *v_ptr = v;
   return 0;
}

/* --------------------------------------------------------------------
 * Destroy vector object.
 * -------------------------------------------------------------------- */
int
my_Free(braid_App    app,
        braid_Vector u)
{
   HYPRE_SStructVectorDestroy( u->x );
   free( u );

   return 0;
}

/* --------------------------------------------------------------------
 * Compute vector sum y = alpha*x + beta*y.
 * -------------------------------------------------------------------- */
int
my_Sum(braid_App    app,
       double       alpha,
       braid_Vector x,
       double       beta,
       braid_Vector y)
{
   int i;
   double *values_x, *values_y;
   
   values_x = (double *) calloc( app->man->grid_size, sizeof(double) );
   values_y = (double *) calloc( app->man->grid_size, sizeof(double) );

   HYPRE_SStructVectorGather( x->x );
   HYPRE_SStructVectorGetBoxValues( x->x, 0, (app->man->ilower), (app->man->iupper), 0, values_x );
   HYPRE_SStructVectorGather( y->x );
   HYPRE_SStructVectorGetBoxValues( y->x, 0, (app->man->ilower), (app->man->iupper), 0, values_y );

   for( i = 0; i < app->man->grid_size; i++ )
   {
      values_y[i] = alpha*values_x[i] + beta*values_y[i];
   }

   HYPRE_SStructVectorSetBoxValues( y->x, 0, (app->man->ilower), (app->man->iupper), 0, values_y );

   free( values_x );
   free( values_y );
   return 0;
}

/* --------------------------------------------------------------------
 * User access routine to spatial solution vectors and allows for user
 * output.  The default XBraid parameter of access_level=1, calls 
 * my_Access only after convergence and at every time point.
 * -------------------------------------------------------------------- */
int
my_Access(braid_App           app,
          braid_Vector        u,
          braid_AccessStatus  astatus)
{
   double     tstart         = (app->man->tstart);
   double     tstop          = (app->man->tstop);
   int        nt             = (app->man->nt);
   
   double     rnorm, disc_err, t;
   int        iter, level, done, index, myid;
   char       filename[255], filename_mesh[255], filename_err[255], filename_sol[255];
  
   /* Retrieve current time from Status Object */
   braid_AccessStatusGetT(astatus, &t);

   /* Retrieve XBraid State Information from Status Object */
   MPI_Comm_rank(app->comm_x, &myid);
   braid_AccessStatusGetTILD(astatus, &t, &iter, &level, &done);
   braid_AccessStatusGetResidual(astatus, &rnorm);

   if(level == 0)
   {
      /* Print discretization error to screen for only final time */
      index = ((t - tstart) / ((tstop - tstart)/nt) + 0.1);
      compute_disc_err(app->man, u->x, t, app->e, &disc_err);
      if( (t == app->man->tstop) && myid == 0 ) 
      {
         printf("\n  Discr. error         = %1.5e\n", disc_err);
         printf("\n  my_Access():  Braid iter %d,  discr. error at final time:  %1.4e\n", iter, disc_err);
         print_L2_norm(app->man, u->x);
      }
   }

   return 0;
}

/* --------------------------------------------------------------------
 * Compute norm of a spatial vector 
 * -------------------------------------------------------------------- */
int
my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
   *norm_ptr = L2_norm(app->man, u->x);
   return 0;
}

/* --------------------------------------------------------------------
 * Return buffer size needed to pack one spatial braid_Vector.  Here the
 * vector contains one double at every grid point and thus, the buffer 
 * size is the number of grid points.
 * -------------------------------------------------------------------- */
int
my_BufSize(braid_App           app,
           int                 *size_ptr,
           braid_BufferStatus  status)
{
    *size_ptr = (app->man->grid_size)*sizeof(double);
    return 0;
}

/* --------------------------------------------------------------------
 * Pack a braid_Vector into a buffer.
 * -------------------------------------------------------------------- */
int
my_BufPack(braid_App           app,
           braid_Vector        u,
           void                *buffer,
           braid_BufferStatus  status)
{
   double *dbuffer = buffer;
   
   /* Place the values in u into the buffer */
   HYPRE_SStructVectorGather( u->x );
   HYPRE_SStructVectorGetBoxValues( u->x, 0, app->man->ilower, app->man->iupper, 0, &(dbuffer[0]) );

   /* Return the number of bytes actually packed */
   braid_BufferStatusSetSize( status, (app->man->grid_size)*sizeof(double) );
   return 0;
}

/* --------------------------------------------------------------------
 * Unpack a buffer and place into a braid_Vector
 * -------------------------------------------------------------------- */
int
my_BufUnpack(braid_App           app,
             void                *buffer,
             braid_Vector        *u_ptr,
             braid_BufferStatus  status)
{
   double    *dbuffer = buffer;
   my_Vector *u       = (my_Vector *) malloc( sizeof(my_Vector) );
   
   /* Set the values in u based on the values in the buffer */
   initialize_vector(app->man, &(u->x));
   HYPRE_SStructVectorSetBoxValues( u->x, 0, app->man->ilower, app->man->iupper, 0, &(dbuffer[0]) );
   HYPRE_SStructVectorAssemble( u->x );
   *u_ptr = u;

   return 0;
}

/* --------------------------------------------------------------------
 * Main driver
 * -------------------------------------------------------------------- */
int main (int argc, char *argv[])
{
   /* Declare variables -- variables explained when they are set below */
   int *runtime_max_iter_global          = NULL;
   MPI_Comm comm                         = MPI_COMM_WORLD;
   MPI_Comm comm_x, comm_t;
   int i, arg_index, myid, num_procs;
   
   braid_Core    core;
   my_App       *app = (my_App *) malloc(sizeof(my_App));
   double tol, rtol, mystarttime, myendtime, mytime, maxtime, cfl;
   int run_wrapper_tests, correct1, correct2, correct3, correct4, correct5, correct6 ;
   int print_level, access_level, max_nA, nA_max, max_levels, skip, min_coarse;
   int nrelax, nrelax0, cfactor, cfactor0, max_iter, fmg, res, storage, tnorm;
   int fullrnorm, use_seq_soln;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank( comm, &myid );

   /* Default parameters */
   app->man  = (simulation_manager *) malloc(sizeof(simulation_manager));
   app->man->nx              = 8;             /* number of points in the x-dim */
   app->man->ny              = 8;             /* number of points in the y-dim */
   app->man->nt              = 512;           /* number of time steps */
   app->man->dim_x           = 2;             /* Two dimensional problem */
   app->man->K_x             = 2.0;           /* Diffusion coefficient */
   app->man->K_y             = 0.0;           /* Diffusion coefficient */
   app->man->p_beta          = 0.95;
   app->man->p_gamma         = 0.65;
   app->man->px              = 1;
   app->man->py              = 1;
   app->man->tstart          = 0.0;           /* global start time */
   app->man->tstop           = 1.0;
   app->man->object_type     = HYPRE_STRUCT;  /* Hypre Struct interface is used for solver */
   app->man->vartype         = HYPRE_SSTRUCT_VARIABLE_CELL;
   app->man->grid_size       = (app->man->nx - 1) * (app->man->ny - 1);
   
   /* Default XBraid parameters */
   max_levels          = 2;               /* Max levels for XBraid solver */
   skip                = 1;               /* Boolean, whether to skip all work on first down cycle */
   min_coarse          = 2;               /* Minimum possible coarse grid size */
   nrelax0             = 1;               /* Number of CF relaxations only for level 0 -- overrides nrelax */
   tol                 = 1.0e-16;         /* Halting tolerance */
   tnorm               = 2;               /* Halting norm to use (see docstring below) */
   cfactor0            = 16;              /* Coarsening factor for only level 0 -- overrides cfactor */
   max_iter            = 3;            /* Maximum number of iterations */
   res                 = 0;               /* Boolean, if 1, use my residual */
   storage             = 0;               /* Full storage on levels >= 'storage' */
   print_level         = 3;               /* Level of XBraid printing to the screen */
   access_level        = 1;               /* Frequency of calls to access routine: 1 is for only after simulation */
   use_seq_soln        = 1;               /* Use the solution from sequential time stepping as the initial guess */

   /* Other parameters specific to parallel in time */
   app->use_rand       = 1;               /* If 1, use a random initial guess, else use a zero initial guess */
   app->pt             = 1;               /* Number of processors in time */

   /* Check the processor grid (px x py x pt = num_procs?). */
   MPI_Comm_size( comm, &num_procs );

   /* Create communicators for the time and space dimensions */
   braid_SplitCommworld(&comm, (app->man->px)*(app->man->py), &comm_x, &comm_t);
   app->man->comm = comm_x;
   app->comm = comm;
   app->comm_t = comm_t;
   app->comm_x = comm_x;

   /* Determine position (pi, pj)  in the 2D processor grid, 
    * 0 <= pi < px,   0 <= pj < py */
   MPI_Comm_rank( comm_x, &myid );

   /* Define the 2D block of the global grid owned by this processor, that is
    * (ilower[0], iupper[0]) x (ilower[1], iupper[1])
    * defines the piece of the global grid owned by this processor. */ 
   app->man->ilower[0] = 1;
   app->man->iupper[0] = app->man->nx - 1;
   app->man->ilower[1] = 1;
   app->man->iupper[1] = app->man->ny - 1;

   /* Compute grid spacing. */
   app->man->dx = 1.0 / (app->man->nx);
   app->man->dy = 1.0 / (app->man->ny);

   /* Set time-step size */
   app->man->dt = (app->man->tstop - app->man->tstart) / app->man->nt;
   /* Now using dt, compute the final time, tstop value */
   
   /* Set up the variable type, grid, stencil and matrix graph. */
   setUp2Dgrid( comm_x, &(app->man->grid_x), app->man->dim_x,
                app->man->ilower, app->man->iupper, app->man->vartype );

   setUpForcingSpace(app->man);

   /* Allocate items of app, especially A and dt_A which are arrays of discretization 
    * matrices which one matrix and corresponding dt value for each XBraid level */
   max_nA = max_levels; /* use generous value to keep code simple */
   initialize_vector(app->man, &(app->e));
   app->G = (HYPRE_IJVector*) calloc( max_nA, sizeof(HYPRE_IJVector));
   app->Phi = (HYPRE_IJMatrix*) calloc( max_nA, sizeof(HYPRE_IJMatrix));
   
   app->dt_A = (double*) calloc( max_nA, sizeof(double) );
   for( i = 0; i < max_nA; i++ ) 
   {
      app->dt_A[i] = -1.0;
   }
   app->nA = 0;
   app->max_nA = max_nA;

   app->runtime_max_iter = (int*) calloc( max_nA,  sizeof(int) );
   for( i = 0; i < max_nA; i++ )
   {
      app->runtime_max_iter[i] = 0;
   }

   // Run XBraid simulation 
   mystarttime = MPI_Wtime();
   braid_Init(comm, comm_t, app->man->tstart, app->man->tstop, app->man->nt, 
               app, my_Step, my_Init, my_Clone, my_Free, my_Sum, 
               my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);
   
   // Set Braid parameters 
   braid_SetSkip( core, skip );
   braid_SetMaxLevels( core, max_levels );
   braid_SetMinCoarse( core, min_coarse );
   braid_SetPrintLevel( core, print_level);
   braid_SetAccessLevel( core, access_level);
   braid_SetSeqSoln(core, use_seq_soln);

   braid_SetAbsTol(core, tol );
   braid_SetNRelax(core,  -1, nrelax0);
   braid_SetTemporalNorm(core, tnorm);
   braid_SetCFactor(core,  -1, cfactor0);
   braid_SetMaxIter(core, max_iter);

//   braid_SetRefine(core, 1);
//   braid_SetMaxRefinements(core, 2);
   braid_SetRelaxOnlyCG(core, 0);
   braid_SetLyapunovEstimation(core, 1, 1, 1);
//   braid_SetResidualComputation(core, 0);

   if (res) 
   {
      braid_SetResidual(core, my_Residual);
   }
   if (storage >= -2) 
   {
      braid_SetStorage(core, storage);
   }

   MPI_Comm_rank( comm, &myid );
   if( myid == 0 ) 
   {
      printf("\n  --------------------- \n");
      printf("  Begin simulation \n");
      printf("  --------------------- \n\n");
   }
   
   // This call "Drives" or runs the simulation -- woo hoo!
   braid_Drive(core);
   
   // Compute run time
   myendtime = MPI_Wtime();
   mytime    = myendtime - mystarttime;
   MPI_Reduce( &mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm );

   // Determine maximum number of iterations for hypre PFMG spatial solves at each time level 
   MPI_Allreduce( &(app->nA), &nA_max, 1, MPI_INT, MPI_MAX, comm ); 
   runtime_max_iter_global = (int*) malloc( nA_max*sizeof(int) );
   for( i = 0; i < nA_max; i++ )
   {
      MPI_Allreduce( &(app->runtime_max_iter[i]), &runtime_max_iter_global[i], 1, MPI_INT, MPI_MAX, comm );
   }

   if( myid == 0 ) 
   {
      printf("  --------------------- \n");
      printf("  End simulation \n");
      printf("  --------------------- \n\n");
   
      printf("  Time step size                    %1.2e\n", app->man->dt);
      printf("  Spatial grid size:                %d,%d\n", app->man->nx, app->man->ny);
      printf("  Spatial mesh width (dx,dy):      (%1.2e, %1.2e)\n", app->man->dx, app->man->dy);
      printf("  Run time:                      %1.2e\n", maxtime);
      printf("\n");
   }  

   braid_Destroy(core);

   // Free app->man structures
   HYPRE_SStructGridDestroy( app->man->grid_x );
//   HYPRE_IJMatrixDestroy( app->man->A );
   free( app->man->Fxy );
   free( app->man );

   // Free app-> structures
   for( i = 0; i < app->nA; i++ ) 
   {
      HYPRE_IJVectorDestroy( app->G[i] );
      HYPRE_IJMatrixDestroy( app->Phi[i] );
   }
   HYPRE_SStructVectorDestroy( app->e );
   free( app->dt_A );
   free( app->G );
   free( app->Phi );
   free( app->runtime_max_iter );
   free( app );
   
   /* Other Free */
   if( runtime_max_iter_global != NULL) 
   {
      free( runtime_max_iter_global );
   }
   MPI_Comm_free( &comm_x );
   MPI_Comm_free( &comm_t );

   MPI_Finalize();
   return 0;
}

