#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "_hypre_sstruct_mv.h"
#include "braid.h"

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
   HYPRE_SStructVector          e;                /* temporary vector used for error computations */
   int                     nA;               /* number of discr. matrices that have been created */
   int                     max_nA;           /* max nA value allowed */
   HYPRE_IJMatrix         *A;                /* nA sized array of discr. matrices (one per time level) */
   HYPRE_IJMatrix         *B;
   double                 *dt_A;             /* nA sized array of time step sizes for each  matrix  */
   HYPRE_Solver           *solver;           /* nA sized array of solvers (one per time level) */
   int                     use_rand;         /* binary, use random initial guess (1) or zero initial guess (0) */
   int                    *runtime_max_iter; /* runtime info for the max number of spatial solve iterations at each level */
   int                    *max_iter_x;       /* length 2 array of expensive and cheap max PFMG iters for spatial solves*/
} my_App;

int print_app(my_App * app)
{
   int myid,i;
   MPI_Comm_rank( app->comm, &myid );
   printf("\n\nmyid:  %d,  App contents:\n", myid);
   printf("myid:  %d,  pt:            %d\n", myid, app->pt);
   printf("myid:  %d,  use_rand:      %d\n", myid, app->use_rand);
   printf("myid:  %d,  nA:            %d\n", myid, app->nA);
   printf("myid:  %d,  max_iter_x[0]: %d\n", myid, app->max_iter_x[0]);
   printf("myid:  %d,  max_iter_x[1]: %d\n", myid, app->max_iter_x[1]);
   for(i = 0; i < app->nA; i++)
   {
      printf("myid:  %d,  runtime_max_iter[%d]: %d\n", myid, i, app->runtime_max_iter[i]);
   }
   for(i = 0; i < app->nA; i++)
   {
      printf("myid:  %d,  dt_A[%d]:           %1.2e\n", myid, i, app->dt_A[i]);
   }
   printf("\nmyid:  %d,  Note that some object members like comm, comm_t, comm_x, man, A and solver cannot be printed\n\n", myid);
   return 0;
}

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
      app->dt_A[level] = tstop-tstart;
      setUpImplicitMatrix( app->man);
//      HYPRE_IJMatrixPrint(app->man->A, "txt/A");
      app->A[level] = app->man->A;
//      HYPRE_IJMatrixPrint(app->A[level], "txt/levelA");
      setUpExplicitMatrix( app->man);
      app->B[level] = app->man->B;
      
      /* Set up the PFMG solver using u->x as dummy vectors. */
      setUpStructSolver( app->man, u->x, u->x );
      app->solver[level] = app->man->solver;
   } 


   /* Time integration to next time point: Solve the system Ax = b.
    * First, "trick" the user's manager with the right matrix and solver */ 
   app->man->A = app->A[level];
   app->man->B = app->B[level];
   app->man->solver = app->solver[level];

   /* Use level specific max_iter */
   if( level == 0 )
   {
      app->man->max_iter = app->max_iter_x[0];
   }
   else
   {
      app->man->max_iter = app->max_iter_x[1];
   }

   /* Take step */
   if (fstop == NULL)
   {
      bstop = NULL;
   }
   else
   {
      bstop = fstop->x;
   }
   take_step(app->man, ustop->x, bstop, u->x, tstart, tstop, &iters_taken);

   /* Store iterations taken */
   app->runtime_max_iter[level] = max_i( (app->runtime_max_iter[level]), iters_taken);

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
   if( app->dt_A[level] == -1.0 ){
      app->nA++;
      app->dt_A[level] = tstop-tstart;

      setUpImplicitMatrix( app->man);
      app->A[level] = app->man->A;
      setUpExplicitMatrix( app->man);
      app->B[level] = app->man->B;
      
      /* Set up the PFMG solver using r->x as dummy vectors. */
      setUpStructSolver( app->man, r->x, r->x );
      app->solver[level] = app->man->solver;
   } 

   /* Compute residual Ax */
   app->man->A = app->A[level];
   app->man->B = app->B[level];
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
   values = (double *) calloc( (app->man->nlx)*(app->man->nly), sizeof(double) );
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
   
   values_x = (double *) calloc( (app->man->nlx)*(app->man->nly), sizeof(double) );
   values_y = (double *) calloc( (app->man->nlx)*(app->man->nly), sizeof(double) );

   HYPRE_SStructVectorGather( x->x );
   HYPRE_SStructVectorGetBoxValues( x->x, 0, (app->man->ilower), (app->man->iupper), 0, values_x );
   HYPRE_SStructVectorGather( y->x );
   HYPRE_SStructVectorGetBoxValues( y->x, 0, (app->man->ilower), (app->man->iupper), 0, values_y );

   for( i = 0; i < (app->man->nlx)*(app->man->nly); i++ )
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

      }
   }
   if (t == tstop)
   {
      print_L2_norm(app->man, u->x);
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
   norm(u->x, norm_ptr);
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
    *size_ptr = (app->man->nlx)*(app->man->nly)*sizeof(double);
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
   HYPRE_SStructVectorGetBoxValues( u->x, 0, app->man->ilower, 
                          app->man->iupper, 0, &(dbuffer[0]) );

   /* Return the number of bytes actually packed */
   braid_BufferStatusSetSize( status, (app->man->nlx)*(app->man->nly)*sizeof(double) );
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
   HYPRE_SStructVectorSetBoxValues( u->x, 0, app->man->ilower, 
                           app->man->iupper, 0, &(dbuffer[0]) );
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
   double tol, mystarttime, myendtime, mytime, maxtime, cfl;
   int run_wrapper_tests, correct1, correct2;
   int print_level, access_level, max_nA, nA_max, max_levels, skip, min_coarse;
   int nrelax, nrelax0, cfactor, cfactor0, max_iter, fmg, res, storage, tnorm;
   int fullrnorm, use_seq_soln;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank( comm, &myid );

   /* Default parameters */
   app->man  = (simulation_manager *) malloc(sizeof(simulation_manager));
   app->man->px              = 1;             /* my processor number in the x-direction, px*py=num procs in space */
   app->man->py              = 1;             /* my processor number in the y-direction, px*py=num procs in space */
   app->man->nx              = 9;            /* number of points in the x-dim */
   app->man->ny              = 9;            /* number of points in the y-dim */
   app->man->nt              = 512;            /* number of time steps */
   app->man->tol             = 1.0e-09;       /* PFMG halting tolerance */
   app->man->dim_x           = 2;             /* Two dimensional problem */
   app->man->K_x             = 2.0;           /* Diffusion coefficient */
   app->man->K_y             = 0.5;           /* Diffusion coefficient */
   app->man->p_beta          = 0.6;
   app->man->p_gamma         = 0.7;
   app->man->nlx             = 9;            /* number of point ~local~ to this processor in the x-dim */
   app->man->nly             = 9;            /* number of point ~local~ to this processor in the y-dim */
   app->man->tstart          = 0.0;           /* global start time */
   app->man->object_type     = HYPRE_STRUCT;  /* Hypre Struct interface is used for solver */
   app->man->vartype         = HYPRE_SSTRUCT_VARIABLE_CELL;
   app->man->grid_size       = (app->man->nx) * (app->man->ny);
   
   /* Default XBraid parameters */
   max_levels          = 4;              /* Max levels for XBraid solver */
   skip                = 1;               /* Boolean, whether to skip all work on first down cycle */
   min_coarse          = 2;               /* Minimum possible coarse grid size */
   nrelax              = 1;               /* Number of CF relaxation sweeps on all levels */
   nrelax0             = -1;              /* Number of CF relaxations only for level 0 -- overrides nrelax */
   tol                 = 1.0e-09;         /* Halting tolerance */
   tnorm               = 2;               /* Halting norm to use (see docstring below) */
   cfactor             = 16;               /* Coarsening factor */
   cfactor0            = -1;              /* Coarsening factor for only level 0 -- overrides cfactor */
   max_iter            = 100;             /* Maximum number of iterations */
   fmg                 = 0;               /* Boolean, if 1, do FMG cycle.  If 0, use a V cycle */
   res                 = 0;               /* Boolean, if 1, use my residual */
   storage             = -1;              /* Full storage on levels >= 'storage' */
   print_level         = 2;               /* Level of XBraid printing to the screen */
   access_level        = 1;               /* Frequency of calls to access routine: 1 is for only after simulation */
   fullrnorm           = 0;               /* Do not compute full residual from user routine each iteration */
   use_seq_soln        = 0;               /* Use the solution from sequential time stepping as the initial guess */

   /* Other parameters specific to parallel in time */
   app->use_rand       = 0;               /* If 1, use a random initial guess, else use a zero initial guess */
   app->pt             = 1;               /* Number of processors in time */
   app->max_iter_x     = (int*) malloc( 2*sizeof(int) );
   app->max_iter_x[0]  = 50;              /* Maximum number of PFMG iters (the spatial solver from hypre) on XBraid level 0 */
   app->max_iter_x[1]  = 50;              /* Maximum number of PFMG iter on all coarse XBraid levels */

   /* Check the processor grid (px x py x pt = num_procs?). */
   MPI_Comm_size( comm, &num_procs );
   if( ((app->man->px)*(app->man->py)*(app->pt)) != num_procs)
   {
       if( myid == 0 )
       {
           printf("Error: px x py x pt does not equal the number of processors!\n");
       }
       MPI_Finalize();
       return (0);
   }

   /* Create communicators for the time and space dimensions */
   braid_SplitCommworld(&comm, (app->man->px)*(app->man->py), &comm_x, &comm_t);
   app->man->comm = comm_x;
   app->comm = comm;
   app->comm_t = comm_t;
   app->comm_x = comm_x;

   /* Determine position (pi, pj)  in the 2D processor grid, 
    * 0 <= pi < px,   0 <= pj < py */
   MPI_Comm_rank( comm_x, &myid );
   app->man->pi = myid % (app->man->px);
   app->man->pj = ( (myid - app->man->pi)/(app->man->px) ) % (app->man->py);

   /* Define the 2D block of the global grid owned by this processor, that is
    * (ilower[0], iupper[0]) x (ilower[1], iupper[1])
    * defines the piece of the global grid owned by this processor. */ 
   GetDistribution_x( (app->man->nx), (app->man->px), (app->man->pi), 
                      &(app->man->ilower[0]), &(app->man->iupper[0]) );
   GetDistribution_x( (app->man->ny), (app->man->py), (app->man->pj), 
                      &(app->man->ilower[1]), &(app->man->iupper[1]) );

   /* Determine local problem size. */
   app->man->nlx = app->man->iupper[0] - app->man->ilower[0] + 1;
   app->man->nly = app->man->iupper[1] - app->man->ilower[1] + 1;

   /* Compute grid spacing. */
   app->man->dx = 1.0 / (app->man->nx - 1);
   app->man->dy = 1.0 / (app->man->ny - 1);

   /* Set time-step size, noting that the CFL number definition 
    *     K*(dt/dx^2 + dt/dy^2) = CFL
    * implies that dt is equal to 
    *     dt = ( CFL dx^2 dy^2) / ( K(dx^2 + dy^2)) */
   app->man->dt = 1.0 / app->man->nt;
   /* Now using dt, compute the final time, tstop value */
   app->man->tstop =  app->man->tstart + app->man->nt*app->man->dt;

   /* Set up the variable type, grid, stencil and matrix graph. */
   setUp2Dgrid( comm_x, &(app->man->grid_x), app->man->dim_x,
                app->man->ilower, app->man->iupper, app->man->vartype, 1 );

   setUpForcingSpace(app->man);
   /* Allocate items of app, especially A and dt_A which are arrays of discretization 
    * matrices which one matrix and corresponding dt value for each XBraid level */
   max_nA = 512*max_levels; /* use generous value to keep code simple */
   initialize_vector(app->man, &(app->e));
   app->A = (HYPRE_IJMatrix*) calloc( max_nA, sizeof(HYPRE_IJMatrix));
   app->B = (HYPRE_IJMatrix*) calloc( max_nA, sizeof(HYPRE_IJMatrix));
   app->dt_A = (double*) calloc( max_nA, sizeof(double) );
   for( i = 0; i < max_nA; i++ ) 
   {
      app->dt_A[i] = -1.0;
   }
   app->nA = 0;
   app->max_nA = max_nA;

   /* Allocate memory for array of solvers. */
   app->solver = (HYPRE_Solver*) malloc( max_nA*sizeof(HYPRE_Solver));

   /* Array for tracking runtime iteration counts of PFMG. */
   app->runtime_max_iter = (int*) calloc( max_nA,  sizeof(int) );
   for( i = 0; i < max_nA; i++ )
   {
      app->runtime_max_iter[i] = 0;
   }


   /* Run XBraid simulation */

   mystarttime = MPI_Wtime();
   braid_Init(comm, comm_t, app->man->tstart, app->man->tstop, app->man->nt, 
               app, my_Step, my_Init, my_Clone, my_Free, my_Sum, 
               my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &core);
   
   /* Set Braid parameters */
   braid_SetSkip( core, skip );
   braid_SetMaxLevels( core, max_levels );
   braid_SetMinCoarse( core, min_coarse );
   braid_SetPrintLevel( core, print_level);
   braid_SetAccessLevel( core, access_level);
   braid_SetNRelax(core, -1, nrelax);
   braid_SetSeqSoln(core, use_seq_soln);
   if (nrelax0 > -1) 
   {
      braid_SetNRelax(core,  0, nrelax0);
   }
   braid_SetAbsTol(core, tol/
      sqrt( (app->man->dx)*(app->man->dy)*(app->man->dt)) );
   braid_SetTemporalNorm(core, tnorm);
   braid_SetCFactor(core, -1, cfactor);
   if (fullrnorm) 
   {
      braid_SetFullRNormRes(core, my_Residual);        
   }
   if( cfactor0 > 0 ) 
   {
      braid_SetCFactor(core,  0, cfactor0);
   }
   braid_SetMaxIter(core, max_iter);
   if (fmg) 
   {
      braid_SetFMG(core);
   }
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
   
   /* This call "Drives" or runs the simulation -- woo hoo! */
   braid_Drive(core);
   
   /* Compute run time */
   myendtime = MPI_Wtime();
   mytime    = myendtime - mystarttime;
   MPI_Reduce( &mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm );

   /* Determine maximum number of iterations for hypre PFMG spatial solves at each time level */
   MPI_Allreduce( &(app->nA), &nA_max, 1, MPI_INT, MPI_MAX, comm ); 
   runtime_max_iter_global = (int*) malloc( nA_max*sizeof(int) );
   for( i = 0; i < nA_max; i++ )
   {
      MPI_Allreduce( &(app->runtime_max_iter[i]), 
                     &runtime_max_iter_global[i], 1, MPI_INT, MPI_MAX, comm );
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
      printf("\n   Level   Max PCG Iters\n");
      printf("  -----------------------\n");
      for(i = 0; i < nA_max; i++)
      {
         printf("     %d           %d\n", i, runtime_max_iter_global[i]);
      }
      printf("\n");
   }  

   braid_Destroy(core);

   /* Free app->man structures */
   HYPRE_SStructGridDestroy( app->man->grid_x );

   /* Free app-> structures */
   for( i = 0; i < app->nA; i++ ) 
   {
      HYPRE_IJMatrixDestroy( app->A[i] );
      HYPRE_IJMatrixDestroy( app->B[i] );
      HYPRE_ParCSRPCGDestroy( app->solver[i] );
   }
//   printf("HERE!\n");
   HYPRE_SStructVectorDestroy( app->e );
   free( app->man );
   free( app->dt_A );
   free( app->A );
   free( app->B );
   free( app->solver );
   free( app->runtime_max_iter );
   free( app->max_iter_x );
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

