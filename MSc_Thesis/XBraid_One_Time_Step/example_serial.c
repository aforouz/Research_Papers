#include "example_lib.c"

int main (int argc, char *argv[])
{
   /* Declare variables -- variables explained when they are set below */
   MPI_Comm    comm;
   HYPRE_SStructVector u, e;
   simulation_manager * man;

   int object_type = HYPRE_STRUCT;
   int i, arg_index, myid, num_procs, iters_taken, max_iters_taken;
   int ndim, nx, ny, nlx, nly, nt, forcing, ilower[2], iupper[2];
   double p_beta, p_gamma, K_x, K_y, tstart, tstop, dx, dy, dt, tol;
   double myendtime, mystarttime, mytime, maxtime;
   double disc_err, max_disc_err, max_disc_err_time;
   int max_iter, px, py, pi, pj, max_disc_err_iter;
   char filename[255], filename_mesh[255], filename_err[255], filename_sol[255];

   /* Initialize MPI */
   MPI_Init(&argc, &argv);

   /* Default parameters. */
   comm                = MPI_COMM_WORLD;
   ndim                = 2;       /* Two dimensional problem */
   K_x                 = 2.0;     /* Diffusion coefficient */
   K_y                 = 0.5;
   p_beta              = 0.6;
   p_gamma             = 0.7;
   nx                  = 8;      /* number of points in the x-dim */
   ny                  = 8;      /* number of points in the y-dim */
   tstart              = 0.0;     /* global start time */
   nt                  = 512;      /* number of time steps */

   MPI_Comm_rank( comm, &myid );
   MPI_Comm_size( comm, &num_procs );

   ilower[0] = 1;
   iupper[0] = nx-1;
   ilower[1] = 1;
   iupper[1] = ny-1;

   // Compute grid spacing.
   dx = 1.0 / nx;
   dy = 1.0 / ny;

   // Set time-step size.
   dt = 1.0 / nt;
   
   // Now using dt, compute the final time, tstop value
   tstop =  tstart + nt*dt;

   /* -----------------------------------------------------------------
    * Set up the manager 
    * ----------------------------------------------------------------- */
   man               = (simulation_manager *) malloc(sizeof(simulation_manager));
   man->comm         = comm;
   man->p_beta       = p_beta;
   man->p_gamma      = p_gamma;
   man->K_x          = K_x;
   man->K_y          = K_y;
   man->dim_x        = ndim;
   man->nx           = nx;
   man->ny           = ny;
   man->grid_size    = (nx-1)*(ny-1);
   man->tstart       = tstart;
   man->tstop        = tstop;
   man->nt           = nt;
   man->dx           = dx;
   man->dy           = dy;
   man->dt           = dt;
   man->ilower[0]    = ilower[0];
   man->ilower[1]    = ilower[1];
   man->iupper[0]    = iupper[0];
   man->iupper[1]    = iupper[1];
   man->object_type  = object_type;
   man->max_iter     = max_iter;
   man->tol          = tol;
   
   /* Set up the variable type, grid, stencil and matrix graph. */
   man->vartype           = HYPRE_SSTRUCT_VARIABLE_CELL;
   setUp2Dgrid( comm, &(man->grid_x), man->dim_x,
                man->ilower, man->iupper, man->vartype);

   /* Set up initial state vector */
   set_initial_condition(man, &u, 0.0);

   /* Set up error vector */
   initialize_vector(man, &e);
 
   /* Set up the matrix */
   setUpPhiMatrix( man );
   
   setUpForcingSpace(man);
   setUpGVector( man );   

   if( myid == 0 ) {
      printf("\n  --------------------- \n");
      printf("  Begin simulation \n");
      printf("  --------------------- \n\n");
   }
   
   /* Run a simulation */
   mystarttime = MPI_Wtime();
   max_iters_taken = -1;
   max_disc_err = -1;
   tstart = 0.0;
   tstop = tstart + dt;


   for(i = 0; i < man->nt; i++)
   {
      if( myid == 0 ) 
      {
         //if( i % 50 == 0)  
         printf("  Taking iteration %d...\n", i);
      }

      /* Take Step */
      take_step(man, u, NULL, u, tstart, tstop);
      if( i < man->nt-1)
      {
         tstart = tstop;
         tstop = tstop + dt;
      }

      /* Output */
      compute_disc_err(man, u, tstop, e, &disc_err);

      /* Store PFMG iters taken and maximum discretization error */
      max_iters_taken = max_i(max_iters_taken, 100);
      max_disc_err = max_d(max_disc_err, disc_err);
      if( max_disc_err == disc_err) 
      {
         max_disc_err_iter = i;
         max_disc_err_time = tstop;
      }

   }
   myendtime = MPI_Wtime();

   /* Print some additional statistics */
   mytime    = myendtime - mystarttime;
   MPI_Reduce( &mytime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, comm );
   if( myid == 0 )
   {
      printf("\n  --------------------- \n");
      printf("  End simulation \n");
      printf("  --------------------- \n\n");
      
      printf("  Start time                    %1.5e\n", 0.0);
      printf("  Stop time                     %1.5e\n", tstop);
      printf("  Time step size                %1.5e\n", man->dt);
      printf("  Time steps taken:             %d\n\n", i);
      printf("  Spatial grid size:            %d,%d\n", man->nx, man->ny);
      printf("  Spatial mesh width (dx,dy):  (%1.2e, %1.2e)\n", man->dx, man->dy);           
      printf("  Run time:                     %1.2e\n", maxtime);
      printf("  Max PCG Iterations:          %d\n", max_iters_taken);
      printf("  Discr. error at final time:   %1.4e\n", disc_err);
      printf("  Max discr. error:             %1.3e\n",max_disc_err);
      printf("     found at iteration:        %d\n", max_disc_err_iter);
      printf("     found at time:             %1.2e\n\n", max_disc_err_time);
   }      

   print_L2_norm(man, u);


   /* Visualize final error */

   /* Free memory */
   HYPRE_SStructVectorDestroy( u );
   HYPRE_SStructVectorDestroy( e );
   HYPRE_SStructGridDestroy( man->grid_x );
   HYPRE_IJMatrixDestroy( man->Phi );
   HYPRE_IJVectorDestroy( man->G );
   free(man->Fxy);
   free(man);
   
   MPI_Finalize();
   return 0;
}
