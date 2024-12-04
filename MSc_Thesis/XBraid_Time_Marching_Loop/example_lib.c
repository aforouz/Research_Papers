#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_utilities.h"
#include "HYPRE_sstruct_ls.h"
#include "_hypre_sstruct_mv.h"
#include "braid.h"

#ifdef M_PI
   #define PI M_PI
#else
   #define PI 3.14159265358979
#endif

typedef struct _simulation_manager_struct {
   MPI_Comm                comm;
   double                  p_beta, p_gamma;
   double                  K_x, K_y;
   int                     dim_x;
   int                     nlx, nly;
   int                     nx, ny;
   int                     grid_size;
   double                  tstart;
   double                  tstop;
   int                     nt;
   double                  dx, dy;
   double                  dt;
   HYPRE_SStructVariable   vartype;
   HYPRE_SStructGrid       grid_x;
   HYPRE_IJMatrix          A;
   HYPRE_IJMatrix          B;
   double                  *Fxy;
   int                     px, py;
   int                     pi, pj;
   int                     ilower[2], iupper[2];
   int                     object_type;
   HYPRE_Solver            solver;
   int                     max_iter;
   double                  tol;
} simulation_manager;

/* --------------------------------------------------------------------
 * Define max, min functions
 * -------------------------------------------------------------------- */
int max_i( int a, int b )
{
  return (a >= b ? a : b );
}
int min_i( int a, int b )
{
  return (a <= b ? a : b );
}
double max_d( double a, double b )
{
  return (a >= b ? a : b );
}
double min_d( double a, double b )
{
  return (a <= b ? a : b );
}

/* --------------------------------------------------------------------
 * Grab spatial discretization information from a manager
 * -------------------------------------------------------------------- */
int grab_manager_spatial_info(simulation_manager     *man,
                              int                     ilower[2],
                              int                     iupper[2],
                              int                    *nlx,
                              int                    *nly,
                              int                    *nx,
                              int                    *ny,
                              int                    *px,
                              int                    *py,
                              int                    *pi,
                              int                    *pj,
                              double                 *dx,
                              double                 *dy)
{
   (*nlx) = man->nlx;
   (*nly) = man->nly;
   (*nx)  = man->nx;
   (*ny)  = man->ny;
   (*dx)  = man->dx;
   (*dy)  = man->dy;

   (*px)  = man->px;
   (*py)  = man->py;
   (*pi)  = man->pi;
   (*pj)  = man->pj;

   ilower[0] = man->ilower[0];
   ilower[1] = man->ilower[1];
   iupper[0] = man->iupper[0];
   iupper[1] = man->iupper[1];

   return 0;
}

/* --------------------------------------------------------------------
 * Compute a processor's 2D block of the global 2D grid
 * -------------------------------------------------------------------- */
int GetDistribution_x( int    npoints,
                       int    nprocs,
                       int    proc,
                       int   *ilower_ptr,
                       int   *iupper_ptr )
{
   int  ilower, iupper;
   int  quo, rem, p;

   quo = npoints/nprocs;
   rem = npoints%nprocs;

   p = proc;
   ilower = p*quo + (p < rem ? p : rem);
   p = proc+1;
   iupper = p*quo + (p < rem ? p : rem) - 1;

   *ilower_ptr = ilower;
   *iupper_ptr = iupper;

   return 0;
}


/* --------------------------------------------------------------------
 * Initial condition 
 * -------------------------------------------------------------------- */
double U0(double x, double y)
{
    return 10 * pow(x * (1 - x), 2) * pow(y * (1 - y), 2);
}

/* --------------------------------------------------------------------
 * Exact solution 
 * -------------------------------------------------------------------- */
double U_exact(simulation_manager *man, double x, double y, double t)
{
   return exp(-t) * 10 * pow(x * (1 - x), 2) * pow(y * (1 - y), 2);
}

/* --------------------------------------------------------------------
 * Boundary condition: zero Dirichlet condition for now
 * -------------------------------------------------------------------- */
double B0(double x, double y)
{
    return 0.0;
}

/* --------------------------------------------------------------------
 * Forcing term: b(x,y,t) = -sin(x)*sin(y)*( sin(t) - 2*K*cos(t) )
 * -------------------------------------------------------------------- */
double F_space(double x, double y, double K_x, double K_y, double p_beta, double p_gamma)
{
   return -pow(x * (1 - x), 2) * pow(y * (1 - y), 2) + 
   K_x * pow(y - pow(y, 2), 2) / cos(p_beta * PI) * 
   ( 
   (pow(x, 2 - 2*p_beta) + pow(1 - x, 2 - 2*p_beta)) / tgamma(3 - 2*p_beta) 
   -6 * (pow(x, 3 - 2*p_beta) + pow(1 - x, 3 - 2*p_beta)) / tgamma(4 - 2*p_beta) 
   +12 * (pow(x, 4 - 2*p_beta) + pow(1 - x, 4 - 2*p_beta)) / tgamma(5 - 2*p_beta) 
   ) 
      + 
   K_y * pow(x - pow(x, 2), 2) / cos(p_gamma * PI) * 
   ( 
   (pow(y, 2 - 2*p_gamma) + pow(1 - y, 2 - 2*p_gamma)) / tgamma(3 - 2*p_gamma) 
   -6 * (pow(y, 3 - 2*p_gamma) + pow(1 - y, 3 - 2*p_gamma)) / tgamma(4 - 2*p_gamma) 
   +12 * (pow(y, 4 - 2*p_gamma) + pow(1 - y, 4 - 2*p_gamma)) / tgamma(5 - 2*p_gamma) 
   );
} // check time_coeff for negative sign

/* --------------------------------------------------------------------
 * Bundle together three common calls that initialize a vector
 * -------------------------------------------------------------------- */
int initialize_vector(simulation_manager   *man,
                      HYPRE_SStructVector  *u)
{
   HYPRE_SStructVectorCreate( man->comm, man->grid_x, u );
   HYPRE_SStructVectorSetObjectType( *u, man->object_type ); //HYPRE_STRUCT 
   HYPRE_SStructVectorInitialize( *u );
   return 0;
}

int initialize_vector_parcsr(simulation_manager   *man,
                      HYPRE_SStructVector  *u)
{
   HYPRE_SStructVectorCreate( man->comm, man->grid_x, u );
   HYPRE_SStructVectorSetObjectType( *u,  HYPRE_PARCSR); //HYPRE_PARCSR
   HYPRE_SStructVectorInitialize( *u );
   return 0;
}

int initialize_vector_ij(simulation_manager   *man,
                      HYPRE_IJVector  *u)
{
   HYPRE_IJVectorCreate( man->comm, 0, man->grid_size-1, u );
   HYPRE_IJVectorSetObjectType( *u,  HYPRE_PARCSR); //HYPRE_STRUCT
   HYPRE_IJVectorInitialize( *u );
   return 0;
}

/* --------------------------------------------------------------------
 * Set initial condition in a vector for a time t
 *  t == 0 : results in the initial condition
 *  t > 0  : results in a zero vector
 *  t < 0  : results in a uniformly random vector
 * -------------------------------------------------------------------- */
int set_initial_condition(simulation_manager   *man,
                          HYPRE_SStructVector  *u,
                          double                t)
{
   double *values; 
   int i, j, m;

   initialize_vector(man, u);
   values = (double *) malloc( (man->nlx)*(man->nly)*sizeof(double) );
   
   if( t == 0.0)
   {
      /* Set the values in left-to-right, bottom-to-top order. */ 
      for( m = 0, j = 0; j < man->nly; j++ )
      {
         for (i = 0; i < man->nlx; i++, m++)
         {
            values[m] = U0( ((man->ilower[0])+i)*(man->dx), 
                           ((man->ilower[1])+j)*(man->dy) );
         }
      }
   }
   else if (t < 0.0)
   {
      for( m = 0; m < (man->nlx)*(man->nly); m++ )
      {
            values[m] = ((double)braid_Rand())/braid_RAND_MAX;
      }
   }
   else
   {
      for( m = 0; m < (man->nlx)*(man->nly); m++ )
      {
            values[m] = 0.0;
      }
   }

   HYPRE_SStructVectorSetBoxValues( *u, 0, man->ilower, man->iupper, 0, values ); 
   HYPRE_SStructVectorAssemble( *u );
   free(values);
   return 0;
}

/* --------------------------------------------------------------------
 * Set up a 2D grid.
 * -------------------------------------------------------------------- */

void setUp2Dgrid( MPI_Comm               comm,
             HYPRE_SStructGrid     *grid_ptr,
             int                    ndim,
             int                   *ilower, 
             int                   *iupper,
             HYPRE_SStructVariable  vartype,
             int                    nghost)
{
   /* We have one part and one variable. */
   int num_ghost[4];
   
   num_ghost[0] = nghost;
   num_ghost[1] = nghost;
   num_ghost[2] = nghost;
   num_ghost[3] = nghost;

   HYPRE_SStructGrid grid;

   /* Create an empty 2D grid object. */
   HYPRE_SStructGridCreate( comm, ndim, 1, &grid );
   
   /* Set the variable type for each part 
    * This call MUST go before setting the number of ghost */
   HYPRE_SStructGridSetVariables( grid, 0, 1, &vartype );

   /* Add a new box to the grid. */
   if ((ilower[0] <= iupper[0]) && (ilower[1] <= iupper[1]))
   {
      HYPRE_SStructGridSetExtents( grid, 0, ilower, iupper );
      HYPRE_SStructGridSetNumGhost(grid, num_ghost);
   }

   /* This is a collective call finalizing the grid assembly.
    * The grid is now ``ready to be used''. */
   HYPRE_SStructGridAssemble( grid );

   *grid_ptr = grid;
}

/* --------------------------------------------------------------------
 * Set up the Block matrix of beta
 * -------------------------------------------------------------------- */

void create_A_1_b(double **values, int grid_size, int nx, double coeff, double rho) 
{
   int b;
   int i, j, k, l;
   double temp_value;
   int M_rho = nx-1;

   temp_value = pow(2, 6 - 2*rho) + 16*rho - 40;
   temp_value *= coeff;

   for (i = 1; i <= M_rho-1; i++)
   {
      for (b = nx; b < grid_size-nx; b += nx)
      {
         values[i +b][i +b] += temp_value;
      }
   }

   temp_value = 2 * pow(3, 4 - 2*rho) + (2*rho - 6) * pow(2, 5 - 2*rho) - 16*rho + 34;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b = nx; b < grid_size-nx; b += nx)
      {
         values[j +b][j+1 +b] += temp_value;
         values[j+1 +b][j +b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value = 4 * (4 - 2*rho) * (-pow(l - 1, 3 - 2*rho) + 2*pow(l, 3 - 2*rho) - pow(l+1, 3 - 2*rho)) - 2*pow(l - 2, 4 - 2*rho) + 4*pow(l - 1, 4 - 2*rho) - 4*pow(l + 1, 4 - 2*rho) + 2*pow(l + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
         for (b = nx; b < grid_size-nx; b += nx)
         {
            values[k +b][k+l +b] += temp_value;
            values[k+l +b][k +b] += temp_value;
         }
      }
   }  
}


void create_A_2_b(double **values, int grid_size, int nx, double coeff, double rho) 
{
   int b, bb;
   int i, j, k, l, m, p;
   double temp_value;
   int M_rho = nx-1;

   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (i = 1; i <= M_rho-1; i++)
   {
      for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
      {
         values[i +b][i +bb] += temp_value;
         values[i +bb][i +b] += temp_value;
      }
   }

   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
      {
         values[j +b][j+1 +bb] += temp_value;
         values[j+1 +bb][j +b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value =(4 - 2*rho) * (pow(l - 2, 3 - 2*rho) - pow(l - 1, 3 - 2*rho) - pow(l, 3 - 2*rho) + pow(l+1, 3-2*rho)) + 2*pow(l - 2, 4 - 2*rho) - 6*pow(l - 1, 4 - 2*rho) + 6*pow(l, 4 - 2*rho) - 2*pow(l + 1, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
         for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
         {
            values[k +b][k+l +bb] += temp_value;
            values[k+l +bb][k +b] += temp_value;
         }
      }
   }  

   for (m = 1; m <= M_rho-2; m++)
   {
      temp_value =(4 - 2*rho) * (pow(m - 1, 3 - 2*rho) - pow(m, 3 - 2*rho) - pow(m + 1, 3 - 2*rho) + pow(m + 2, 3 - 2*rho)) + 2*pow(m - 1, 4 - 2*rho) - 6*pow(m, 4 - 2*rho) + 6*pow(m + 1, 4 - 2*rho) - 2*pow(m + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (p = 1; p <= (M_rho - m - 1); p++)
      {
         for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
         {
            values[p+m +b][p +bb] += temp_value;
            values[p +bb][p+m +b] += temp_value;
         }
      }
   }  
}

/* --------------------------------------------------------------------
 * Set up the Block matrix of gamma
 * -------------------------------------------------------------------- */

void create_A_1_g(double **values, int grid_size, int nx, int ny, double coeff, double rho) 
{
   int b;
   int i, j, k, l;
   double temp_value;
   int M_rho = ny-1;

   double **data;
   data = (double**)calloc(grid_size, sizeof(double*));
   for (i = 0; i < grid_size; i++)
   {
      data[i] = (double*)calloc(grid_size, sizeof(double)); // default value is zero.
   }

   int sz = (nx-2)*(ny-2);
   int nx2 = nx - 2;
   int ny2 = ny - 2;
   int *per, *ind;
   per = (int*)calloc(sz, sizeof(int));
   ind = (int*)calloc(sz, sizeof(int));
   for (j = 0; j < ny2; j++)
   {
      for (i = 0; i < nx2; i++)
      {
         per[j + i*ny2] = nx*(j+1) + 1 + i; 
//         printf("%d: %d\n", j + i*ny2, nx*(j+1) + 1 + i);
      }
   }
   for (k = 0, j = 1; j <= ny2; j++)
   {
      for (i = nx*j + 1; i < nx*(j+1)-1; i++, k++)
      {
         ind[k] = i;
//         printf("%d: %d\n", k, i);
      }
   }


   temp_value = pow(2, 6 - 2*rho) + 16*rho - 40;
   temp_value *= coeff;
   for (i = 1; i <= M_rho-1; i++)
   {
      for (b = nx; b < grid_size-nx; b += nx)
      {
         data[i +b][i +b] += temp_value;
      }
   }

   temp_value = 2 * pow(3, 4 - 2*rho) + (2*rho - 6) * pow(2, 5 - 2*rho) - 16*rho + 34;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b = nx; b < grid_size-nx; b += nx)
      {
         data[j +b][j+1 +b] += temp_value;
         data[j+1 +b][j +b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value = 4 * (4 - 2*rho) * (-pow(l - 1, 3 - 2*rho) + 2*pow(l, 3 - 2*rho) - pow(l+1, 3 - 2*rho)) - 2*pow(l - 2, 4 - 2*rho) + 4*pow(l - 1, 4 - 2*rho) - 4*pow(l + 1, 4 - 2*rho) + 2*pow(l + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
         for (b = nx; b < grid_size-nx; b += nx)
         {
            data[k +b][k+l +b] += temp_value;
            data[k+l +b][k +b] += temp_value;
         }
      }
   }  


   for (i = 0; i < sz; i++)
   {
      for (j = 0; j < sz; j++)
      {
         values[ind[i]][ind[j]] += data[per[i]][per[j]];
      }
   }

   for (i = 0; i < grid_size; i++)
   {
      free(data[i]);
   }
   free(data);
   free(per);
   free(ind);
}


void create_A_2_g(double **values, int grid_size, int nx, int ny, double coeff, double rho) 
{
   int b, bb;
   int i, j, k, l, m, p;
   double temp_value;
   int M_rho = ny-1;

   double **data;
   data = (double**)calloc(grid_size, sizeof(double*));
   for (i = 0; i < grid_size; i++)
   {
      data[i] = (double*)calloc(grid_size, sizeof(double));
   }

   int sz = (nx-2)*(ny-2);
   int nx2 = nx - 2;
   int ny2 = ny - 2;
   int *per, *ind;
   per = (int*)calloc(sz, sizeof(int));
   ind = (int*)calloc(sz, sizeof(int));
   for (j = 0; j < ny2; j++)
   {
      for (i = 0; i < nx2; i++)
      {
         per[j + i*ny2] = nx*(j+1) + 1 + i; 
      }
   }
   for (k = 0, j = 1; j <= ny2; j++)
   {
      for (i = nx*j + 1; i < nx*(j+1)-1; i++, k++)
      {
         ind[k] = i;
      }
   }


   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (i = 1; i <= M_rho-1; i++)
   {
      for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
      {
         data[i +b][i +bb] += temp_value;
         data[i +bb][i +b] += temp_value;
      }
   }

   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
      {
         data[j +b][j+1 +bb] += temp_value;
         data[j+1 +bb][j +b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value =(4 - 2*rho) * (pow(l - 2, 3 - 2*rho) - pow(l - 1, 3 - 2*rho) - pow(l, 3 - 2*rho) + pow(l+1, 3-2*rho)) + 2*pow(l - 2, 4 - 2*rho) - 6*pow(l - 1, 4 - 2*rho) + 6*pow(l, 4 - 2*rho) - 2*pow(l + 1, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
         for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
         {
            data[k +b][k+l +bb] += temp_value;
            data[k+l +bb][k +b] += temp_value;
         }
      }
   }  

   for (m = 1; m <= M_rho-2; m++)
   {
      temp_value =(4 - 2*rho) * (pow(m - 1, 3 - 2*rho) - pow(m, 3 - 2*rho) - pow(m + 1, 3 - 2*rho) + pow(m + 2, 3 - 2*rho)) + 2*pow(m - 1, 4 - 2*rho) - 6*pow(m, 4 - 2*rho) + 6*pow(m + 1, 4 - 2*rho) - 2*pow(m + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (p = 1; p <= (M_rho - m - 1); p++)
      {
         for (b=nx, bb=2*nx; bb < grid_size-nx; b += nx, bb +=nx)
         {
            data[p+m +b][p +bb] += temp_value;
            data[p +bb][p+m +b] += temp_value;
         }
      }
   }  


   for (i = 0; i < sz; i++)
   {
      for (j = 0; j < sz; j++)
      {
         values[ind[i]][ind[j]] += data[per[i]][per[j]];
      }
   }

   for (i = 0; i < grid_size; i++)
   {
      free(data[i]);
   }
   free(data);
   free(per);
   free(ind);
}

/* --------------------------------------------------------------------
 * Set up the implicit discretization matrix. 
 * First, we set the stencil values at every node neglecting the 
 * boundary. Then, we correct the matrix stencil at boundary nodes.
 * We have to eliminate the coefficients reaching outside of the domain
 * boundary. Furthermore, to incorporate boundary conditions, we remove
 * the connections between the interior nodes and boundary nodes.
 * -------------------------------------------------------------------- */

void setUpImplicitMatrix( simulation_manager *man)
{
   MPI_Comm             comm        = man->comm;
   int                  object_type = HYPRE_PARCSR;
   double               dt          = man->dt;
    double K_x = man->K_x;
    double K_y = man->K_y;
    double p_beta = man->p_beta;
    double p_gamma = man->p_gamma;
   
   double               dx, dy;
   int                  ilower[2], iupper[2], nlx, nly, nx, ny, px, py, pi, pj;  

   int i, j;

   double **values;
   int bc_ilower[2];
   int bc_iupper[2];

   HYPRE_IJMatrix A;

   /* Grab info from manager */
   grab_manager_spatial_info(man, ilower, iupper, &nlx, &nly, &nx, 
                             &ny, &px, &py, &pi, &pj, &dx, &dy);

    double M_coeff = dx * dy / 12;
    double A_x_coeff = dt/2 * K_x * pow(dx, 1 - 2*p_beta) * dy / (2 * cos(p_beta * PI) * tgamma(5 - 2*p_beta));
    double A_y_coeff = dt/2 * K_y * dx * pow(dy, 1 - 2*p_gamma) / (2 * cos(p_gamma * PI) * tgamma(5 - 2*p_gamma));
    int grid_size = man->grid_size;
   double temp_value, temp_value_2;
   int col[1]={1};

   /* Create an empty matrix object. */
//   HYPRE_SStructMatrixCreate( comm, graph, &A);
   HYPRE_IJMatrixCreate(comm, 0, grid_size-1, 0, grid_size-1, &A);

   /* Use symmetric storage? The function below is for symmetric stencil
    * entries (use HYPRE_SStructMatrixSetNSSymmetric for non-stencil 
    * entries). */
//   HYPRE_SStructMatrixSetSymmetric( A, 0, 0, 0, 0);

   /* Set the object type. */
   HYPRE_IJMatrixSetObjectType( A, object_type );

   /* Indicate that the matrix coefficients are ready to be set. */
   HYPRE_IJMatrixInitialize( A );

   // ************************************************** Allocating
   values = (double**)calloc(grid_size, sizeof(double*));
   for (i = 0; i < grid_size; i++)
   {
      values[i] = (double*)calloc(grid_size, sizeof(double));
   }

   // ************************************************** Mh
   // diag 0
   for (i = 0; i < grid_size; i++)
   {
      values[i][i] = M_coeff*6;
   }
   // diag 1, -1
   for (i = 0, j = 1; j < grid_size; ++i, ++j)
   {
      if (j%nx)
      {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
      }
   }
   // diag mb, -mb
   for (i = 0, j = nx; j < grid_size; ++i, ++j)
   {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
   }
   // diag mb+1, -mb-1
   for (i = 0, j = nx+1; j < grid_size; ++i, ++j)
   {
      if (j%nx)
      {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
      }
   }

   // ************************************************** A1b
   create_A_1_b(values, grid_size, nx, A_x_coeff, p_beta); 

   // ************************************************** A2b
   create_A_2_b(values, grid_size, nx, A_x_coeff, p_beta); 

   // ************************************************** A1g
   create_A_1_g(values, grid_size, nx, ny, A_y_coeff, p_gamma); 

   // ************************************************** A2g
   create_A_2_g(values, grid_size, nx, ny, A_y_coeff, p_gamma); 


   /* 2. correct stencils at boundary nodes */     
   /* Allocate vectors for values on boundary planes */
         
               /* Recall that the system we are solving is:
      *
      *   [A_ii 0; [x_i;    [b_i - A_ib*u_b;
      *      0  I]  x_b ] =        u_b       ].
      * 
      * This requires removing the connections between the interior
      * and boundary nodes that we have set up when we set the
      * 5pt stencil at each node. */

   /* a) boundaries y = 0 or y = 1 */
   /* The stencil at the boundary nodes is 1-0-0-0-0. Because
      * we have I x_b = u_b. */
         
   /* Processors at y = 0 */
   for (i = 0; i < nx; i++)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }

         
   /* Processors at y = 1 */
   for (i = grid_size-nx; i < grid_size; i++)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }
         
   /* b) boundaries x = 0 or x = 1 */    
   /* The stencil at the boundary nodes is 1-0-0-0-0. Because
      * we have I x_b = u_b. */

         
   /* Processors at x = 0 */
   for (i = nx; i < grid_size; i+=nx)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }

         
   /* Processors at x = 1 */
   for (i = nx-1; i < grid_size; i+=nx)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }
    


   /* Finalize the matrix assembly. */
   for (i = 0; i < grid_size; i++)
   {
      for (j = 0; j < grid_size; j++)
      {
         HYPRE_IJMatrixSetValues(A, 1, col, &i, &j, &values[i][j]);
      }
   }    
   HYPRE_IJMatrixAssemble( A );
   man->A = A;

   for (i = 0; i < grid_size; i++)
   {
      free(values[i]);
   }
   free(values);
}


/* --------------------------------------------------------------------
 * Set up the explicit discretization matrix. 
 * First, we set the stencil values at every node neglecting the 
 * boundary. Then, we correct the matrix stencil at boundary nodes.
 * We have to eliminate the coefficients reaching outside of the domain
 * boundary. 
 * -------------------------------------------------------------------- */
void setUpExplicitMatrix( simulation_manager *man)
{
   MPI_Comm             comm        = man->comm;
   int                  object_type = HYPRE_PARCSR;
   double               dt          = man->dt;
    double K_x = man->K_x;
    double K_y = man->K_y;
   double p_beta = man->p_beta;
    double p_gamma = man->p_gamma;
   
   double               dx, dy;
   int                  ilower[2], iupper[2], nlx, nly, nx, ny, px, py, pi, pj;  

   int i, j;

   double **values;
   int bc_ilower[2];
   int bc_iupper[2];

   HYPRE_IJMatrix B;

   /* Grab info from manager */
   grab_manager_spatial_info(man, ilower, iupper, &nlx, &nly, &nx, 
                             &ny, &px, &py, &pi, &pj, &dx, &dy);

    double M_coeff = dx * dy / 12;
    double A_x_coeff = -dt/2 * K_x * pow(dx, 1 - 2*p_beta) * dy / (2 * cos(p_beta * PI) * tgamma(5 - 2*p_beta));
    double A_y_coeff = -dt/2 * K_y * dx * pow(dy, 1 - 2*p_gamma) / (2 * cos(p_gamma * PI) * tgamma(5 - 2*p_gamma));
    int grid_size = man->grid_size;
   double temp_value, temp_value_2;
   int col[1]={1};

   /* Create an empty matrix object. */
//   HYPRE_SStructMatrixCreate( comm, graph, &A);
   HYPRE_IJMatrixCreate(comm, 0, grid_size-1, 0, grid_size-1, &B);

   /* Use symmetric storage? The function below is for symmetric stencil
    * entries (use HYPRE_SStructMatrixSetNSSymmetric for non-stencil 
    * entries). */
//   HYPRE_SStructMatrixSetSymmetric( A, 0, 0, 0, 0);

   /* Set the object type. */
   HYPRE_IJMatrixSetObjectType( B, object_type );

   /* Indicate that the matrix coefficients are ready to be set. */
   HYPRE_IJMatrixInitialize( B );


   // ************************************************** Allocating
   values = (double**)calloc(grid_size, sizeof(double*));
   for (i = 0; i < grid_size; i++)
   {
      values[i] = (double*)calloc(grid_size, sizeof(double));
   }

   // ************************************************** Mh
   // diag 0
   for (i = 0; i < grid_size; i++)
   {
      values[i][i] = M_coeff*6;
   }
   // diag 1, -1
   for (i = 0, j = 1; j < grid_size; ++i, ++j)
   {
      if (j%nx)
      {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
      }
   }
   // diag mb, -mb
   for (i = 0, j = nx; j < grid_size; ++i, ++j)
   {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
   }
   // diag mb+1, -mb-1
   for (i = 0, j = nx+1; j < grid_size; ++i, ++j)
   {
      if (j%nx)
      {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
      }
   }

   // ************************************************** A1b
   create_A_1_b(values, grid_size, nx, A_x_coeff, p_beta); 

   // ************************************************** A2b
   create_A_2_b(values, grid_size, nx, A_x_coeff, p_beta); 

   // ************************************************** A1g
   create_A_1_g(values, grid_size, nx, ny, A_y_coeff, p_gamma); 

   // ************************************************** A2g
   create_A_2_g(values, grid_size, nx, ny, A_y_coeff, p_gamma); 

   /* 2. correct stencils at boundary nodes */     
   /* Allocate vectors for values on boundary planes */
         
               /* Recall that the system we are solving is:
      *
      *   [A_ii 0; [x_i;    [b_i - A_ib*u_b;
      *      0  I]  x_b ] =        u_b       ].
      * 
      * This requires removing the connections between the interior
      * and boundary nodes that we have set up when we set the
      * 5pt stencil at each node. */

   /* a) boundaries y = 0 or y = 1 */
   /* The stencil at the boundary nodes is 1-0-0-0-0. Because
      * we have I x_b = u_b. */
         
   /* Processors at y = 0 */
   for (i = 0; i < nlx; i++)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }

         
   /* Processors at y = 1 */
   for (i = grid_size-nlx; i < grid_size; i++)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }
         
   /* b) boundaries x = 0 or x = 1 */    
   /* The stencil at the boundary nodes is 1-0-0-0-0. Because
      * we have I x_b = u_b. */

         
   /* Processors at x = 0 */
   for (i = nlx; i < grid_size; i+=nlx)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }

         
   /* Processors at x = 1 */
   for (i = nlx-1; i < grid_size; i+=nlx)
   {
      for (j = 0; j < grid_size; j++)
      {
         if (i == j)
         {
            values[i][i] = 1; 
         }
         else
         {
            values[i][j] = 0;
            values[j][i] = 0;
         }
      }
   }
    
 
    /* Finalize the matrix assembly. */
   for (i = 0; i < grid_size; i++)
   {
      for (j = 0; j < grid_size; j++)
      {
         HYPRE_IJMatrixSetValues(B, 1, col, &i, &j, &values[i][j]);
      }
   }    
   HYPRE_IJMatrixAssemble( B );

   man->B = B;

   for (i = 0; i < grid_size; i++)
   {
      free(values[i]);
   }
   free(values);
}


/* --------------------------------------------------------------------
 * Set up PFMG solver.
 * -------------------------------------------------------------------- */
void
setUpStructSolver( simulation_manager  *man,
                   HYPRE_SStructVector  F,
                   HYPRE_SStructVector  U_curr )                
{
   MPI_Comm            comm     = man->comm;
   HYPRE_IJMatrix      A        = man->A;
   int                 max_iter = man->max_iter;
   double              tol      = man->tol;
   
   HYPRE_Solver  solver;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector parcsr_U_curr, parcsr_F;


    HYPRE_IJMatrixGetObject( A, (void **) &parcsr_A );
    HYPRE_SStructVectorGetObject( F, (void **) &parcsr_F );
    HYPRE_SStructVectorGetObject( U_curr, (void **) &parcsr_U_curr );


    // Create and set up the solver
    HYPRE_ParCSRPCGCreate(comm, &solver);
    HYPRE_PCGSetTol(solver, tol);
    HYPRE_PCGSetPrintLevel(solver, 0); // Print residual norm every iteration
    HYPRE_PCGSetTwoNorm(solver, 1); // Use the two-norm for stopping criteria


    // Setup the PCG solver 
//    HYPRE_ParCSRPCGSetup(solver, parcsr_A, parcsr_F, parcsr_U_curr); 
    HYPRE_PCGSetMaxIter(solver, max_iter);
        
   man->solver = solver;
}


/* --------------------------------------------------------------------
 * Add forcing term:
 * We have to multiply the RHS of the PDE by dt.
 * -------------------------------------------------------------------- */


void setUpForcingSpace(simulation_manager *man)
{
    // Problem parameters
    int i, j, m, k, n;

      struct ElementPoints
    {
        double p1x;
        double p1y;
        double p2x;
        double p2y;
        double p3x;
        double p3y;
    } element[6];
   
   double     dt          = man->dt;
   double     dx, dy;
   int        ilower[2], iupper[2], nlx, nly, nx, ny, px, py, pi, pj; 
      double    *values; 
   double     time_coeff;
   int        rhs_ilower[2];
   int        rhs_iupper[2];
   int        istart, iend, jstart, jend;
   
   /* Grab info from manager */
   grab_manager_spatial_info(man, ilower, iupper, &nlx, &nly, &nx, 
                             &ny, &px, &py, &pi, &pj, &dx, &dy);
   
   /* Add the values from the RHS of the PDE in left-to-right,
    * bottom-to-top order. Entries associated with DoFs on the boundaries
    * are not considered so that boundary conditions are not messed up. */
    values = (double *) malloc( nlx*nly*sizeof(double) );
   
   rhs_ilower[0] = ilower[0];
   rhs_ilower[1] = ilower[1];
   istart        = 0;
   iend          = nlx;
   
   rhs_iupper[0] = iupper[0];
   rhs_iupper[1] = iupper[1];
   jstart        = 0;
   jend          = nly;
   
   /* Adjust box to not include boundary nodes. */
   
   /* a) Boundaries y = 0 or y = 1 */
   /*    i) Processors at y = 0 */
   if( pj == 0 )
   {
      rhs_ilower[1] += 1;
      jstart        += 1;
   }
   /*    ii) Processors at y = 1 */
   if( pj == py-1 )
   {
      rhs_iupper[1] -= 1;
      jend          -= 1;
   }
   
   /* b) Boundaries x = 0 or x = 1 */
   /*    i) Processors at x = 0 */
   if( pi == 0 )
   {
      rhs_ilower[0] += 1;
      istart        += 1;
   }
   /*    ii) Processors at x = 1 */
   if( pi == px-1 )
   {
      rhs_iupper[0] -= 1;
      iend          -= 1;
   }

    double K_x = man->K_x;
    double K_y = man->K_y;
    double p_beta = man->p_beta;
    double p_gamma = man->p_gamma;

    // Some parameters
    double p0x, p0y, p01x, p01y;
    double p21x, p21y, p31x, p31y;
    double point_x, point_y, F_value;
    double temp_value;
    int *row, *col;

    int point_count = 14;
    double point_s[] = {
        6.943184420297371e-2,
        6.943184420297371e-2, 
        6.943184420297371e-2, 
        6.943184420297371e-2, 
        6.943184420297371e-2, 
        0.330009478207572,       
        0.330009478207572,       
        0.330009478207572,      
        0.330009478207572,      
        0.669990521792428,       
        0.669990521792428,       
        0.669990521792428,      
        0.930568155797026,      
        0.930568155797026
    };

    double point_t[] = {
        4.365302387072518e-2,
        0.214742881469342,
        0.465284077898513,
        0.715825274327684,
        0.886915131926301,
        4.651867752656094e-2,
        0.221103222500738,
        0.448887299291690,
        0.623471844265867,
        3.719261778493340e-2,
        0.165004739103786,
        0.292816860422638,
        1.467267513102734e-2,
        5.475916907194637e-2
    };

    double weight[] = {
        1.917346464706755e-2,
        3.873334126144628e-2,
        4.603770904527855e-2,
        3.873334126144628e-2,
        1.917346464706755e-2,
        3.799714764789616e-2,
        7.123562049953998e-2,
        7.123562049953998e-2,
        3.799714764789616e-2,
        2.989084475992800e-2,
        4.782535161588505e-2,
        2.989084475992800e-2,
        6.038050853208200e-3,
        6.038050853208200e-3
    };

    // -------------------------------------------------- Element struct    
    // element 1, t
    element[0].p1x = -dx;
    element[0].p1y = -dy;
    element[0].p2x = 0;
    element[0].p2y = -dy;
    element[0].p3x = 0;
    element[0].p3y = 0;
    // element 2, t
    element[1].p1x = 0;
    element[1].p1y = -dy;
    element[1].p2x = dx;
    element[1].p2y = 0;
    element[1].p3x = 0;
    element[1].p3y = 0;
    // element 3, 1-s-t
    element[2].p1x = 0;
    element[2].p1y = 0;
    element[2].p2x = dx;
    element[2].p2y = 0;
    element[2].p3x = dx;
    element[2].p3y = dy;
    // element 4, 1-s-t
    element[3].p1x = 0;
    element[3].p1y = 0;
    element[3].p2x = dx;
    element[3].p2y = dy;
    element[3].p3x = 0;
    element[3].p3y = dy;
    // element 5, s
    element[4].p1x = -dx;
    element[4].p1y = 0;
    element[4].p2x = 0;
    element[4].p2y = 0;
    element[4].p3x = 0;
    element[4].p3y = dy;
    // element 6, s
    element[5].p1x = -dx;
    element[5].p1y = -dy;
    element[5].p2x = 0;
    element[5].p2y = 0;
    element[5].p3x = -dx;
    element[5].p3y = 0;    

        // -------------------------------------------------- Space of source term
  for( m = 0, j = jstart; j < jend; j++ )
   {
      for( i = istart; i < iend; i++, m++ )
      {

        temp_value = 0;

        p0x = i*dx;
        p0y = j*dy;
        for (k = 0; k < 6; ++k)
        {
            p21x = element[k].p2x - element[k].p1x;
            p21y = element[k].p2y - element[k].p1y;
            p31x = element[k].p3x - element[k].p1x;
            p31y = element[k].p3y - element[k].p1y;

            p01x = p0x + element[k].p1x;
            p01y = p0y + element[k].p1y;

            for (n = 0; n < point_count; ++n)
            {
                point_x = p01x + point_s[n]*p21x + point_t[n]*p31x;
                point_y = p01y + point_s[n]*p21y + point_t[n]*p31y;
                F_value = F_space(point_x, point_y, K_x, K_y, p_beta, p_gamma);
                switch (k/2)
                {
                case 0: // t
                    F_value *= point_t[n];
                    break;
                case 1: // 1-s-t
                    F_value *= (1-point_s[n]-point_t[n]);
                    break;
                case 2: // s
                    F_value *= point_s[n];
                    break;
                }
                temp_value += weight[n]*F_value;
            }
        }
        temp_value *= (dx * dy);
        values[m] = temp_value;
    }
   }
   man->Fxy = values;
}

void addForcingToRHS( simulation_manager *man, 
                 double              t,
                 HYPRE_SStructVector      F )                
{
   double     dt          = man->dt;
   double     dx, dy;
   int        ilower[2], iupper[2], nlx, nly, nx, ny, px, py, pi, pj; 
      double    *values; 
   double     time_coeff, temp_value;
   int        i, j, m;
   int        rhs_ilower[2];
   int        rhs_iupper[2];
   int        istart, iend, jstart, jend;
   
   /* Grab info from manager */
   grab_manager_spatial_info(man, ilower, iupper, &nlx, &nly, &nx, 
                             &ny, &px, &py, &pi, &pj, &dx, &dy);
   
   /* Add the values from the RHS of the PDE in left-to-right,
    * bottom-to-top order. Entries associated with DoFs on the boundaries
    * are not considered so that boundary conditions are not messed up. */
    values = (double *) malloc( nlx*nly*sizeof(double) );
   
   rhs_ilower[0] = ilower[0];
   rhs_ilower[1] = ilower[1];
   istart        = 0;
   iend          = nlx;
   
   rhs_iupper[0] = iupper[0];
   rhs_iupper[1] = iupper[1];
   jstart        = 0;
   jend          = nly;
   
   /* Adjust box to not include boundary nodes. */
   
   /* a) Boundaries y = 0 or y = 1 */
   /*    i) Processors at y = 0 */
   if( pj == 0 )
   {
      rhs_ilower[1] += 1;
      jstart        += 1;
   }
   /*    ii) Processors at y = 1 */
   if( pj == py-1 )
   {
      rhs_iupper[1] -= 1;
      jend          -= 1;
   }
   
   /* b) Boundaries x = 0 or x = 1 */
   /*    i) Processors at x = 0 */
   if( pi == 0 )
   {
      rhs_ilower[0] += 1;
      istart        += 1;
   }
   /*    ii) Processors at x = 1 */
   if( pi == px-1 )
   {
      rhs_iupper[0] -= 1;
      iend          -= 1;
   }

    time_coeff = exp(-t + dt) - exp(-t);
    time_coeff *= 10;
      for( m = 0, j = jstart; j < jend; j++ )
      {
        for( i = istart; i < iend; i++, m++ )
        {
            // (rhs_ilower[0]+i-istart), (rhs_ilower[1]+j-jstart)
            values[m] = time_coeff * (man->Fxy[m]);
        }
      }



       HYPRE_SStructVectorAddToBoxValues(F, 0, rhs_ilower,
                                     rhs_iupper, 0, values);
   
/*   HYPRE_SStructVectorAssemble( b );*/
   free(values);
}




/* --------------------------------------------------------------------
 * Time integration routine
 * -------------------------------------------------------------------- */
int take_step(simulation_manager * man,         /* manager holding basic sim info */
              HYPRE_SStructVector       U_stop,       /* approximation at tstop */
              HYPRE_SStructVector       F_stop,       /* additional RHS forcing */
              HYPRE_SStructVector       U_curr,           /* vector to evolve */
              double               tstart,      /* evolve x from tstart to tstop */
              double               tstop, 
              int                 *iters_taken) /* if implicit, returns the number of iters taken */
{
   int      iters      = man->max_iter; /* if implicit, max iters for solve */
   double   tol        = man->tol;      /* if implicit, solver tolerance */
   HYPRE_Int num_iters = 0;

    double *values;
    values = (double *) malloc((man->grid_size) * sizeof(double));

   HYPRE_SStructVector F;
   HYPRE_ParCSRMatrix  parcsr_A, parcsr_B;
   HYPRE_ParVector parcsr_F, parcsr_F_stop;
   HYPRE_ParVector  parcsr_U_stop, parcsr_U_curr;
   HYPRE_IJVector ij_U_curr, ij_U_stop;

   initialize_vector_ij(man, &ij_U_curr);
   HYPRE_IJVectorAssemble(ij_U_curr);
   HYPRE_IJVectorGetObject( ij_U_curr, (void **) &parcsr_U_curr );

   initialize_vector_ij(man, &ij_U_stop);
   HYPRE_IJVectorAssemble(ij_U_stop);
   HYPRE_IJVectorGetObject( ij_U_stop, (void **) &parcsr_U_stop );


   /* Grab these object pointers for use below */
   HYPRE_IJMatrixGetObject( man->A, (void **) &parcsr_A );
   HYPRE_IJMatrixGetObject( man->B, (void **) &parcsr_B );
   

   HYPRE_SStructVectorGetBoxValues(U_stop, 0, man->ilower, man->iupper, 0, values);
   HYPRE_IJVectorSetValues( ij_U_stop, man->grid_size, NULL, values );

   /* Create temporary vector */
   initialize_vector_parcsr(man, &F);
   addForcingToRHS( man, tstop, F );
   HYPRE_SStructVectorAssemble(F);
   HYPRE_SStructVectorGetObject( F, (void **)&parcsr_F );
   
    HYPRE_ParCSRMatrixMatvec(1.0, parcsr_B, parcsr_U_stop, 1.0, parcsr_F); 
//    HYPRE_ParVectorPrint(parcsr_F, "F");
    /* add RHS of PDE: g_i = Phi*dt*b_i, i > 0 */

    if (F_stop != NULL)
    {
        abort();
        // Add extra forcing from braid
        HYPRE_SStructVectorGetObject( F_stop, (void **) &parcsr_F_stop );
        HYPRE_ParVectorAxpy(1.0, parcsr_F_stop, parcsr_F);
    }

    if (U_stop != U_curr)
    {
        // Set initial guess
        HYPRE_ParVectorCopy(parcsr_U_stop, parcsr_U_curr);
    }

    /* Solve system */

    HYPRE_PCGSetTol( man->solver, tol );
    HYPRE_PCGSetMaxIter( man->solver, iters);
   HYPRE_ParCSRPCGSetup(man->solver, parcsr_A, parcsr_F, parcsr_U_curr); 
    HYPRE_ParCSRPCGSolve( man->solver, parcsr_A, parcsr_F, parcsr_U_curr );
    HYPRE_ParCSRPCGGetNumIterations( man->solver, &num_iters);
    (*iters_taken) = num_iters;

    HYPRE_IJVectorGetValues( ij_U_curr, man->grid_size, NULL, values );
    HYPRE_SStructVectorSetBoxValues(U_curr, 0, man->ilower, man->iupper, 0, values);


   /* free memory */
   free(values);
    HYPRE_SStructVectorDestroy(F);
    HYPRE_IJVectorDestroy(ij_U_curr);
    HYPRE_IJVectorDestroy(ij_U_stop);
   return 0;
}


/* --------------------------------------------------------------------
 * Residual routine
 * -------------------------------------------------------------------- */
int comp_res(simulation_manager * man,         /* manager holding basic sim info */
             HYPRE_SStructVector  U_stop,       /* approximation at tstop */
             HYPRE_SStructVector  r,           /* approximation at tstart */
             double               tstart,
             double               tstop)
{
   HYPRE_SStructVector F;
   HYPRE_ParCSRMatrix parcsr_A, parcsr_B;
   HYPRE_ParVector parcsr_U_stop, parcsr_F;
   HYPRE_ParVector parcsr_r; 
   HYPRE_IJVector ij_r, ij_U_stop;

   double *values;
   values = (double *) malloc((man->grid_size) * sizeof(double));

   initialize_vector_ij( man, &ij_U_stop );
   HYPRE_IJVectorAssemble(ij_U_stop);
   HYPRE_IJVectorGetObject( ij_U_stop, (void **) &parcsr_U_stop );

   initialize_vector_ij( man, &ij_r );
   HYPRE_IJVectorAssemble(ij_r);
   HYPRE_IJVectorGetObject( ij_r, (void **) &parcsr_r );


   // Grab these object pointers for use below
   HYPRE_IJMatrixGetObject( man->A, (void **) &parcsr_A );
   HYPRE_IJMatrixGetObject( man->B, (void **) &parcsr_B );


   HYPRE_SStructVectorGetBoxValues(U_stop, 0, man->ilower, man->iupper, 0, values);
   HYPRE_IJVectorSetValues( ij_U_stop, man->grid_size, NULL, values );


   HYPRE_SStructVectorGetBoxValues(r, 0, man->ilower, man->iupper, 0, values);
   HYPRE_IJVectorSetValues(ij_r, man->grid_size, NULL, values );


   /* Create temporary vector */
   initialize_vector_parcsr(man, &F);
   addForcingToRHS( man, tstop, F );
   HYPRE_SStructVectorAssemble(F);
   HYPRE_SStructVectorGetObject( F, (void **)&parcsr_F );


   HYPRE_ParCSRMatrixMatvec(1.0, parcsr_B, parcsr_r, 1.0, parcsr_F); 
   HYPRE_ParCSRMatrixMatvec( 1.0, parcsr_A, parcsr_U_stop, 0.0, parcsr_U_stop );
   HYPRE_ParVectorAxpy(-1.0, parcsr_F, parcsr_U_stop);

   HYPRE_IJVectorGetValues( ij_U_stop, man->grid_size, NULL, values );
   HYPRE_SStructVectorSetBoxValues(r, 0, man->ilower, man->iupper, 0, values);


   /* free memory */
   free(values);
   HYPRE_SStructVectorDestroy(F);
   HYPRE_IJVectorDestroy(ij_r);
   HYPRE_IJVectorDestroy(ij_U_stop);
   return 0;
}


/* --------------------------------------------------------------------
 * Compute the current error vector, relative to the true continuous 
 * solution.  Assumes that e has already been allocated and setup
 * -------------------------------------------------------------------- */
int compute_error(simulation_manager  *man, 
                  HYPRE_SStructVector  x, 
                  double               t,
                  HYPRE_SStructVector  e) 

{
   double  *values;
   int i, j, m;
   

   values = (double *) malloc( (man->nlx) * (man->nly)*sizeof(double) );
   HYPRE_SStructVectorGetBoxValues( x, 0, man->ilower, man->iupper, 0, values );
   
   /* Compute error */
   m = 0;
   for( j = 0; j < man->nly; j++ ){
      for( i = 0; i < man->nlx; i++ ){
         values[m] =  U_exact(man, 
                              ((man->ilower[0])+i)*(man->dx), 
                              ((man->ilower[1])+j)*(man->dy), t) 
                      - values[m];
         m++;
      }
   }
   HYPRE_SStructVectorSetBoxValues( e, 0, man->ilower, man->iupper, 0, values );
   free( values );
   HYPRE_SStructVectorAssemble( e );

   return 0;
}


/* --------------------------------------------------------------------
 * Compute || x ||_2 and return in norm_ptr
 * The 2-norm (Euclidean norm) is used.
 * -------------------------------------------------------------------- */
int norm(HYPRE_SStructVector  x, 
         double              *norm_ptr)
{
   double dot;
   hypre_SStructInnerProd( x, x, &dot );
   *norm_ptr = sqrt(dot);
   return 0;
}

/* --------------------------------------------------------------------
 * Compute the little l2 norm of the discretization error
 * -------------------------------------------------------------------- */
int compute_disc_err(simulation_manager  *man, 
                     HYPRE_SStructVector  u, 
                     double               tstop, 
                     HYPRE_SStructVector  e, 
                     double              *disc_err)
{
   /* be sure to scale by mesh size so that this is the little l2 norm */
   compute_error(man, u, tstop, e);
   norm(e, disc_err);
   *disc_err = sqrt( (*disc_err)*(*disc_err)*(man->dx)*(man->dy) );
   return 0;
}



/* --------------------------------------------------------------------
 * norm L2
 * -------------------------------------------------------------------- */
double compute_L2_error(simulation_manager  *man, 
                  HYPRE_SStructVector  u, 
                  double               t) 

{
    // Problem parameters
    int i, j, m, k, n;
   
   double     dt          = man->dt;
   double     dx = man->dx;
   double  dy= man->dy;
   int        nlx = man->nlx;
   int  nly = man->nly; 
      double    *values;
      double temp_value, error; 

   
   /* Grab info from manager */

   
   /* Add the values from the RHS of the PDE in left-to-right,
    * bottom-to-top order. Entries associated with DoFs on the boundaries
    * are not considered so that boundary conditions are not messed up. */


    int point_count = 14;
    double point_s[] = {
        6.943184420297371e-2,
        6.943184420297371e-2, 
        6.943184420297371e-2, 
        6.943184420297371e-2, 
        6.943184420297371e-2, 
        0.330009478207572,       
        0.330009478207572,       
        0.330009478207572,      
        0.330009478207572,      
        0.669990521792428,       
        0.669990521792428,       
        0.669990521792428,      
        0.930568155797026,      
        0.930568155797026
    };

    double point_t[] = {
        4.365302387072518e-2,
        0.214742881469342,
        0.465284077898513,
        0.715825274327684,
        0.886915131926301,
        4.651867752656094e-2,
        0.221103222500738,
        0.448887299291690,
        0.623471844265867,
        3.719261778493340e-2,
        0.165004739103786,
        0.292816860422638,
        1.467267513102734e-2,
        5.475916907194637e-2
    };

    double weight[] = {
        1.917346464706755e-2,
        3.873334126144628e-2,
        4.603770904527855e-2,
        3.873334126144628e-2,
        1.917346464706755e-2,
        3.799714764789616e-2,
        7.123562049953998e-2,
        7.123562049953998e-2,
        3.799714764789616e-2,
        2.989084475992800e-2,
        4.782535161588505e-2,
        2.989084475992800e-2,
        6.038050853208200e-3,
        6.038050853208200e-3
    };


   int Th = 2*(nlx-1)*(nly-1);

   int index_diff = nlx + 1;
   int down_index = 0;
   int up_index = index_diff;


   struct ElementNodes
   {
      int p1;
      int p2;
      int p3;
   } *element_node;

   element_node = (struct ElementNodes*)calloc(Th, sizeof(struct ElementNodes));
   for (i = 0; i < Th; i+=2)
   {
      
      element_node[i].p1 = up_index - index_diff;
      element_node[i].p2 = up_index - 1;
      element_node[i].p3 = up_index;

      element_node[i+1].p1 = down_index;
      element_node[i+1].p2 = down_index + 1;
      element_node[i+1].p3 = down_index + index_diff;


      down_index = down_index + 1;
      up_index = up_index + 1;

      if ((down_index+1)%nlx == 0)
      {
        down_index = down_index + 1;
        up_index = up_index + 1;
      }
   }


        // -------------------------------------------------- Space of source term
   values = (double *) malloc( nlx * nly*sizeof(double) );
   HYPRE_SStructVectorGetBoxValues( u, 0, man->ilower, man->iupper, 0, values );

   double p1x, p1y, p2x, p2y, p3x, p3y;
   double u1, u2, u3;
   double p12x, p12y, p32x, p32y;
   double point_x, point_y, U_value;

   error = 0;
   for( m = 0; m < Th; m++ )
   {
      p1x = ((element_node[m].p1)%nlx)*dx;
      p1y = ((element_node[m].p1)/nly)*dy;
      p2x = ((element_node[m].p2)%nlx)*dx;
      p2y = ((element_node[m].p2)/nly)*dy;
      p3x = ((element_node[m].p3)%nlx)*dx;
      p3y = ((element_node[m].p3)/nly)*dy;

      u1 = values[element_node[m].p1];
      u2 = values[element_node[m].p2];
      u3 = values[element_node[m].p3];

//      printf("-----\n");
//      printf("%d: (%lf, %lf), %lf\n", element_node[m].p1, p1x, p1y, u1);
//      printf("%d: (%lf, %lf), %lf\n", element_node[m].p2, p2x, p2y, u2);
//      printf("%d: (%lf, %lf), %lf\n", element_node[m].p3, p3x, p3y, u3);

      temp_value = 0;

      p32x = p3x - p2x;
      p32y = p3y - p2y;
      p12x = p1x - p2x;
      p12y = p1y - p2y;

      for (n = 0; n < point_count; ++n)
      {
         point_x = p2x + point_s[n]*p32x + point_t[n]*p12x;
         point_y = p2y + point_s[n]*p32y + point_t[n]*p12y;
         U_value = U_exact(man, point_x, point_y, t); 

         if (m%2 == 0)
         {
            U_value -= (u1 + ((u2 - u3)*p1x)/dx + ((u1 - u2)*p1y)/dy + (-u2 + u3)/dx* point_x + (-u1 + u2)/dy* point_y);
         }
         else
         {
            U_value -= (u1 + ((u1 - u2)*p1x)/dx + ((u2 - u3)*p1y)/dy + (-u1 + u2)/dx*point_x + (-u2 + u3)/dy*point_y);

         }
         temp_value += weight[n]*pow(U_value, 2);
      }
//      printf("%d: %e\n", m, temp_value);
      error += temp_value;
   }
   error = sqrt(error*(dx * dy));
   free(values);
   free(element_node);
   return error;
}

void print_L2_norm(simulation_manager * man, HYPRE_SStructVector u)
{
   int i;
   double dx = man->dx;
   double dy = man->dy;
   double tstop = man->tstop;
   int nx = man->nx;
   int ny = man->ny;

          // -------------------------------------------------- Print the solution
           int num_values; 
    double L2error, *values;

    num_values = (man->nlx) * (man->nly); 
    values = (double *)malloc(num_values * sizeof(double)); 
    HYPRE_SStructVectorGetBoxValues(u, 0, man->ilower, man->iupper, 0, values);
    L2error = compute_L2_error(man, u, tstop);
    	printf("L2 error: %0.4e\n", L2error);

    FILE *myfile;
   myfile = fopen("result.txt", "w");

    	fprintf(myfile, "L2 error: %0.4e\n", L2error);
    fprintf(myfile, "Solution:\nPoint(x,y)\t\t\tExact\tNumeric\tError\n"); 
    for (i = 0; i < num_values; i++) 
    { 
        fprintf(myfile, "(%lf, %lf): %lf \t %lf \t %lf\n", (i%nx)*dx, (i/ny)*dy, 
                                                U_exact(man, (i%nx)*dx, (i/ny)*dy, 1),
                                                values[i],
                                                fabs(U_exact(man, (i%nx)*dx, (i/ny)*dy, 1) - values[i])); 
    } 
    fprintf(myfile,"\n");
    fclose(myfile); 

    free(values);

}