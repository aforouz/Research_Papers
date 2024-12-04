#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
   int                     nx, ny;
   int                     grid_size;
   double                  tstart, tstop;
   int                     nt;
   double                  dx, dy;
   double                  dt;
   int                     px, py;
   int                     nlx, nly;
   HYPRE_SStructVariable   vartype;
   HYPRE_SStructGrid       grid_x;
   HYPRE_IJMatrix          A, Phi;
   HYPRE_IJVector          G;
   double                  *Fxy;
   int                     ilower[2], iupper[2];
   int                     object_type;
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
                              int                    *nx,
                              int                    *ny,
                              double                 *dx,
                              double                 *dy)
{
   (*nx)  = man->nx;
   (*ny)  = man->ny;
   (*dx)  = man->dx;
   (*dy)  = man->dy;

   ilower[0] = man->ilower[0];
   ilower[1] = man->ilower[1];
   iupper[0] = man->iupper[0];
   iupper[1] = man->iupper[1];

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
 * Forcing term: b(x,y,t) = -sin(x)*sin(y)*( sin(t) - 2*K*cos(t) )
 * -------------------------------------------------------------------- */
double F_space(double x, double y, double K_x, double K_y, double p_beta, double p_gamma)
{
//   return 0;
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
   values = (double *) calloc( man->grid_size, sizeof(double) );
   
   if( t == 0.0)
   {
      /* Set the values in left-to-right, bottom-to-top order. */ 
      for( m = 0, j = 1; j < man->ny; j++ )
      {
         for (i = 1; i < man->nx; i++, m++)
         {
            values[m] = U0( i*(man->dx), j*(man->dy) );
         }
      }
   }
   else if (t < 0.0)
   {
      for( m = 0; m < man->grid_size; m++ )
      {
            values[m] = ((double)braid_Rand())/braid_RAND_MAX;
      }
   }
   else
   {
      for( m = 0; m < man->grid_size; m++ )
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
             HYPRE_SStructVariable  vartype)
{
   HYPRE_SStructGrid grid;

   /* Create an empty 2D grid object. */
   HYPRE_SStructGridCreate( comm, ndim, 1, &grid );
   
   /* Set the variable type for each part 
    * This call MUST go before setting the number of ghost */
   HYPRE_SStructGridSetVariables( grid, 0, 1, &vartype );

   /* Add a new box to the grid. */
   HYPRE_SStructGridSetExtents( grid, 0, ilower, iupper );

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
   int M_rho = nx;
   int diff = nx-1;

   temp_value = pow(2, 6 - 2*rho) + 16*rho - 40;
   temp_value *= coeff;

   for (i = 1; i <= M_rho-1; i++)
   {
      for (b = 0; b < grid_size; b += diff)
      {
         values[i -1+b][i -1+b] += temp_value;
      }
   }

   temp_value = 2 * pow(3, 4 - 2*rho) + (2*rho - 6) * pow(2, 5 - 2*rho) - 16*rho + 34;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b = 0; b < grid_size; b += diff)
      {
         values[j -1+b][j+1 -1+b] += temp_value;
         values[j+1 -1+b][j -1+b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value = 4 * (4 - 2*rho) * (-pow(l - 1, 3 - 2*rho) + 2*pow(l, 3 - 2*rho) - pow(l+1, 3 - 2*rho)) - 2*pow(l - 2, 4 - 2*rho) + 4*pow(l - 1, 4 - 2*rho) - 4*pow(l + 1, 4 - 2*rho) + 2*pow(l + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
         for (b = 0; b < grid_size; b += diff)
         {
            values[k -1+b][k+l -1+b] += temp_value;
            values[k+l -1+b][k -1+b] += temp_value;
         }
      }
   }  
}

void create_A_2_b(double **values, int grid_size, int nx, double coeff, double rho) 
{
   int b, bb;
   int i, j, k, l, m, p;
   double temp_value;
   int M_rho = nx;
      int diff = nx-1;

   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (i = 1; i <= M_rho-1; i++)
   {
      for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
      {
         values[i -1+b][i -1+bb] += temp_value;
         values[i -1+bb][i -1+b] += temp_value;
      }
   }

   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
      {
         values[j -1+b][j+1 -1+bb] += temp_value;
         values[j+1 -1+bb][j -1+b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value =(4 - 2*rho) * (pow(l - 2, 3 - 2*rho) - pow(l - 1, 3 - 2*rho) - pow(l, 3 - 2*rho) + pow(l+1, 3-2*rho)) + 2*pow(l - 2, 4 - 2*rho) - 6*pow(l - 1, 4 - 2*rho) + 6*pow(l, 4 - 2*rho) - 2*pow(l + 1, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
         for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
         {
            values[k -1+b][k+l -1+bb] += temp_value;
            values[k+l -1+bb][k -1+b] += temp_value;
         }
      }
   }  

   for (m = 1; m <= M_rho-2; m++)
   {
      temp_value =(4 - 2*rho) * (pow(m - 1, 3 - 2*rho) - pow(m, 3 - 2*rho) - pow(m + 1, 3 - 2*rho) + pow(m + 2, 3 - 2*rho)) + 2*pow(m - 1, 4 - 2*rho) - 6*pow(m, 4 - 2*rho) + 6*pow(m + 1, 4 - 2*rho) - 2*pow(m + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (p = 1; p <= (M_rho - m - 1); p++)
      {
         for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
         {
            values[p+m -1+b][p -1+bb] += temp_value;
            values[p -1+bb][p+m -1+b] += temp_value;
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
   int M_rho = ny;
      int diff = nx-1;

   double **data;
   data = (double**)calloc(grid_size, sizeof(double*));
   for (i = 0; i < grid_size; i++)
   {
      data[i] = (double*)calloc(grid_size, sizeof(double)); // default value is zero.
   }

   int sz = grid_size; //(nx-1)*(ny-1);
   int nx2 = nx - 1;
   int ny2 = ny - 1;
   int *per, *ind;
   per = (int*)calloc(sz, sizeof(int));
   ind = (int*)calloc(sz, sizeof(int));
   for (j = 0; j < ny2; j++)
   {
      for (i = 0; i < nx2; i++)
      {
         per[j + i*ny2] = nx2*j + i; 
      }
   }
   for (k = 0; k < grid_size; k++)
   {
      ind[k] = k;
   }


   temp_value = pow(2, 6 - 2*rho) + 16*rho - 40;
   temp_value *= coeff;
   for (i = 1; i <= M_rho-1; i++)
   {
      for (b = 0; b < grid_size; b += diff)
      {
         data[i -1+b][i -1+b] += temp_value;
      }
   }

   temp_value = 2 * pow(3, 4 - 2*rho) + (2*rho - 6) * pow(2, 5 - 2*rho) - 16*rho + 34;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b = 0; b < grid_size; b += diff)
      {
         data[j -1+b][j+1 -1+b] += temp_value;
         data[j+1 -1+b][j -1+b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value = 4 * (4 - 2*rho) * (-pow(l - 1, 3 - 2*rho) + 2*pow(l, 3 - 2*rho) - pow(l+1, 3 - 2*rho)) - 2*pow(l - 2, 4 - 2*rho) + 4*pow(l - 1, 4 - 2*rho) - 4*pow(l + 1, 4 - 2*rho) + 2*pow(l + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
      for (b = 0; b < grid_size; b += diff)
         {
            data[k -1+b][k+l -1+b] += temp_value;
            data[k+l -1+b][k -1+b] += temp_value;
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
   int M_rho = ny;
      int diff = nx-1;

   double **data;
   data = (double**)calloc(grid_size, sizeof(double*));
   for (i = 0; i < grid_size; i++)
   {
      data[i] = (double*)calloc(grid_size, sizeof(double));
   }

   int sz = grid_size; // (nx-1)*(ny-1);
   int nx2 = nx - 1;
   int ny2 = ny - 1;
   int *per, *ind;
   per = (int*)calloc(sz, sizeof(int));
   ind = (int*)calloc(sz, sizeof(int));
   for (j = 0; j < ny2; j++)
   {
      for (i = 0; i < nx2; i++)
      {
         per[j + i*ny2] = nx2*j + i; 
      }
   }
   for (k = 0; k < grid_size; k++)
   {
      ind[k] = k;
   }


   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (i = 1; i <= M_rho-1; i++)
   {
      for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
      {
         data[i -1+b][i -1+bb] += temp_value;
         data[i -1+bb][i -1+b] += temp_value;
      }
   }

   temp_value = 4 - pow(2, 4 - 2*rho)*rho;
   temp_value *= coeff;
   for (j = 1; j <= M_rho-2; j++)
   {
      for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
      {
         data[j -1+b][j+1 -1+bb] += temp_value;
         data[j+1 -1+bb][j -1+b] += temp_value;
      }
   }

   for (l = 2; l <= M_rho-2; l++)
   {
      temp_value =(4 - 2*rho) * (pow(l - 2, 3 - 2*rho) - pow(l - 1, 3 - 2*rho) - pow(l, 3 - 2*rho) + pow(l+1, 3-2*rho)) + 2*pow(l - 2, 4 - 2*rho) - 6*pow(l - 1, 4 - 2*rho) + 6*pow(l, 4 - 2*rho) - 2*pow(l + 1, 4 - 2*rho);
      temp_value *= coeff;
      for (k = 1; k <= (M_rho - l - 1); k++)
      {
         for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
         {
            data[k -1+b][k+l -1+bb] += temp_value;
            data[k+l -1+bb][k -1+b] += temp_value;
         }
      }
   }  

   for (m = 1; m <= M_rho-2; m++)
   {
      temp_value =(4 - 2*rho) * (pow(m - 1, 3 - 2*rho) - pow(m, 3 - 2*rho) - pow(m + 1, 3 - 2*rho) + pow(m + 2, 3 - 2*rho)) + 2*pow(m - 1, 4 - 2*rho) - 6*pow(m, 4 - 2*rho) + 6*pow(m + 1, 4 - 2*rho) - 2*pow(m + 2, 4 - 2*rho);
      temp_value *= coeff;
      for (p = 1; p <= (M_rho - m - 1); p++)
      {
         for (b=0, bb=diff; bb < grid_size; b += diff, bb += diff)
         {
            data[p+m -1+b][p -1+bb] += temp_value;
            data[p -1+bb][p+m -1+b] += temp_value;
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
 * Set up the implicit matrix. 
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
   grab_manager_spatial_info(man, ilower, iupper, &nx, &ny, &dx, &dy);

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
      if (j%(nx-1))
      {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
      }
   }
   // diag mb, -mb
   for (i = 0, j = nx-1; j < grid_size; ++i, ++j)
   {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
   }
   // diag mb+1, -mb-1
   for (i = 0, j = nx; j < grid_size; ++i, ++j)
   {
      if (j%(nx-1))
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
 * Set up the explicit matrix. 
 * -------------------------------------------------------------------- */

void setUpExplicitMatrix( simulation_manager *man, double **values)
{
   MPI_Comm             comm        = man->comm;
   int                  object_type = HYPRE_PARCSR;
   double               dt          = man->dt;
    double K_x = man->K_x;
    double K_y = man->K_y;
   double p_beta = man->p_beta;
    double p_gamma = man->p_gamma;
   
   double               dx, dy;
   int                  ilower[2], iupper[2], nx, ny;  

   int i, j;

//   double **values;
   int bc_ilower[2];
   int bc_iupper[2];

//   HYPRE_IJMatrix B;

   /* Grab info from manager */
   grab_manager_spatial_info(man, ilower, iupper, &nx, &ny, &dx, &dy);

    double M_coeff = dx * dy / 12;
    double A_x_coeff = -dt/2 * K_x * pow(dx, 1 - 2*p_beta) * dy / (2 * cos(p_beta * PI) * tgamma(5 - 2*p_beta));
    double A_y_coeff = -dt/2 * K_y * dx * pow(dy, 1 - 2*p_gamma) / (2 * cos(p_gamma * PI) * tgamma(5 - 2*p_gamma));
    int grid_size = man->grid_size;
   double temp_value, temp_value_2;
   int col[1]={1};


   // ************************************************** Allocating

   // ************************************************** Mh
   // diag 0
   for (i = 0; i < grid_size; i++)
   {
      values[i][i] = M_coeff*6;
   }
   // diag 1, -1
   for (i = 0, j = 1; j < grid_size; ++i, ++j)
   {
      if (j%(nx-1))
      {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
      }
   }
   // diag mb, -mb
   for (i = 0, j = nx-1; j < grid_size; ++i, ++j)
   {
         values[i][j] = M_coeff;
         values[j][i] = M_coeff;
   }
   // diag mb+1, -mb-1
   for (i = 0, j = nx; j < grid_size; ++i, ++j)
   {
      if (j%(nx-1))
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

    /* Finalize the matrix assembly. */
}

/* --------------------------------------------------------------------
 * Set up the Phi matrix. 
 * -------------------------------------------------------------------- */

void setUpPhiMatrix(simulation_manager *man)
{
   MPI_Comm             comm        = man->comm;

   HYPRE_Solver  solver;
   HYPRE_IJMatrix Phi;
   HYPRE_IJVector b, x;
   HYPRE_ParVector parcsr_b, parcsr_x;
   HYPRE_ParCSRMatrix parcsr_Phi, parcsr_A;
   
   int i, j;
   int grid_size = man->grid_size;
   double temp_value;
   int col[1]={1};
   double **values;

   // Allocate
   values = (double**)calloc(grid_size, sizeof(double*));
   for (i = 0; i < grid_size; i++)
   {
      values[i] = (double*)calloc(grid_size, sizeof(double));
   }

   // Structure   
   HYPRE_IJMatrixCreate(comm, 0, grid_size-1, 0, grid_size-1, &Phi);
   HYPRE_IJMatrixSetObjectType( Phi , HYPRE_PARCSR );
   HYPRE_IJMatrixInitialize( Phi );

   initialize_vector_ij(man, &b);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject( b, (void **) &parcsr_b );

   initialize_vector_ij(man, &x);
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject( x, (void **) &parcsr_x );

   // Init
   setUpImplicitMatrix(man);
   HYPRE_IJMatrixGetObject( man->A, (void **) &parcsr_A );

   setUpExplicitMatrix(man, values);

   HYPRE_ParCSRPCGCreate(comm, &solver);
   HYPRE_PCGSetTwoNorm(solver, 1); // Use the two-norm for stopping criteria
//   HYPRE_PCGSetPrintLevel(solver, 2);
   HYPRE_PCGSetTol( solver, 0.0 );
   HYPRE_PCGSetMaxIter( solver, 100 );
   HYPRE_ParCSRPCGSetup( solver, parcsr_A, parcsr_b, parcsr_x ); 
   
   // Phi = A^{-1} B
   for (j = 0; j < grid_size; j++)
   {
      // B^{(j)}
      for (i = 0; i < grid_size; i++)
      {
         HYPRE_IJVectorSetValues(b, 1, &i, &values[i][j]);
      }
      
      // Solver A^{-1} B^{(j)}
      HYPRE_ParCSRPCGSolve( solver, parcsr_A, parcsr_b, parcsr_x );

      // Phi^{(j)}
      for (i = 0; i < grid_size; i++)
      {
         HYPRE_IJVectorGetValues(x, 1, &i, &temp_value);
         HYPRE_IJMatrixSetValues(Phi, 1, col, &i, &j, &temp_value);
      }
   }
   HYPRE_IJMatrixAssemble( Phi );

   man->Phi = Phi;

   // Free
   for (i = 0; i < grid_size; i++)
   {
      free(values[i]);
   }
   free(values);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);
   HYPRE_ParCSRPCGDestroy(solver);
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
   
   int grid_size = man->grid_size;
   double     dt          = man->dt;
   double     dx, dy;
   int        ilower[2], iupper[2], nx, ny; 
      double    *values; 
   double     time_coeff;
   int        istart, iend, jstart, jend;
   
   /* Grab info from manager */
   grab_manager_spatial_info(man, ilower, iupper, &nx, 
                             &ny, &dx, &dy);
   
   /* Add the values from the RHS of the PDE in left-to-right,
    * bottom-to-top order. Entries associated with DoFs on the boundaries
    * are not considered so that boundary conditions are not messed up. */
    values = (double *) calloc( grid_size, sizeof(double) );
   
   istart        = ilower[0];
   iend          = iupper[0];
   
   jstart        = ilower[1];
   jend          = iupper[1];

   /* Adjust box to not include boundary nodes. */
   
    double K_x = man->K_x;
    double K_y = man->K_y;
    double p_beta = man->p_beta;
    double p_gamma = man->p_gamma;

    // Some parameters
    double p0x, p0y, p01x, p01y;
    double p21x, p21y, p31x, p31y;
    double point_x, point_y, F_value;
    double temp_value;
   
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
   for( m = 0, j = jstart; j <= jend; j++ )
   {
      for( i = istart; i <= iend; i++, m++ )
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

void setUpGVector(simulation_manager *man)
{
   MPI_Comm             comm        = man->comm;

   HYPRE_Solver  solver;
   HYPRE_IJVector b, G;
   HYPRE_ParVector parcsr_b, parcsr_G;
   HYPRE_ParCSRMatrix parcsr_A;
   
   int i, j;
   int grid_size = man->grid_size;
   double temp_value;
   int col[1]={1};

   // Allocate

   // Structure   
   initialize_vector_ij(man, &b);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject( b, (void **) &parcsr_b );

   initialize_vector_ij(man, &G);
   HYPRE_IJVectorAssemble(G);
   HYPRE_IJVectorGetObject( G, (void **) &parcsr_G );

   HYPRE_IJMatrixGetObject( man->A, (void **) &parcsr_A );
   HYPRE_ParCSRPCGCreate(comm, &solver);
   HYPRE_PCGSetTwoNorm(solver, 1); // Use the two-norm for stopping criteria
//   HYPRE_PCGSetPrintLevel(solver, 2);
   HYPRE_PCGSetTol( solver, 0.0 );
   HYPRE_PCGSetMaxIter( solver, 100 );
   HYPRE_ParCSRPCGSetup( solver, parcsr_A, parcsr_b, parcsr_G ); 

   // Init
   
   // G^{n} = A^{-1} F^{n}

   // F^{n}
   for (i = 0; i < grid_size; i++)
   {
      HYPRE_IJVectorSetValues(b, 1, &i, &(man->Fxy[i]));
   }
      
   // Solver A^{-1} B^{(j)}
   HYPRE_ParCSRPCGSolve( solver, parcsr_A, parcsr_b, parcsr_G );

   man->G = G;

   // Free
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJMatrixDestroy(man->A);
   HYPRE_ParCSRPCGDestroy(solver);
}

void addForcingToRHS( simulation_manager *man, 
                 double              t,
                 HYPRE_SStructVector      F )                
{
   int grid_size           = man->grid_size;
   double     dt           = man->dt;
   double    *values; 
   double     time_coeff;
   int        m;
 
   
   /* Add the values from the RHS of the PDE in left-to-right,
    * bottom-to-top order. Entries associated with DoFs on the boundaries
    * are not considered so that boundary conditions are not messed up. */
   
   values = (double *) calloc( grid_size, sizeof(double) );
   
   time_coeff = 10*(exp(-t + dt) - exp(-t));
   HYPRE_IJVectorGetValues(man->G, grid_size, NULL, values);
   for( m = 0; m < grid_size; m++ )
   {
      values[m] *= time_coeff;
   }

   HYPRE_SStructVectorAddToBoxValues(F, 0, man->ilower, man->iupper, 0, values);

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
              double               tstop) /* if implicit, returns the number of iters taken */
{
   double time_coeff = 10*(exp(-tstart) - exp(-tstop));
    double *values;
    values = (double *) calloc( man->grid_size, sizeof(double));
//   HYPRE_SStructVector F;
   HYPRE_ParCSRMatrix  parcsr_Phi;
   HYPRE_ParVector parcsr_G, parcsr_F_stop;
   HYPRE_ParVector  parcsr_U_stop, parcsr_U_curr;
   HYPRE_IJVector ij_U_curr, ij_U_stop, ij_F_stop;

   initialize_vector_ij(man, &ij_U_curr);
   HYPRE_IJVectorAssemble(ij_U_curr);
   HYPRE_IJVectorGetObject( ij_U_curr, (void **) &parcsr_U_curr );

   initialize_vector_ij(man, &ij_U_stop);
   HYPRE_IJVectorAssemble(ij_U_stop);
   HYPRE_IJVectorGetObject( ij_U_stop, (void **) &parcsr_U_stop );

   HYPRE_IJMatrixGetObject( man->Phi, (void **) &parcsr_Phi );  
   HYPRE_IJVectorGetObject( man->G, (void **) &parcsr_G );
   
   // Grab these object pointers for use below
   HYPRE_SStructVectorGetBoxValues(U_stop, 0, man->ilower, man->iupper, 0, values);
   HYPRE_IJVectorSetValues( ij_U_stop, man->grid_size, NULL, values );

   /* Create temporary vector */
//   printf("F: ");
//   initialize_vector_parcsr(man, &F);
//   addForcingToRHS( man, tstop, F );
//   HYPRE_SStructVectorAssemble(F);
//   HYPRE_SStructVectorGetObject( F, (void **)&parcsr_F );
//   printf("Pass\n");

   // U^{n} = Phi * U^{n-1} + G^{n}
//   printf("Main: ");
   HYPRE_ParCSRMatrixMatvec(1.0, parcsr_Phi, parcsr_U_stop, 0.0, parcsr_U_curr); 
//   HYPRE_ParVectorAxpy(time_coeff, parcsr_G, parcsr_U_curr);         
//   printf("Pass\n");

   if (F_stop != NULL)
   {
//      printf("F\n");
      initialize_vector_ij(man, &ij_F_stop);
      HYPRE_IJVectorAssemble(ij_F_stop);
      HYPRE_IJVectorGetObject( ij_F_stop, (void **) &parcsr_F_stop );
      
      // Add extra forcing from braid
      HYPRE_SStructVectorGetBoxValues(F_stop, 0, man->ilower, man->iupper, 0, values);
      HYPRE_IJVectorSetValues(ij_F_stop, man->grid_size, NULL, values );
      HYPRE_ParVectorAxpy(1.0, parcsr_F_stop, parcsr_U_curr);         
      
      HYPRE_IJVectorDestroy(ij_F_stop);
   }
   else
   {
         HYPRE_ParVectorAxpy(time_coeff, parcsr_G, parcsr_U_curr);         
   }

   HYPRE_IJVectorGetValues( ij_U_curr, man->grid_size, NULL, values );
   HYPRE_SStructVectorSetBoxValues(U_curr, 0, man->ilower, man->iupper, 0, values);

   /* free memory */
   free(values);
//   HYPRE_SStructVectorDestroy(F);
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
   double time_coeff = 10*(exp(-tstart) - exp(-tstop));

//   HYPRE_SStructVector F;
   HYPRE_ParCSRMatrix parcsr_Phi;
   HYPRE_ParVector parcsr_U_stop, parcsr_G;
   HYPRE_ParVector parcsr_r; 
   HYPRE_IJVector ij_r, ij_U_stop;

   double *values;
   values = (double *) calloc(man->grid_size, sizeof(double));

   initialize_vector_ij( man, &ij_U_stop );
   HYPRE_IJVectorAssemble(ij_U_stop);
   HYPRE_IJVectorGetObject( ij_U_stop, (void **) &parcsr_U_stop );

   initialize_vector_ij( man, &ij_r );
   HYPRE_IJVectorAssemble(ij_r);
   HYPRE_IJVectorGetObject( ij_r, (void **) &parcsr_r );

   // Grab these object pointers for use below
   HYPRE_IJMatrixGetObject( man->Phi, (void **) &parcsr_Phi);
   HYPRE_IJVectorGetObject( man->G, (void **) &parcsr_G );

   HYPRE_SStructVectorGetBoxValues(U_stop, 0, man->ilower, man->iupper, 0, values);
   HYPRE_IJVectorSetValues( ij_U_stop, man->grid_size, NULL, values );

   HYPRE_SStructVectorGetBoxValues(r, 0, man->ilower, man->iupper, 0, values);
   HYPRE_IJVectorSetValues(ij_r, man->grid_size, NULL, values );

   // Create temporary vector 
//   initialize_vector_parcsr(man, &F);
//   addForcingToRHS( man, tstop, F );
//   HYPRE_SStructVectorAssemble(F);
//   HYPRE_SStructVectorGetObject( F, (void **)&parcsr_F );

   HYPRE_ParCSRMatrixMatvec( -1.0, parcsr_Phi, parcsr_r, 1.0, parcsr_U_stop); 
   HYPRE_ParVectorAxpy( -time_coeff, parcsr_G, parcsr_U_stop);

   HYPRE_IJVectorGetValues( ij_U_stop, man->grid_size, NULL, values );
   HYPRE_SStructVectorSetBoxValues(r, 0, man->ilower, man->iupper, 0, values);

   // free memory
   free(values);
//   HYPRE_SStructVectorDestroy(F);
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
   

   values = (double *) calloc( man->grid_size, sizeof(double) );
   HYPRE_SStructVectorGetBoxValues( x, 0, man->ilower, man->iupper, 0, values );
   
   /* Compute error */
   m = 0;
   for( j = 1; j < man->ny; j++ ){
      for( i = 1; i < man->nx; i++ ){
         values[m] =  U_exact(man, 
                              ((man->ilower[0]))*(man->dx), 
                              ((man->ilower[1]))*(man->dy), t) 
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

double L2_norm(simulation_manager  *man, 
                  HYPRE_SStructVector  u) 
{
    // Problem parameters
    int i, j, m, k, n;
   
   double     dt          = man->dt;
   double     dx = man->dx;
   double  dy= man->dy;
   int grid_size = man->grid_size;
   int        nx = man->nx;
   int  ny = man->ny; 
      double    *values, *temp_val;
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


   int Th = 2*nx*ny;

   int index_diff = nx + 2;
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

      if ((down_index+1)%(nx+1) == 0)
      {
        down_index = down_index + 1;
        up_index = up_index + 1;
      }
   }


        // -------------------------------------------------- Space of source term
   temp_val = (double *) calloc( grid_size, sizeof(double) );
   HYPRE_SStructVectorGetBoxValues( u, 0, man->ilower, man->iupper, 0, temp_val );
   values = (double*)calloc((nx+1)*(ny+1), sizeof(double));
   for (j = 1; j < ny; ++j)
   {
      for (i = 1; i < nx; ++i)
      {
         values[i + j*(nx+1)] = temp_val[(i-1) + (j-1)*(nx-1)];
      }
   }
   free(temp_val);


   double p1x, p1y, p2x, p2y, p3x, p3y;
   double u1, u2, u3;
   double p12x, p12y, p32x, p32y;
   double point_x, point_y, U_value;

   error = 0;
   for( m = 0; m < Th; m++ )
   {
      p1x = ((element_node[m].p1)%(nx+1))*dx;
      p1y = ((element_node[m].p1)/(nx+1))*dy;
      p2x = ((element_node[m].p2)%(nx+1))*dx;
      p2y = ((element_node[m].p2)/(nx+1))*dy;
      p3x = ((element_node[m].p3)%(nx+1))*dx;
      p3y = ((element_node[m].p3)/(nx+1))*dy;

//      printf("%d: %d %d %d\n", m, element_node[m].p1, element_node[m].p2, element_node[m].p3);
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
         U_value = 0;//-U_exact(man, point_x, point_y, t); 

         if (m%2 == 0)
         {
            U_value += (u1 + ((u2 - u3)*p1x)/dx + ((u1 - u2)*p1y)/dy + (-u2 + u3)/dx* point_x + (-u1 + u2)/dy* point_y);
         }
         else
         {
            U_value += (u1 + ((u1 - u2)*p1x)/dx + ((u2 - u3)*p1y)/dy + (-u1 + u2)/dx*point_x + (-u2 + u3)/dy*point_y);

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


double compute_L2_error(simulation_manager  *man, 
                  HYPRE_SStructVector  u, 
                  double               t) 
{
    // Problem parameters
    int i, j, m, k, n;
   
   double     dt          = man->dt;
   double     dx = man->dx;
   double  dy= man->dy;
   int grid_size = man->grid_size;
   int        nx = man->nx;
   int  ny = man->ny; 
      double    *values, *temp_val;
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


   int Th = 2*nx*ny;

   int index_diff = nx + 2;
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

      if ((down_index+1)%(nx+1) == 0)
      {
        down_index = down_index + 1;
        up_index = up_index + 1;
      }
   }


        // -------------------------------------------------- Space of source term
   temp_val = (double *) calloc( grid_size, sizeof(double) );
   HYPRE_SStructVectorGetBoxValues( u, 0, man->ilower, man->iupper, 0, temp_val );
   values = (double*)calloc((nx+1)*(ny+1), sizeof(double));
   for (j = 1; j < ny; ++j)
   {
      for (i = 1; i < nx; ++i)
      {
         values[i + j*(nx+1)] = temp_val[(i-1) + (j-1)*(nx-1)];
      }
   }
   free(temp_val);


   double p1x, p1y, p2x, p2y, p3x, p3y;
   double u1, u2, u3;
   double p12x, p12y, p32x, p32y;
   double point_x, point_y, U_value;

   error = 0;
   for( m = 0; m < Th; m++ )
   {
      p1x = ((element_node[m].p1)%(nx+1))*dx;
      p1y = ((element_node[m].p1)/(nx+1))*dy;
      p2x = ((element_node[m].p2)%(nx+1))*dx;
      p2y = ((element_node[m].p2)/(nx+1))*dy;
      p3x = ((element_node[m].p3)%(nx+1))*dx;
      p3y = ((element_node[m].p3)/(nx+1))*dy;

//      printf("%d: %d %d %d\n", m, element_node[m].p1, element_node[m].p2, element_node[m].p3);
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
         U_value = -U_exact(man, point_x, point_y, t); 

         if (m%2 == 0)
         {
            U_value += (u1 + ((u2 - u3)*p1x)/dx + ((u1 - u2)*p1y)/dy + (-u2 + u3)/dx* point_x + (-u1 + u2)/dy* point_y);
         }
         else
         {
            U_value += (u1 + ((u1 - u2)*p1x)/dx + ((u2 - u3)*p1y)/dy + (-u1 + u2)/dx*point_x + (-u2 + u3)/dy*point_y);

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
   int grid_size = man->grid_size;

          // -------------------------------------------------- Print the solution
   int num_values = man->grid_size; 
    double L2error, *values;

    values = (double *)calloc(num_values, sizeof(double)); 
    HYPRE_SStructVectorGetBoxValues(u, 0, man->ilower, man->iupper, 0, values);
    L2error = compute_L2_error(man, u, tstop);
    	printf("L2 error: %0.4e\n", L2error);

    FILE *myfile;
   myfile = fopen("result.txt", "w");

    	fprintf(myfile, "L2 error: %0.10e\n", L2error);
    fprintf(myfile, "Solution:\nPoint(x,y)\t\t\t\tExact\t\t\tNumeric\t\t\tError\n"); 
    for (i = 0; i < num_values; i++) 
    { 
        fprintf(myfile, "(%lf, %lf): %0.10lf \t %0.10lf \t %0.10lf\n", (i%(nx-1) + 1)*dx, (i/(ny-1) + 1)*dy, 
                                                U_exact(man, (i%(nx-1) + 1)*dx, (i/(ny-1) + 1)*dy, 1),
                                                values[i],
                                                fabs(U_exact(man, (i%(nx-1) + 1)*dx, (i/(ny-1) + 1)*dy, 1) - values[i])); 
    } 
    fprintf(myfile,"\n");
    fclose(myfile); 

    free(values);

}