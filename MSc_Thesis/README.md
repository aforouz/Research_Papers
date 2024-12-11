# M.Sc. thesis Codes
These codes are implimentations of numerical approchese which used in this thesis.
For using xbraid, you must use Linux (or wsl for windows) and install:
- [MPI](https://www.mpich.org/),
- [HYPRE](https://github.com/hypre-space/hypre/),
- [XBraid](https://github.com/XBraid/xbraid/).

This thesis is based on:
- Yue, X., Shu, S., Xu, X., Bu, W., & Pan, K. (2019). Parallel-in-time multigrid for space–time finite element approximations of two-dimensional space-fractional diffusion equations. Comput. Math. Appl., 78. 3471–3484.

## References
- [Parallel-in-time multigrid for space–time FE approximations of 2D SFDEs](https://doi.org/10.1016/j.camwa.2019.05.017),
- [HYPRE](https://computing.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods),
- [XBraid](https://computing.llnl.gov/projects/parallel-time-integration-multigrid).

## Abstract
Nowadays, temporal or spatial fractional partial differential equations (FPDEs) have found applications in real-world problems in science and engineering. 
This popularity stems from the fractional derivative's nonlocal property compared to the local nature of the integer-order derivative. 
One of the most important FPDEs is the space-fractional diffusion equation (SFD) which has applications in modeling anomalous diffusion, investigating subdiffusive phenomena, and describing chaotic dynamics.
SFD over two-dimensional spaces is widely recognized as a key diffusion equation. 
It is derived from generalizing the spatial derivatives from integer order to fractional order within the partial differential equation. 
Since most SFD equations can not be solved analytically, various numerical methods, such as the finite difference method, local discontinuous Galerkin approach, and finite element method (FEM), have been proposed to achieve both high accuracy and efficiency.
It is important to note that fractional derivatives use global information, while classical derivatives rely on local information. 
As a result, regardless of the discretization method used, significant computational effort is required due to the nonlocality introduced by fractional differential operators. 
Many researchers have worked on developing fast algorithms to address this challenge. 
In addition to these rapid solutions, parallel computing approaches, such as multigrid reduction in time (MGRIT), should also be considered potential techniques. 
In this thesis, we begin by discussing fundamental concepts in functional analysis, including vector spaces, function spaces, and Sobolev spaces, as well as the principles of fractional calculus. 
We explain that fractional derivatives and integrals are foundational to fractional calculus, with the Ritz fractional derivative being particularly favored for applications in spatial domains. 
Next, we examine various fractional spaces, such as fractional Sobolev spaces, and investigate their properties.
Additionally, we address spaces associated with FPDEs. 
After that, we investigate the SFD problem with Dirichlet boundary conditions. 
We start by explaining the FEM and its properties. 
We construct the weak form of the SFD equation and apply space-time discretization, utilizing uniform spatial discretization and non-uniform temporal discretization. 
This process results in a large, sparse system of equations.
To solve the SFD equation numerically, we represent the method as a time-marching loop, where a spatial linear system is solved at each time step. 
This time-marching loop acts as a one-step temporal method, similar to solving a lower bidiagonal block system over time. 
We also discuss various schemes for temporal parallelization and provide a brief historical context.
One notable technique is MGRIT, which utilizes a multigrid reduction approach. 
The MGRIT offers two significant advantages: it minimizes interference with existing codes and allows optimal scalability. 
Subsequently, we employ a two-stage version of the MGRIT method to solve the lower bidiagonal system of the block unit and analyzing its convergence performance. 
Finally, we implement a numerical example in MATLAB and XBraid and examine some tests. 
The results indicate that the method demonstrates adequate consistency and convergence for numerical solutions of such SFD equations, and it can be extended to solve some complicated FPDEs.

## Key words
- Fractional calculus
- Space-fractional diffusion equations (SFDEs)
- Finite elements method (FEM)
- Parallel-in-time (PinT)
- Multigrid reduction-in-time (MGRIT)
- XBraid
- HYPRE