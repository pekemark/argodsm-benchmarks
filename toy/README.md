# Toy ArgoDSM benchmarks

Contains toy benchmarks that are used to evaluate the performance of ArgoDSM.

## Building

Build individually each application using the following

```shell
export ARGO_INSTALL_DIRECTORY=${HOME}/argodsm     # path to argodsm
mpic++ -O3 -fopenmp -std=c++11 -o app app.cpp \   # app.cpp to build
-L${ARGO_INSTALL_DIRECTORY}/build/lib         \
-I${ARGO_INSTALL_DIRECTORY}/src               \
-largo -largobackend-mpi -lnuma -lrt          \
-Wl,-rpath=${ARGO_INSTALL_DIRECTORY}/build/lib
```

This assumes you have already installed [argodsm](https://github.com/etascale/argodsm).

## Benchmarks

## daxpy_strong

Performs the computation: `y += a * x` where `x`, `y`, are two vectors of `double`,
of size `N` and `a` is a scalar.

### Usage

```sh
mpirun $OMPIFLAGS ./daxpy_strong N ITER [CHECK]

where:

N       the size of the vectors x and y
ITER    number of iterations for each to execute the computation
CHECK   an optional parameter that enables checks to make sure the comptuation is correct
```

## matvec-strong

Calculates the matrix-vector product: `y = A * x`, where `A` is a matrix of `M` rows
and `N` columns.

### Usage

```sh
mpirun $OMPIFLAGS ./matvec_strong M N ITER [CHECK]

where:

M       the rows of matrix A
N       the columns of matrix A
ITER    number of iterations for which to execute the computations
CHECK   an optional parameter that enables checks to make sure the comptuation is correct
```
