# Solving Travelling Salesman Problem using Parallel Genetic Algorithm
---
### Compiling the code
Assuming a Kepler architecture,

```
cd gpu
nvcc -arch=sm_35 -rdc=true -o dev main.cu
./dev

```
For CPU code:

```
cd cpu
nvcc -arch=sm_35 main_cpu.cu -lcurand
./a.out
```
Each of the versions are very similar to each other and structured as follows:

* `main.cu`: Contains the main function and the kernels for GA operators
* `utils.h`: Contains utils functions for the kernels
* `constants.c`: Declares all the hyperparemeters

For documentation on how the code runs, please check out each individual file in `gpu/`.

