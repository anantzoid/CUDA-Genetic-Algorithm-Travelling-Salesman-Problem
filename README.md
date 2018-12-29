#Solving Travelling Salesman Problem using Parallel Genetic Algorithm

## Anant Gupta (ag4508)
---
### Compiling the code
On cuda1 or cuda5:

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

For documentation on how the code runs, please check out each individual file in `gpu/`. They're cleaned and documented.

