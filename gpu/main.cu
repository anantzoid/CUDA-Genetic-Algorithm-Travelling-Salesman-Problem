/*
Title: Solving  Travelling  Salesman  Problem  using  Parallel  Genetic  Algorithm
Author: Anant Gupta (ag4508)
Please see README.txt in order to compile and execute this code
 */

#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "constants.c"
#include "utils.h"

/*
 * Mutation kernel
 */
__global__ void mutation(int* population_d, float* population_cost_d, float* population_fitness_d, curandState*  states_d) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    curandState localState = states_d[tid];
    // Only mutate by random chance
    if (curand_uniform(&localState) < mutation_ratio) {

        // Don't mutate the first city in the route. 
        // Using a float version of 1 as implicit type-cast
        int random_num1 = 1 + curand_uniform(&localState) *  (num_cities - 1.00001);
        int random_num2 = 1 + curand_uniform(&localState) * (num_cities - 1.00001);

        int city_temp = population_d[tid*num_cities + random_num1];
        population_d[tid*num_cities + random_num1] = population_d[tid*num_cities + random_num2];
        population_d[tid*num_cities + random_num2] = city_temp;

        states_d[tid] = localState; 
    }
}

/*
 * Fitness kernel: Evaluates population fitness
 */
__global__ void getPopulationFitness(int* population_d, float* population_cost_d, float* population_fitness_d, float* citymap_d) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    // Calcuates cost and fitness of the route
    evaluateRoute(population_d, population_cost_d, population_fitness_d, citymap_d, tid); 
}

/*
 * Crossover kernel: Perform merging of parents 
 */
__global__ void crossover(int* population_d, float* population_cost_d,
        float* population_fitness_d, int* parent_cities_d, curandState* states_d, float* citymap_d, int index) {

    // Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    // For ease of implementation, the rows are indexed out in registers
    population_d[tid*num_cities] = parent_cities_d[tid* (2*num_cities)];

    int parent_city_ptr[num_cities];
    for(int i=0; i<num_cities;i++)
        parent_city_ptr[i] = parent_cities_d[tid*num_cities*2 + i];

    int tourarray[num_cities];
    for(int i=0; i<num_cities;i++)
        tourarray[i] = population_d[tid*num_cities + i];

    int current_city_id = population_d[tid*num_cities + index - 1];

    // Choose next valid city based on the last one in the route from each parent
    int c1 = getValidNextCity(parent_city_ptr, tourarray, current_city_id, index);

    for(int i=0; i<num_cities;i++)
        parent_city_ptr[i] = parent_cities_d[tid*num_cities*2+num_cities + i];

    int c2 = getValidNextCity(parent_city_ptr, tourarray, current_city_id, index);

    // Keep the better choice from both the parents by checking the one that is closer
    if(citymap_d[c1*num_cities + current_city_id] <= citymap_d[c2*num_cities + current_city_id])
        population_d[tid*num_cities + index] = c1;
    else
        population_d[tid*num_cities + index] = c2;

}

/*
 * Tourname Selection kernel
 * Subroutine of Selection kernel
 * Subsamples a tournament from the existing population and chooses the best
 * candidate route based on fitness 
 */
__device__ int* tournamentSelection(int* population_d, float* population_cost_d, 
        float* population_fitness_d, curandState* states_d, int tid) {
    int tournament[tournament_size*num_cities];
    float tournament_fitness[tournament_size];
    float tournament_cost[tournament_size];

    int random_num;
    for (int i = 0; i < tournament_size; i++) {
        // gets random number from global random state on GPU
        random_num = curand_uniform(&states_d[tid]) * (ISLANDS - 1);

        for(int c=0; c<num_cities; c++) {
            tournament[i*num_cities + c] = population_d[random_num*num_cities + c];
            tournament_cost[i] = population_cost_d[random_num];
            tournament_fitness[i] = population_fitness_d[random_num];  
        }
    }
    int fittest = getFittestTourIndex(tournament, tournament_cost, tournament_fitness);
    int fittest_route[num_cities];
    for(int c=0; c<num_cities; c++) {
        fittest_route[c] = tournament[fittest*num_cities + c];
    }
    return fittest_route;
}

/*
 * Selection kernel: Chooses 2 parent throught tournament selection
 * and stores them in the parent array in global memory
 */
__global__ void selection(int* population_d, float* population_cost_d,
        float* population_fitness_d, int* parent_cities_d, curandState* states_d) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    int* parent1;

    /*
       if(ELITISM && (blockIdx.x == 0)) {

       int fittest = getFittestTourIndex(population_d, population_cost_d, population_fitness_d);
       for(int c=0; c<num_cities; c++) {
       parent_cities_d[tid* (2*num_cities) +c] = population_d[fittest*num_cities + c];
       parent_cities_d[tid* (2*num_cities) +num_cities +c] = population_d[fittest*num_cities + c];
       }


       } else {
     */
    parent1  = tournamentSelection(population_d, population_cost_d, 
            population_fitness_d, states_d, tid);

    for(int c=0; c<num_cities; c++)
        parent_cities_d[tid* (2*num_cities) +c] = parent1[c];

    parent1  = tournamentSelection(population_d, population_cost_d, 
            population_fitness_d, states_d, tid);

    for(int c=0; c<num_cities; c++)
        parent_cities_d[tid* (2*num_cities) +num_cities +c] = parent1[c];

    //}
}


/* this GPU kernel function is used to initialize the random states */
__global__ void init(curandState_t* states) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    curand_init(1337, tid, 0, &states[tid]);
}

/*
 * Main Function
 * Declare relevant variables in host
 * Intialize random tours and adjacecny matrix
 */
int main() {
    cudaSetDevice(1);

    cudaError_t err = cudaSuccess;

    int max_val = 250;

    float citymap[num_cities*num_cities];

    int* population = (int*)calloc(ISLANDS*num_cities, sizeof(int));
    float* population_fitness = (float*)calloc(ISLANDS, sizeof(float));
    float* population_cost = (float*)calloc(ISLANDS, sizeof(float));

    printf("Num islands: %d\n", ISLANDS);
    printf("Population size: %d\n", ISLANDS*num_cities);

    //building cost table
    for(int i=0; i<num_cities; i++) {
        for(int j=0; j<num_cities; j++) {
            if(i!=j) {
                citymap[i*num_cities+j] = L2distance(city_x[i], city_y[i], city_x[j], city_y[j]);
            } else {
                citymap[i*num_cities+j] = max_val * max_val;
            }
        }
    }

    initalizeRandomPopulation(population, population_cost, population_fitness, citymap);

    int fittest = getFittestScore(population_fitness);
    printf("min distance: %f\n", population_cost[fittest]);

    // Device Variables
    int* population_d;
    float* population_fitness_d;
    float* population_cost_d;
    int* parent_cities_d;
    float* citymap_d;
    curandState *states_d;

    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);
    cudaEventRecord (start);

    cudaMalloc((void **)&population_d, ISLANDS*num_cities*sizeof(int));
    cudaMalloc((void **)&population_cost_d, ISLANDS*sizeof(float));
    cudaMalloc((void **)&population_fitness_d, ISLANDS*sizeof(float));
    cudaMalloc((void **)&parent_cities_d, 2*ISLANDS*num_cities*sizeof(int));
    cudaMalloc((void **)&citymap_d, num_cities*num_cities*sizeof(float));
    cudaMalloc((void **)&states_d, ISLANDS*sizeof(curandState));

    cudaMemcpy(population_d, population, ISLANDS*num_cities*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(population_cost_d, population_cost, ISLANDS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(population_fitness_d, population_fitness, ISLANDS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(citymap_d, citymap, num_cities*num_cities*sizeof(float), cudaMemcpyHostToDevice);

    init<<<num_blocks, num_threads>>>(states_d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Init Kernel: %s\n", cudaGetErrorString(err));
        exit(0);
    }

    // Get initial fitness of population
    getPopulationFitness<<<num_blocks, num_threads>>>(
            population_d, population_cost_d, population_fitness_d, citymap_d);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Fitness Kernel: %s\n", cudaGetErrorString(err));
        exit(0);
    }


    for(int i = 0; i < num_generations; i++ ) {

        selection<<<num_blocks, num_threads>>>(
                population_d, population_cost_d, population_fitness_d, parent_cities_d, states_d);
        //cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Selection Kernel: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        for (int j = 1; j < num_cities; j++){
            crossover<<<num_blocks, num_threads>>>(population_d, population_cost_d, population_fitness_d, parent_cities_d, states_d, citymap_d, j); 
            //printf("%d", j);
            //cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Crossover Kernel: %s\n", cudaGetErrorString(err));
                exit(0);
            }

        }

        mutation<<<num_blocks, num_threads>>>(
                population_d, population_cost_d, population_fitness_d, states_d);
        //cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Mutation Kernel: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        getPopulationFitness<<<num_blocks, num_threads>>>(
                population_d, population_cost_d, population_fitness_d, citymap_d);
        //cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Mutation Kernel: %s\n", cudaGetErrorString(err));
            exit(0);
        }

        // Print things for sanity check
        if(i > 0 && i % print_interval == 0) {
            cudaMemcpy(population_fitness, population_fitness_d,  ISLANDS*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(population_cost, population_cost_d,  ISLANDS*sizeof(float), cudaMemcpyDeviceToHost);
            fittest = getFittestScore(population_fitness);
            printf("Iteration:%d, min distance: %f\n", i, population_cost[fittest]);
        }
    }

    cudaEventRecord (stop);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&milliseconds, start, stop);

    cudaMemcpy(population, population_d,  ISLANDS*num_cities*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(population_fitness, population_fitness_d,  ISLANDS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(population_cost, population_cost_d,  ISLANDS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    fittest = getFittestScore(population_fitness);
    printf("time: %f,  min distance: %f\n", milliseconds/1000, population_cost[fittest]);

    cudaFree(population_d);
    cudaFree(population_fitness_d);
    cudaFree(population_cost_d);
    cudaFree(parent_cities_d);
    cudaFree(citymap_d);
    cudaFree(states_d);

    return 0;
}

