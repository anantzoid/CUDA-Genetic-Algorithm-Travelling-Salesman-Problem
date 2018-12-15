#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>
#include "constants.c"
#include "utils.h"

__global__ void mutation(int* population_d, float* population_cost_d, float* population_fitness_d, curandState*  states_d) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    curandState localState = states_d[tid];
    if (curand_uniform(&localState) < mutation_ratio) {
        int randNum1 = 1 + curand_uniform(&localState) *  (num_cities - 1.0000001);
        int randNum2 = 1 + curand_uniform(&localState) * (num_cities - 1.0000001);

        int city_temp = population_d[tid*num_cities + randNum1];
        population_d[tid*num_cities + randNum1] = population_d[tid*num_cities + randNum2];
        population_d[tid*num_cities + randNum2] = city_temp;

        states_d[tid] = localState; 
    }
}

__global__ void getPopulationFitness(int* population_d, float* population_cost_d, float* population_fitness_d, float* citymap_d) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    evaluateRoute(population_d, population_cost_d, population_fitness_d, citymap_d, tid); 
}

__global__ void crossover(int* population_d, float* population_cost_d,
        float* population_fitness_d, int* parent_cities_d, curandState* states_d, float* citymap_d, int index) {

    // Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    population_d[tid*num_cities] = parent_cities_d[tid* (2*num_cities)];

    int parent_city_ptr[num_cities];
    for(int i=0; i<num_cities;i++)
        parent_city_ptr[i] = parent_cities_d[tid*num_cities*2 + i];

    int tourarray[num_cities];
    for(int i=0; i<num_cities;i++)
        tourarray[i] = tourarray[tid*num_cities + i];

    int current_city_id = population_d[tid*num_cities + index - 1];

    int c1 = getValidNextCity(parent_city_ptr, tourarray, current_city_id, index);


    for(int i=0; i<num_cities;i++)
        parent_city_ptr[i] = parent_cities_d[tid*num_cities*2+num_cities + i];

    int c2 = getValidNextCity(parent_city_ptr, tourarray, current_city_id, index);
   
    if(citymap_d[c1*num_cities + current_city_id] <= citymap_d[c2*num_cities + current_city_id])
        population_d[tid*num_cities + index] = c1;
    else
        population_d[tid*num_cities + index] = c2;

}

__device__ int getFittestTourIndex(int* tournament, float* tournament_cost,
        float* tournament_fitness) {
    int fittest = 0;
    float fitness = tournament_fitness[0];

    for (int i = 1; i < tournament_size-1; i++) {
        if (tournament_fitness[i] >= fitness) {
            fittest = i;
            fitness = tournament_fitness[i];        
        }
    }
    return fittest;
}

__device__ int* tournamentSelection(int* population_d, float* population_cost_d, 
        float* population_fitness_d, curandState* states_d, int tid)
{
    int tournament[tournament_size*num_cities];
    float tournament_fitness[tournament_size];
    float tournament_cost[tournament_size];
    
    int randNum;
    for (int i = 0; i < tournament_size; i++) {
        // gets random number from global random state on GPU
        randNum = curand_uniform(&states_d[tid]) * (ISLANDS - 1);
        //printf("%d %d ", tid, &states_d[tid]);
        for(int c=0; c<num_cities; c++) {
            tournament[i*num_cities + c] = population_d[randNum*num_cities + c];
            tournament_cost[i] = population_cost_d[i];
            tournament_fitness[i] = population_fitness_d[i];  
        }
    }
    int fittest = getFittestTourIndex(tournament, tournament_cost, tournament_fitness);
    //printf("%d %d %.5f\n", tid, fittest, tournament_fitness[fittest]);
    int fittest_route[num_cities];
    for(int c=0; c<num_cities; c++) {
        fittest_route[c] = tournament[fittest*num_cities + c];
    }
    //printf("\n");
    return fittest_route;
}


__global__ void geneticAlgorithmGeneration(
        int* population_d, float* population_cost_d,
        float* population_fitness_d, int* parent_cities_d, curandState* states_d) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    int* parent1;
    parent1  = tournamentSelection(population_d, population_cost_d, 
            population_fitness_d, states_d, tid);
     
    for(int c=0; c<num_cities; c++)
        parent_cities_d[tid* (2*num_cities) +c] = parent1[c];
    parent1  = tournamentSelection(population_d, population_cost_d, 
            population_fitness_d, states_d, tid);

    for(int c=0; c<num_cities; c++)
        parent_cities_d[tid* (2*num_cities) +num_cities +c] = parent1[c];
}


/* this GPU kernel function is used to initialize the random states */
__global__ void init(curandState_t* states) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;
    
    curand_init(1337, tid, 0, &states[tid]);

}

int main() {
    cudaSetDevice(1);
   
    cudaError_t err = cudaSuccess;

    int max_val = 250;

    //ag: costTable
    float citymap[num_cities*num_cities];
    
    int population[ISLANDS*num_cities];
    float population_fitness[ISLANDS];
    float population_cost[ISLANDS];
  
    printf("Num islands: %d\n", ISLANDS);
    printf("Population size: %d\n", ISLANDS*num_cities);
     
     //building cost table
     for(int i=0; i<num_cities; i++) {
         for(int j=0; j<num_cities; j++) {
             if(i!=j) {
                 citymap[i*num_cities+j] = L2distance(city_x[i], city_y[i], city_x[j], city_y[j]);
             } else {
                 citymap[i*num_cities+j] = max_val;
             }
         }
     }

     initalizeRandomPopulation(population, population_cost, population_fitness, citymap);
     printf("Initial Routes: ");
     for(int i=0; i<ISLANDS*num_cities; i++)
         printf("%d ", population[i]);
     printf("\n");
     printf("Initial total costs: ");
     for(int i=0; i<ISLANDS; i++)
         printf("%.2f ", population_cost[i]);
     printf("\n");
     printf("Initial total fitness: ");
     for(int i=0; i<ISLANDS; i++)
         printf("%.5f ", population_fitness[i]);
     printf("\n");

    //////////////
     // GPU data
    //////////////
     int* population_d;
     float* population_fitness_d;
     float* population_cost_d;
     int* parent_cities_d;
     float* citymap_d;
     curandState *states_d;

     err = cudaMalloc((void **)&population_d, ISLANDS*num_cities*sizeof(int));
     if (err != cudaSuccess) {
         fprintf(stderr, "Error on population Malloc (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
     err = cudaMalloc((void **)&population_cost_d, ISLANDS*sizeof(float));
     if (err != cudaSuccess) {
         fprintf(stderr, "Error on population_cost Malloc (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     err = cudaMalloc((void **)&population_fitness_d, ISLANDS*sizeof(float));

     err = cudaMalloc((void **)&parent_cities_d, 2*ISLANDS*num_cities*sizeof(int));
     if (err != cudaSuccess) {
         fprintf(stderr, "Error on parent_cities Malloc (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     err = cudaMalloc((void **)&citymap_d, num_cities*num_cities*sizeof(float));
     if (err != cudaSuccess) {
         fprintf(stderr, "Error on citymap Malloc (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     err = cudaMalloc((void **)&states_d, ISLANDS*sizeof(curandState));
     if (err != cudaSuccess) {
         fprintf(stderr, "Error on rand_state Malloc (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }

     // TODO error reporting
     err = cudaMemcpy(population_d, population, ISLANDS*num_cities*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "ffdfdndom seed generator (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     err = cudaMemcpy(population_cost_d, population_cost, ISLANDS*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "2Error in random seed generator (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     err = cudaMemcpy(population_fitness_d, population_fitness, ISLANDS*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "3Error in random seed generator (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
     err = cudaMemcpy(citymap_d, citymap, num_cities*num_cities*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "4Error in random seed generator (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float milliseconds;
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);
    cudaEventRecord (start);

    init<<<num_blocks*num_blocks, num_islands*num_islands>>>(states_d);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in random seed generator (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    getPopulationFitness<<<num_blocks*num_blocks, num_islands*num_islands>>>(
        population_d, population_cost_d, population_fitness_d, citymap_d);

    for(int i=0; i < num_generations; i++ ) {

        geneticAlgorithmGeneration<<<num_blocks*num_blocks, num_islands*num_islands>>>(
                population_d, population_cost_d, population_fitness_d, parent_cities_d, states_d);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch geneticAlgorithmGeneration kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < num_cities; j++){
            crossover<<<num_blocks*num_blocks, num_islands*num_islands>>>(population_d, population_cost_d, population_fitness_d, parent_cities_d, states_d, citymap_d, j); 
        }

        mutation<<<num_blocks*num_blocks, num_islands*num_islands>>>(
                population_d, population_cost_d, population_fitness_d, states_d);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch Mutation kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        getPopulationFitness<<<num_blocks*num_blocks, num_islands*num_islands>>>(
                population_d, population_cost_d, population_fitness_d, citymap_d);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch Evaluation kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


    }

    ///Fin/////
    cudaEventRecord (stop);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&milliseconds, start, stop);

    cudaMemcpy(population, population_d,  ISLANDS*num_cities*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(population_fitness, population_fitness_d,  ISLANDS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(population_cost, population_cost_d,  ISLANDS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch geneticAlgorithmGeneration kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    //get fitness
    int fittest = 0;
    for(int i=1; i<ISLANDS; i++) {
        if(population_fitness[i] >= population_fitness[fittest])
            fittest = i;
    } 

    printf("time: %f,  min distance: %f\n", milliseconds/1000, population_cost[fittest]);

    cudaFree(population_d);
    cudaFree(population_fitness_d);
    cudaFree(population_cost_d);
    cudaFree(parent_cities_d);
    cudaFree(citymap_d);
    cudaFree(states_d);

    return 0;
}

