#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "constants.c"
#include "utils.h"

__global__ void mutation(int* population_d, float* population_cost_d, float* population_fitness_d, curandState*  states_d) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= ISLANDS) return;

    curandState localState = states_d[tid];
    if (curand_uniform(&localState) < mutation_ratio) {

        // This gives better score than using Random
        int randNum1 = 1 + curand_uniform(&localState) *  (num_cities - 1.00001);
        int randNum2 = 1 + curand_uniform(&localState) * (num_cities - 1.00001);
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
        tourarray[i] = population_d[tid*num_cities + i];

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

__device__ int* tournamentSelection(int* population_d, float* population_cost_d, 
        float* population_fitness_d, curandState* states_d, int tid) {
    int tournament[tournament_size*num_cities];
    float tournament_fitness[tournament_size];
    float tournament_cost[tournament_size];

    int randNum;
    for (int i = 0; i < tournament_size; i++) {
        // gets random number from global random state on GPU
        randNum = curand_uniform(&states_d[tid]) * (ISLANDS - 1);

        for(int c=0; c<num_cities; c++) {
            tournament[i*num_cities + c] = population_d[randNum*num_cities + c];
            tournament_cost[i] = population_cost_d[randNum];
            tournament_fitness[i] = population_fitness_d[randNum];  
        }
    }
    int fittest = getFittestTourIndex(tournament, tournament_cost, tournament_fitness);
    int fittest_route[num_cities];
    for(int c=0; c<num_cities; c++) {
        fittest_route[c] = tournament[fittest*num_cities + c];
    }
    return fittest_route;
}

__global__ void selection(
        int* population_d, float* population_cost_d,
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

    init<<<num_blocks, num_threads>>>(states_d);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in random seed generator (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    getPopulationFitness<<<num_blocks, num_threads>>>(
            population_d, population_cost_d, population_fitness_d, citymap_d);

    for(int i = 0; i < num_generations; i++ ) {

        selection<<<num_blocks, num_threads>>>(
                population_d, population_cost_d, population_fitness_d, parent_cities_d, states_d);
        //cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch geneticAlgorithmGeneration kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        for (int j = 1; j < num_cities; j++){
            crossover<<<num_blocks, num_threads>>>(population_d, population_cost_d, population_fitness_d, parent_cities_d, states_d, citymap_d, j); 
            //printf("%d", j);
            //cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch Crossover kernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }

        mutation<<<num_blocks, num_threads>>>(
                population_d, population_cost_d, population_fitness_d, states_d);
        //cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch Mutation kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        getPopulationFitness<<<num_blocks, num_threads>>>(
                population_d, population_cost_d, population_fitness_d, citymap_d);
        //cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch Evaluation kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

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
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch geneticAlgorithmGeneration kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


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

