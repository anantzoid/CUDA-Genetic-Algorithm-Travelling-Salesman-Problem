#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>
#include "constants.c"
#include "utils.h"

__device__ float
computeFitnessValue( unsigned char *populationRow, float*populationFitness)
{
    float max = 0;

    for(int i = 0; i < num_chromosomes; i++ ) {
        populationFitness[i] = i;
        if( populationFitness[i] > max) {
            max = populationFitness[i];
        }
    }
    return max;
}

__device__ void
crossover(
        unsigned char *populationRow,
        unsigned char *newPopulation,
        float *selectedPopulation,
        curandState_t *randomState)
{
    int i,j;
    int selectedPhenotype,
        selectedPhenotypeA,
        selectedPhenotypeB;
    int treshold = 0;
    for( i = 0; i < num_chromosomes; i++) {

        selectedPhenotypeA = selectedPopulation[ curand(randomState) % num_chromosomes ];
        selectedPhenotypeB = selectedPopulation[ curand(randomState) % num_chromosomes ];

        treshold = curand(randomState) % chromosome_size;

        for(j = 0; j < chromosome_size; j++) {
            if(j < treshold) {
                selectedPhenotype = selectedPhenotypeA;
            } else {
                selectedPhenotype = selectedPhenotypeB;
            }

            newPopulation[i * chromosome_size + j] =
                    populationRow[selectedPhenotype * chromosome_size];
        }
    }
}

__device__ void
mutation(
        unsigned char *newPopulation,
        curandState_t *randomState)
{
    int i;

    for( i = 0; i < num_chromosomes; i++) {
        if(curand_uniform(randomState) < mutation_ratio) {
            newPopulation[ i* chromosome_size + (curand(randomState) % chromosome_size ) ]
                           = curand(randomState) % MAX_CONFIG_VAL;
        }
    }
}

__device__ void
killPreviousPopulation(
        unsigned char *populationRow,
        unsigned char *newPopulation
)
{
    int i;

    for( i = 0; i < num_chromosomes * chromosome_size; i++) {
        populationRow[i] = newPopulation[i];
    }
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

    tid *= num_cities;

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

/* this GPU kernel function is used to initialize the random states */
__global__ void randomizePopulation(curandState_t* states, unsigned char* population ) {
    int island_y = blockDim.y * blockIdx.y + threadIdx.y;
    int island_x = blockDim.x * blockIdx.x + threadIdx.x;

    //Ques: why is this shared?
    __shared__ curandState_t randomState;
    randomState = states[blockDim.y * blockIdx.y ];

    unsigned char * populationRow = &population[island_y * chromosome_size * num_chromosomes * num_islands + island_x * chromosome_size * num_chromosomes ];

    for(int i = 0; i < chromosome_size * num_chromosomes; i++) {
        populationRow[i] = curand(&randomState) % MAX_CONFIG_VAL;
    };
}

int main() {
   
    cudaError_t err = cudaSuccess;

    int max_val = 250;

    //ag: costTable
    float citymap[num_cities*num_cities];
    
    int population[ISLANDS*num_cities];
    float population_fitness[ISLANDS];
    float population_cost[ISLANDS];
  
    printf("Num islands: %d\n", ISLANDS);
    printf("Population size: %d\n", ISLANDS*num_cities);
     
    float city_x[] = {565,25,345,945,845,880,25,525,580,650};
    float city_y[] = {575,185,750,685,655,660,230,1000,1175,1130};
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
     printf("Initial total fitness: ");
     for(int i=0; i<ISLANDS; i++)
         printf("%.5f ", population_fitness[i]);

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
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in random seed generator (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
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
        /*
        for (int j = 0; j < num_cities; j++){
            crossover<<<num_blocks*num_blocks, num_islands*num_islands>>>(population_d, population_cost_d, population_fitness_d, parent_cities_d, states_d, citymap_d, j); 
        }
        */
    }

/*

    int sizeFloat = sizeof(float);
    int sizeInt = sizeof(unsigned char);

    int populationLength = ISLANDS * chromosome_size * num_chromosomes;
    int sizePopulation = populationLength * sizeInt;
    int sizeBestValue = ISLANDS * sizeFloat;

    int blocksPerGrid = num_blocks*num_blocks;

    unsigned char *cu_populationA = NULL;
    err = cudaMalloc((void **)&cu_populationA, sizePopulation);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector Population (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *cu_bestValue = NULL;
    err = cudaMalloc((void **)&cu_bestValue, sizeBestValue);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector bestValue (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    curandState_t *states = NULL;
    err = cudaMalloc((void**) &states, blocksPerGrid * sizeof(curandState_t));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector randomStates (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    unsigned char *population = (unsigned char *)malloc(sizePopulation);
    float *bestValue = (float*)malloc(sizeBestValue);


    if (population == NULL || bestValue == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    
  
    dim3 dimGrid;
    dimGrid.x = num_blocks;
    dimGrid.y = num_blocks;

    dim3 dimBlock;
    dimBlock.x = num_islands;
    dimBlock.y = num_islands;

    printf("CUDA Init kernel launch with %d blocks of %d threads\n", blocksPerGrid, dimBlock.x * dimBlock.y);
    init<<<dimGrid, 1>>>(time(0), states);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch Init kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    cudaStream_t stream1, stream2;
    cudaStreamCreate ( &stream1) ;
    cudaStreamCreate ( &stream2) ;
    char chunkFileName[20];
    char chunkTargetFileName[20];

    unsigned char *cu_populationLoad = NULL;
    unsigned char *cu_populationUse = NULL;

    float maxTotal = 0;

    printf("Genetic algorithm launch with %d blocks of %d threads\n", dimGrid.x*dimGrid.y, dimBlock.x * dimBlock.y);
    for( int i = 1; i <= num_generations ; i++) {
        printf("====> Generation: %d\n", i);
        //for(int k = 0; k < CHUNKS; k++) {
            cu_populationUse = cu_populationA;
            cu_populationLoad = cu_populationA;
            sprintf(chunkFileName, "chunk.data");
            if(i == 1) {
                randomizePopulation<<<dimGrid, dimBlock, 0, stream1>>>( states, cu_populationUse);
                err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to launch randomizePopulation kernel (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
            }
            if(i !=1){

                //load data for the next chunk
                sprintf(chunkTargetFileName, "chunk.data");
                FILE *ifp = fopen(chunkTargetFileName, "rb");
                fread(population, sizeof(char), sizePopulation, ifp);
                err = cudaMemcpyAsync(cu_populationLoad, population, sizePopulation, cudaMemcpyHostToDevice, stream2);
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to copy data TO device (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
            }


            geneticAlgorithmGeneration<<<dimGrid, dimBlock, 0, stream1>>>(
                    states,
                    cu_populationUse,
                    cu_bestValue
                    );
            cudaDeviceSynchronize();
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch geneticAlgorithmGeneration kernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }


            err = cudaMemcpy(population, cu_populationUse, sizePopulation, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy data FROM device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            FILE *f = fopen(chunkFileName, "wb");
            fwrite(population, sizeof(char), sizePopulation, f);
            fclose(f);

            if( i % 20 == 0 ) {
                float max = 0;
                // Verify that the result vector is correct
                err = cudaMemcpy(bestValue, cu_bestValue, sizeBestValue, cudaMemcpyDeviceToHost);
                if (err != cudaSuccess)
                {
                    fprintf(stderr, "Failed to copy best values from device (error code %s)!\n", cudaGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                for (int i = 0; i < ISLANDS; ++i)
                {
                    if(bestValue[i] > max) {
                        max = bestValue[i];
                    }
                    if(bestValue[i] > maxTotal) {
                        maxTotal = bestValue[i];
                    }

                }
            }

    
        printf("\nMaxTotal %d: %f\n",i, maxTotal);
        printf("\n");
    }


    // Free device global memory
    err = cudaFree(cu_populationA);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(cu_bestValue);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free device global memory
    err = cudaFree(states);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Done\n");
    return 0;
    */
}

