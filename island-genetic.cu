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
    cudaEvent_t start, stop;
    float elapsedTime;
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


__global__ void geneticAlgorithmGeneration(curandState_t* states, unsigned char *population, float* bestValues) {

    //index of the island itself
    int island_y = blockDim.y * blockIdx.y + threadIdx.y;
    int island_x = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned char * populationRow = &population[island_y * chromosome_size * num_chromosomes * num_islands + island_x * chromosome_size * num_chromosomes ];

    __shared__ curandState_t randomState;

    randomState = states[blockDim.x*blockDim.y];

    float populationFitness[num_chromosomes];

    float best = computeFitnessValue(populationRow, populationFitness);

    bestValues[island_y * num_islands + island_x] = best;

    unsigned char  newPopulation[num_chromosomes*chromosome_size];
    crossover(populationRow, newPopulation, populationFitness, &randomState);
    mutation(newPopulation, &randomState);
    killPreviousPopulation(populationRow, newPopulation);
}

/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {

  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
          blockDim.y * blockIdx.y , /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[blockDim.y * blockIdx.y ]);
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

    int ISLANDS = num_islands * num_islands * num_blocks * num_blocks;

    // coordinates of cities
    //ag: tour_t members
    float city_x[num_cities];
    float city_y[num_cities];
    float city_n[num_cities];
    int local_fitness = 0;
    int distance = 0;
    int max_val = 250;

    //ag: costTable
    float fitness[num_cities*num_cities];

    //2. init population
   
    // read initial tour from file 
    FILE* fp; 
    fp = fopen("berlin52.txt", "r");
    char* line = NULL;
    size_t len = 0;
    char* tokens;
    ssize_t read;
     while ((read = getline(&line, &len, fp)) != -1) {
        tokens = strtok(line, " ");
        int n = (int)tokens[0]-1;
        city_x[n] = (float)tokens[1]; 
        city_y[n] = (float)tokens[2]; 
     }
    
     //building cost table
     for(int i=0; i<num_cities; i++) {
         for(int j=0; j<num_cities; j++) {
             if(i!=j) {
                 fitness[i+num_cities+j] = L2distance(city_x[i], city_y[i], city_x[j], city_y[j]);
             } else {
                 fitness[i+num_cities+j] = max_val;
             }
         }
     }
    exit(0);
/*
    {        
        // city coords are in txt file as so:
        // 4 450.3 230.3  -  so, split on spaces
        // index starts from 0
    }

*/




    srand(time(NULL));

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
    /* allocate space on the GPU for the random states */
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
    /* invoke the GPU to initialize all of the random states */

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
    /*
    free(population);
    free(bestValue);
    free(prime);
    */
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
    /*
    for(int x = 0; x < CHUNKS; x++) {
        sprintf(chunkFileName, "chunk%d.data", x);
        remove(chunkFileName);
    }
    */
    printf("Done\n");
    return 0;
}

