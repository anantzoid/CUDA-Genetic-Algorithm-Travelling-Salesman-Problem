#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include <cuda_runtime.h>

const int GENERATIONS = 1;

const int CHECK_VALUES_EVERY = 1;
const int SHOW_ALL_VALUES = 0;
const int SKIP_CUDA_DEVICE = false;


const int ISLANDS_PER_ROW = 4;
const int GENOME_LENGTH=4;
const int BLOCKS_PER_ROW = 16;
const int ISLAND_POPULATION=5;
const int SELECTION_COUNT= 4;
const float MUTATION_CHANCE= 0.8;
const int MAX_CONFIG_VAL = 20;


unsigned int thandle; 

bool IsGpuAvailable()
{
    int devicesCount;
  bool skip = SKIP_CUDA_DEVICE;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if (deviceProperties.major >= 2
            && deviceProperties.minor >= 0)
        {
      if(skip) {
        skip = false;
        continue;
      }
            cudaSetDevice(deviceIndex);
            return true;
        }
    }

    return false;
}


__device__ void
sortByFitness(float*populationFitness, unsigned char* sortedAssoc, float* totalFitness)
{
    int i, j;
    *totalFitness = 1;
    float phenotypeFitness = 0;
    for ( i = 0; i < ISLAND_POPULATION; ++i ){
        sortedAssoc[i] = i;
        phenotypeFitness = populationFitness[i];
        for (
                j = i;
                j > 0 && populationFitness[sortedAssoc[j - 1]] > phenotypeFitness;
                j-- )
        {
            sortedAssoc[j] = sortedAssoc[j - 1];
        }
        sortedAssoc[j] = i;
    }
}

__device__ void
normalizeFitness(float*populationFitness, unsigned char* sortedAssoc, float totalFitness)
{
    int i, j;
    float lastFitness = 0;
    for ( i = 0; i < ISLAND_POPULATION; ++i ){
        j = sortedAssoc[i];
        lastFitness += populationFitness[j];
        populationFitness[j] = lastFitness/totalFitness;
    }
}

__device__ void
selectionTrunc(unsigned char* sortedAssoc, unsigned char* selectedAssoc)
{
    for(int i = 1; i <= SELECTION_COUNT; i++) {
        selectedAssoc[i-1] = sortedAssoc[ISLAND_POPULATION - i];
    }
}

__global__ void primeKernel(int* prime_d, int n) {
    int i = threadIdx.x+ blockIdx.x * blockDim.x;

    for(int b = 2; b <= (i+1)/2; b++){
        if (i%b ==0){
            prime_d[i] = -1;
            break;
        }
    }
}

__device__ float
computeFitnessValue( unsigned char *populationRow, float*populationFitness, int* prime_d, int n)
{
    float max = 0;
    //cutCreateTimer(&thandle);
    for(int i = 0; i < ISLAND_POPULATION; i++ ) {
        //cutStartTimer(thandle);
        primeKernel<<<1, 8>>>(prime_d, n);
        //cutStopTimer(thandle);
        populationFitness[i] = i;//cutGetTimerValue(thandle);
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
        unsigned char *selectedPopulation,
        curandState_t *randomState)
{
    int i,j;
    int selectedPhenotype,
        selectedPhenotypeA,
        selectedPhenotypeB;
    int treshold = 0;
    for( i = 0; i < ISLAND_POPULATION; i++) {

        selectedPhenotypeA = selectedPopulation[ curand(randomState) % SELECTION_COUNT ];
        selectedPhenotypeB = selectedPopulation[ curand(randomState) % SELECTION_COUNT ];

        treshold = curand(randomState) % GENOME_LENGTH;

        for(j = 0; j < GENOME_LENGTH; j++) {
            if(j < treshold) {
                selectedPhenotype = selectedPhenotypeA;
            } else {
                selectedPhenotype = selectedPhenotypeB;
            }

            newPopulation[i * GENOME_LENGTH + j] =
                    populationRow[selectedPhenotype * GENOME_LENGTH];
        }
    }
}

__device__ void
mutation(
        unsigned char *newPopulation,
        curandState_t *randomState)
{
    int i;

    for( i = 0; i < ISLAND_POPULATION; i++) {
        if(curand_uniform(randomState) < MUTATION_CHANCE) {
            newPopulation[ i* GENOME_LENGTH + (curand(randomState) % GENOME_LENGTH ) ]
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

    for( i = 0; i < ISLAND_POPULATION * GENOME_LENGTH; i++) {
        populationRow[i] = newPopulation[i];
    }
}


__global__ void
geneticAlgorithmGeneration(
    curandState_t* states, 
    unsigned char *population,
    float* bestValues, int* prime_d, int n
) 
{

    //index of the island itself
    int island_y = blockDim.y * blockIdx.y + threadIdx.y;
    int island_x = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned char * populationRow = &population[island_y * GENOME_LENGTH * ISLAND_POPULATION * ISLANDS_PER_ROW + island_x * GENOME_LENGTH * ISLAND_POPULATION ];

    __shared__ curandState_t randomState;

    randomState = states[blockDim.x*blockDim.y];

    float populationFitness[ISLAND_POPULATION];

    float best = computeFitnessValue(
            populationRow,
            populationFitness, prime_d, n
            );

    bestValues[island_y * ISLANDS_PER_ROW + island_x] = best;

    unsigned char sortAssoc[ISLAND_POPULATION];
    float totalFitness;

    sortByFitness(populationFitness, sortAssoc, &totalFitness);
    //normalizeFitness(populationFitness, sortAssoc, totalFitness);

    unsigned char selectedAssoc[SELECTION_COUNT];
    selectionTrunc(sortAssoc, selectedAssoc);

    unsigned char  newPopulation[ISLAND_POPULATION*GENOME_LENGTH];
    crossover(populationRow, newPopulation, selectedAssoc, &randomState);
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

    unsigned char * populationRow = &population[island_y * GENOME_LENGTH * ISLAND_POPULATION * ISLANDS_PER_ROW + island_x * GENOME_LENGTH * ISLAND_POPULATION ];

    for(int i = 0; i < GENOME_LENGTH * ISLAND_POPULATION; i++) {
        populationRow[i] = curand(&randomState) % MAX_CONFIG_VAL;
    };
}

/**
 * Host main routine
 */
int
main(void)
{
    if(!IsGpuAvailable()) {
        fprintf(stderr, "Cuda Device is not avaliable!\n");
        exit(EXIT_FAILURE);
    }
    
    cudaError_t err = cudaSuccess;

    /*************************
    *************************
    * Child Kernel memory part
    *************************
    *************************/
    int n = 10;  
    int prime[n+1];
    //Loading the array with numbers from 1 to n
    for(int i = 1; i <= n; i++) {
        prime[i] = i;
    }

    unsigned int num_bytes = (n+1)*sizeof(int);
    int* prime_d;
    err = cudaMalloc((void **)&prime_d, num_bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector prime (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(prime_d, prime, num_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector prime from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    /*************************
    *************************
       * GA memory part
    *************************
    *************************/
    int ISLANDS = ISLANDS_PER_ROW * ISLANDS_PER_ROW * BLOCKS_PER_ROW * BLOCKS_PER_ROW;

    srand(time(NULL));

    int sizeFloat = sizeof(float);
    int sizeInt = sizeof(unsigned char);

    int populationLength = ISLANDS * GENOME_LENGTH * ISLAND_POPULATION;
    int sizePopulation = populationLength * sizeInt;
    int sizeBestValue = ISLANDS * sizeFloat;

    int blocksPerGrid = BLOCKS_PER_ROW*BLOCKS_PER_ROW;

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
    
  
   /* 
    // NOTE These functions will be used if init is done on cpu
    printf("Pop values of pop length: %d\n", populationLength);
    for(int i = 0; i <populationLength; i++) {
        population[i] = rand() % MAX_CONFIG_VAL;
        printf("%d - ", population[i]);
    }
    err = cudaMemcpy( cu_population, population, sizePopulation, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector population from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    */


    dim3 dimGrid;
    dimGrid.x = BLOCKS_PER_ROW;
    dimGrid.y = BLOCKS_PER_ROW;

    dim3 dimBlock;
    dimBlock.x = ISLANDS_PER_ROW;
    dimBlock.y = ISLANDS_PER_ROW;
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
    for( int i = 1; i <= GENERATIONS ; i++) {
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
                    cu_bestValue, prime_d, n
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

            if( i % CHECK_VALUES_EVERY == 0 ) {
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
                    if(SHOW_ALL_VALUES == 1) {
                        printf("%f | ", bestValue[i]);
                    }
                    if(bestValue[i] > max) {
                        max = bestValue[i];
                    }
                    if(bestValue[i] > maxTotal) {
                        maxTotal = bestValue[i];
                    }

                }
            }

    
        if( i % CHECK_VALUES_EVERY == 0 ) {
            printf("\nMaxTotal %d: %f\n",i, maxTotal);
            printf("\n");
        }
    }


    err = cudaFree(prime_d);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device prime_d (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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
    free(population);
    free(bestValue);
    free(prime);

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

