float L2distance(float x1, float y1, float x2, float y2) {
    float x_d = pow(x1 - x2, 2);
    float y_d = pow(y1 - y2, 2);
   return sqrt(x_d + y_d); 
}
__host__ __device__ void findPopulationCost(int* population,float* population_cost, float* fitness) {
    float distance;
    
    for (int i = 0; i < ISLANDS; i++) {
        distance = 0;
        for (int j = 0; j < num_cities-1; j++) {
            distance += fitness[population[i*num_cities + j]*num_cities + population[i*num_cities + j+1]];
        }
        distance += fitness[population[i*num_cities + num_cities-1]*num_cities + population[i*num_cities]];
        population_cost[i] = distance;
    }
}
void initalizeRandomPopulation(int* population, float* population_cost, float* fitness) {
    int linear_tour[num_cities];
    for (int j = 0; j < num_cities; j++) {
        linear_tour[j] = j;
        population[j] = j;
    }

    int temp_tour[num_cities];
    for (int i = 0; i < ISLANDS; i++) {
        memcpy(temp_tour, linear_tour, num_cities * sizeof(float));

        for (int j = 1; j < num_cities; j++) {
            int pos = 1 + (rand()%(num_cities-1));
            int temp = temp_tour[j];
            temp_tour[j] = temp_tour[pos];
            temp_tour[pos] = temp;

        }

        for (int j = 0; j < num_cities; j++) {
            population[i*num_cities + j] = temp_tour[j];
        }
    } 

    findPopulationCost(population, population_cost, fitness); 
} 

