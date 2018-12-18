float L2distance(float x1, float y1, float x2, float y2) {
    float x_d = pow(x1 - x2, 2);
    float y_d = pow(y1 - y2, 2);
   return sqrt(x_d + y_d); 
}

void evaluateRoute(int* population, float* population_cost, float* population_fitness, float* citymap, int i) {
    float distance = 0;
    for (int j = 0; j < num_cities-1; j++) {
        distance += citymap[population[i*num_cities + j]*num_cities + population[i*num_cities + j+1]];
    }
    distance += citymap[population[i*num_cities + num_cities-1]*num_cities + population[i*num_cities]];

    population_cost[i] = distance;
 
    population_fitness[i] = 0;
    if (population_cost[i] != 0)
        population_fitness[i] = (1.0/population_cost[i]);
}


void initalizeRandomPopulation(int* population, float* population_cost, float* population_fitness, float* citymap) {
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
    evaluateRoute(population, population_cost, population_fitness, citymap, i); 
    } 

} 

int getFittestScore(float* population_fitness) {
    int fittest = 0;
    for(int i=1; i<ISLANDS; i++) {
        if(population_fitness[i] >= population_fitness[fittest])
            fittest = i;
    } 
    return fittest; 
}

int getFittestTourIndex(int* tournament, float* tournament_cost,
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

int find_city(int current_city_id, int* tour, int local_num_cities) {
    for (int i = 0; i < local_num_cities; i++) {
        if (current_city_id == tour[i])
            return i;
    }
    return -1;
}

int getCityN(int n, int* parent_cities_ptr) {
    for (int i = 0; i < num_cities; i++) {
        if (parent_cities_ptr[i] == n)
            return parent_cities_ptr[i];
    }

    return 0;
}

int getValidNextCity(int* parent_cities_ptr, int* tourarray, int current_city_id, int index) {    
   
    //finding current city in parent 
    int local_city_index = find_city(current_city_id, parent_cities_ptr, num_cities);

    // search for first valid city (not already in child) 
    // occurring after currentCities location in parent tour
    for (int i = local_city_index+1; i < num_cities; i++)
    {
        // if not in child already, select it!
        if (find_city(parent_cities_ptr[i], tourarray, index) == -1)
            return parent_cities_ptr[i];
    }

    // loop through city id's [1.. NUM_CITIES] and find first valid city
    // to choose as a next point in construction of child tour
    int inTourAlready;
    for (int i = 1; i < num_cities; i++) {
        inTourAlready = 0;
        for (int j = 1; j < index; j++) {
            if (tourarray[j] == i) {
                inTourAlready = 1;
                break;
            }
        }

        if (inTourAlready == 0)
            return getCityN(i, parent_cities_ptr);
    }
    
    printf("no valid city was found!\n\n");
    return 0;
}   


