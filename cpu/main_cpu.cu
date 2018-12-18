#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include "constants_cpu.c"
#include "utils_cpu.h"

void init(curandGenerator_t gen, float* states) {

    curandCreateGeneratorHost(&gen,
             CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, rand());
    curandGenerateUniform(gen, states, ISLANDS);
    curandDestroyGenerator(gen);
}


void getPopulationFitness(int* population_d, float* population_cost_d, float* population_fitness_d, float* citymap_d) {
    for(int i=0;i<ISLANDS;i++) 
        evaluateRoute(population_d, population_cost_d, population_fitness_d, citymap_d, i); 
}

void mutation(int* population_d, float* population_cost_d, float* population_fitness_d, float*  states_1, float* states_2) {

    //generating new set of random nums for randNum2
    for(int tid=0; tid< ISLANDS; tid++) {
        if (states_1[tid] < mutation_ratio) {

            // This gives better score than using Random
            int randNum1 = 1 + states_1[tid] *  (num_cities - 1.0000001);
            int randNum2 = 1 + states_2[tid] * (num_cities - 1.0000001);
            //printf("%d %d\n", randNum1, randNum2);
            int city_temp = population_d[tid*num_cities + randNum1];

            population_d[tid*num_cities + randNum1] = population_d[tid*num_cities + randNum2];
            population_d[tid*num_cities + randNum2] = city_temp;

        }
    }
}

void crossover(int* population_d, float* population_cost_d,
        float* population_fitness_d, int* parent_cities_d, float* citymap_d, int index) {

    // Get thread (particle) ID

    for(int tid=0; tid< ISLANDS; tid++) {
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

}


int* tournamentSelection(int* population_d, float* population_cost_d, 
        float* population_fitness_d, float* states_d, int tid) {
    int tournament[tournament_size*num_cities];
    float tournament_fitness[tournament_size];
    float tournament_cost[tournament_size];

    int randNum;
    for (int i = 0; i < tournament_size; i++) {
        randNum = states_d[i] * (ISLANDS - 1);
        //printf("%d %d\n", states_d[tid], randNum);

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


void selection(
        int* population_d, float* population_cost_d,
        float* population_fitness_d, int* parent_cities_d, float* states_1, float* states_2) {

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
    for(int tid=0; tid< ISLANDS; tid++) {
        parent1  = tournamentSelection(population_d, population_cost_d, 
                population_fitness_d, states_1, tid);

        for(int c=0; c<num_cities; c++)
            parent_cities_d[tid* (2*num_cities) +c] = parent1[c];

        parent1  = tournamentSelection(population_d, population_cost_d, 
                population_fitness_d, states_2, tid);

        for(int c=0; c<num_cities; c++)
            parent_cities_d[tid* (2*num_cities) +num_cities +c] = parent1[c];

    }
    //}
}

int main() {

    int max_val = 250;

    float citymap[num_cities*num_cities];

    int* population = (int*)calloc(ISLANDS*num_cities, sizeof(int));
    float* population_fitness = (float*)calloc(ISLANDS, sizeof(float));
    float* population_cost = (float*)calloc(ISLANDS, sizeof(float));
    int* parent_cities = (int*)calloc(ISLANDS*num_cities*2, sizeof(int));
    float* states = (float*)calloc(ISLANDS, sizeof(float));
    float* states_2 = (float*)calloc(ISLANDS, sizeof(float));
    curandGenerator_t gen;

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


    float milliseconds;
    clock_t start, end;
              
    start = clock();

    for(int i = 0; i < num_generations; i++ ) {
        init(gen, states);
        init(gen, states_2);
        selection(
                population, population_cost, population_fitness, parent_cities, states, states_2);


        for (int j = 1; j < num_cities; j++)
            crossover(population, population_cost, population_fitness, parent_cities, citymap, j); 

        mutation(population, population_cost, population_fitness, states, states_2);
        
        getPopulationFitness(
                population, population_cost, population_fitness, citymap);


        if(i>0 && i % print_interval == 0) {
            fittest = getFittestScore(population_fitness);
            printf("Iteration:%d, min distance: %f\n", i, population_cost[fittest]);

        }
        //printf("---------------\n");
    }

    end = clock();
    milliseconds = ((double) (end - start)) / CLOCKS_PER_SEC;

    fittest = getFittestScore(population_fitness);
    printf("time: %f,  min distance: %f\n", milliseconds, population_cost[fittest]);

    return 0;
}

