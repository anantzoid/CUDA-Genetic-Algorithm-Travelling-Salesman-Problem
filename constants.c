// Kernel config
const int num_blocks = 2;

// GA config
const int num_generations = 20;
const int num_islands = 2;
const float mutation_ratio= 0.05;
const int ISLANDS = num_islands * num_islands * num_blocks * num_blocks;


const int num_cities = 10;
const int tournament_size = 5;


const float city_x[] = {565,25,345,945,845,880,25,525,580,650};
const float city_y[] = {575,185,750,685,655,660,230,1000,1175,1130};
