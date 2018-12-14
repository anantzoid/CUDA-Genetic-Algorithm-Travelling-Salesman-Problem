float L2distance(float x1, float y1, float x2, float y2) {
    float x_d = pow(x1 - x2, 2);
    float y_d = pow(y1 - y2, 2);
   return sqrt(x_d + y_d); 
}

