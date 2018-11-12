#include<stdio.h>

__global__ void greeting(){
    printf("Hello CUDA\n");
}

int main(){
    greeting<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}