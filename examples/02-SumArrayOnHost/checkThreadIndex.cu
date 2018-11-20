#include<cuda_runtime.h>
#include<stdio.h>
#include"../common/common.h"

void initInt(int *ip, int size){
    for(int i=0; i<size; ++i){
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("Matrix: (%d, %d)\n", nx, ny);
    for(int iy=0; iy<ny; ++iy){
        for(int ix=0; ix<nx; ++ix){
            printf("%3d ", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int idx = iy * nx + ix;
    printf("Thread_id: (%d,%d), Block_id: (%d, %d), "
           " Coordinate: (%d, %d), global index: %d, value: %d\n", 
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
           ix, iy, idx, A[idx]);
}

int main(int argc, char** argv){
    // set to device 1
    int dev = 1;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaSetDevice(dev);

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    // allocate host memory
    int *h_A;
    h_A = (int *)malloc(nBytes);
    initInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // allocate device memory
    int *d_A;
    cudaMalloc((int **)&d_A, nBytes);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block (4, 2);
    dim3 grid ((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    printThreadIndex<<<grid, block>>>(d_A, nx, ny);
    cudaDeviceSynchronize();
    cudaGetLastError();

    cudaFree(d_A);
    free(h_A);

    cudaDeviceReset();
    return 0;
}