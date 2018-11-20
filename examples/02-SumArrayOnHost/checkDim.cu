#include<stdio.h>
#include<cuda_runtime.h>
#include"../common/common.h"

__global__ void checkIndexs(){
    printf("blockIdx: (%d, %d, %d), threadIdx: (%d, %d, %d)\n",
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, threadIdx.y, threadIdx.z);
    printf("gridDim: (%d, %d, %d), blockDim: (%d, %d, %d)\n",
            gridDim.x, gridDim.y, gridDim.z,        
            blockDim.x, blockDim.y, blockDim.z);
}

int main(){
    int nElem = 6;

    dim3 block (3, 1, 1);
    dim3 grid ((nElem+block.x - 1)/block.x);

    printf("grid.x: %d, grid.y: %d, grid.z: %d\n",
            grid.x, grid.y, grid.z);

    printf("block.x: %d, block.y: %d, block.z: %d\n",
            block.x, block.y, block.z);

    checkIndexs<<<grid, block>>>();
    CHECK(cudaDeviceReset());

    return 0;
}