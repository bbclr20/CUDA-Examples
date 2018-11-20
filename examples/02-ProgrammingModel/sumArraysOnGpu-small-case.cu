#include<stdlib.h>
#include<string.h>
#include<stdio.h>
#include"../common/common.h"

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0e-8;
    bool match = true;

    for(int i=0; i<N; ++i){
        if (abs(hostRef[i]-gpuRef[i]) > epsilon){
            match = false;
            printf("Array do not match\n!");
            printf("host: %5.2f, gpu: %5.2f at current%d\n", 
                    hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) {
        printf("Array match!!\n");
    }
}

void initData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));
    for(int i=0; i<size; ++i){
        ip[i] = (float)( rand() & 0xFF )/1.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, int N){
    for(int idx=0; idx<N; ++idx){
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C){
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set device to 1
    int dev = 1;
    CHECK(cudaSetDevice(dev));

    // set up the size of the vectors
    int nElem = 32;
    printf("Vector size: %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // init the data
    initData(h_A, nElem);
    initData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // transfer memory from host to the device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at the host side
    dim3 block (nElem);
    dim3 grid (nElem/block.x);

    // compare the computational result
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    printf("Execution configuration: <<<%d, %d>>>\n", grid.x, block.x);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nElem);
    
    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return 0;
}