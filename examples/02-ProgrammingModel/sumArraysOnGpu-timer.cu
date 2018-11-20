#include<cuda_runtime.h>
#include<stdio.h>
#include<sys/time.h>
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

__global__ void sumArraysOnGPU(float *A, float *B, float *C, int N){
    int i = blockDim.x  * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char ** argv){
    printf("%s Starting...\n", argv[0]);
    
    // set the device to 1
    int dev = 1;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Trying to use the device %d: %s\n", 
            dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up the size of the vector
    int nElem = 1 << 24;
    printf("Vector size: %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);
    
    // allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    clock_t iStart, iElaps;
    // init data on the host side
    iStart = clock();
    initData(h_A, nElem);
    initData(h_B, nElem);
    iElaps = clock() - iStart;
    printf("Init data on host: %lfs\n", (double)iElaps/CLOCKS_PER_SEC);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    
    // sum Arrays on host
    iStart = clock();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = clock() - iStart;
    printf("Sum array on host: %lfs\n", (double)iElaps/CLOCKS_PER_SEC);
    
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // transfer data from host to GPU
    iStart = clock();
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    iElaps = clock() - iStart;
    printf("Transfer data to GPU: %lfs\n", (double)iElaps/CLOCKS_PER_SEC);
    
    int iLen[] = {512, 1024};
    for(int idx=0; idx<(sizeof(iLen)/sizeof(int)); ++idx){
        dim3 block (iLen[idx]);
        dim3 grid ((nElem+block.x-1)/block.x);

        // sum arrays on GPU
        iStart = clock();
        sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
        CHECK(cudaDeviceSynchronize());
        iElaps = clock() - iStart;
        printf("Sum array on GPU<<<%d, %d>>>: %lfs\n", 
                grid.x, block.x, (double)iElaps/CLOCKS_PER_SEC); // !! juke !!
        // check kernel error
        CHECK(cudaGetLastError());
    }
    
    // transfer data to host
    iStart = clock();
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes,cudaMemcpyDeviceToHost));
    iElaps = clock() - iStart;
    printf("Transfer data to host: %lfs\n", (double)iElaps/CLOCKS_PER_SEC);
    
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