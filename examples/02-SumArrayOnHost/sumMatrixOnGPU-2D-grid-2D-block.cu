#include"../common/common.h"
#include<cuda_runtime.h>
#include<stdio.h>
#include<sys/time.h>

void initData(float *ip, const int size){
    for(int i=0; i<size; ++i){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for(int iy=0; iy<ny; ++iy){
        for (int ix=0; ix<nx; ++ix){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int nx, int ny){
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny){
        C[idx] = A[idx] + B[idx];
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1e-8;
    bool match = true;

    for(int i=0; i<N; ++i){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = false;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if(match){
        printf("Arrays match!!\n");
    }else{
        printf("Arrays do not match!!\n");
    }
}

int main(int argc, char **argv){
    printf("%s starting...\n", argv[0]);

    // set up device to 1
    int dev = 1;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));
    printf("Using device: %d, %s\n", dev, deviceProp.name);

    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);
    
    // init arrays and allocate memory
    clock_t iStart, iElaps; 
    iStart = clock();
    initData(h_A, nxy);
    initData(h_B, nxy);
    iElaps = clock() - iStart;
    printf("Init matrix: %lf s\n", (double)iElaps/CLOCKS_PER_SEC);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // sum arrays on the host side
    iStart = clock();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = clock() - iStart;
    printf("sumMatrixOnHost: %lf s\n", (double)iElaps/CLOCKS_PER_SEC);

    // allocate memory on the device
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    // transfer the data to device
    iStart = clock();
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    iElaps = clock() - iStart;
    printf("Transfer data: %lf s\n", (double)iElaps/CLOCKS_PER_SEC);

    int dimx = 32;
    int dimy = 32;
    dim3 block (dimx, dimy);
    dim3 grid((nx + block.x-1)/block.x, (ny + block.y - 1)/ block.y);

    // sum arrays on the device
    iStart = clock();
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = clock() - iStart;
    printf("sumMatrixOnGPU2Ds<<<(%d, %d),(%d, %d)>>>: %lf s\n", 
            grid.x, grid.y, block.x, block.y, (double)iElaps/CLOCKS_PER_SEC);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nxy);

    // release the memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return 0;
}