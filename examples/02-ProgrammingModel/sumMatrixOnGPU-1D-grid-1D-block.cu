#include"../common/common.h"
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
    
    for(int iy=0; iy < ny; ++iy){
        for(int ix=0; ix < nx; ++ix){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU1D(float *A, float *B, float *C, int nx, int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;

    if(ix < nx){
        for(int iy=0; iy<ny; ++iy){
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match!!\n");
    else
        printf("Arrays do not match!!\n");
}

int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 1;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 14;
    int ny = 1 << 14;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    
    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    clock_t iStart, iElaps;
    iStart = clock();
    initData(h_A, nxy);
    initData(h_B, nxy);
    iElaps = clock() - iStart;
    printf("Init matrix: %lf s\n", (double)iElaps/CLOCKS_PER_SEC);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = clock();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = clock() - iStart;
    printf("sumMatrixOnHost: %lf s\n", (double)iElaps/CLOCKS_PER_SEC);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 32;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    iStart = clock();
    sumMatrixOnGPU1D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    CHECK(cudaDeviceSynchronize());
    iElaps = clock() - iStart;
    printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>>: %lf s\n", 
            grid.x, grid.y, block.x, block.y, (double)iElaps/CLOCKS_PER_SEC);
    CHECK(cudaGetLastError());


    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return 0;
}