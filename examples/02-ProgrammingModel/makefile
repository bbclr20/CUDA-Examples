
CUDA_FLAGS=-arch sm_60
TARGETS=sumArraysOnHost.out \
        sumArraysOnGpu-small-case.out \
		sumArraysOnGpu-timer.out \
		sumMatrixOnGPU-1D-grid-1D-block.out \
		sumMatrixOnGPU-2D-grid-2D-block.out \
        checkDim.out \
		checkThreadIndex.out \
		checkDeviceInfo.out \
		defineGridBlock.out

.PHONY: build
build: $(TARGETS)

run: sumArraysOnGpu-timer.out
	nvprof --unified-memory-profiling off ./sumArraysOnGpu-timer.out

sumArraysOnHost.out: sumArraysOnHost.c
	nvcc -Xcompiler -std=c99 $^ -o $@

sumArraysOnGpu-small-case.out: sumArraysOnGpu-small-case.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

sumArraysOnGpu-timer.out: sumArraysOnGpu-timer.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

sumMatrixOnGPU-1D-grid-1D-block.out: sumMatrixOnGPU-1D-grid-1D-block.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

sumMatrixOnGPU-2D-grid-2D-block.out: sumMatrixOnGPU-2D-grid-2D-block.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

checkDim.out: checkDim.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

checkThreadIndex.out: checkThreadIndex.cu 
	nvcc $(CUDA_FLAGS) $^ -o $@

checkDeviceInfo.out: checkDeviceInfo.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

defineGridBlock.out: defineGridBlock.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -fr  $(TARGETS)