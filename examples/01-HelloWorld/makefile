
CUDA_FLAGS=-arch sm_60
TARGETS=hello.out 

build: $(TARGETS)

hello.out: hello.cu
	nvcc $(CUDA_FLAGS) $^ -o $@

.PHONY: clean
clean:
	rm -fr  $(TARGETS)