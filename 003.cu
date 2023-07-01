/**
 * shared memory cache-blocking, 1927.7 GOPs
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#define BLOCK_SIZE 32

#define CEIL_DIV(x, y) ((int) ((x + 0.5) / y) )

__global__ void matrix_multiplication_gpu(const float *A, const float *B, float *C, int N){
    
    const uint x = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    const uint y = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

    const uint sub_row = threadIdx.x / BLOCK_SIZE;
    const uint sub_col = threadIdx.x % BLOCK_SIZE;

    __shared__ float buf_a[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float buf_b[BLOCK_SIZE * BLOCK_SIZE];

    if(x < N && y < N){
        float tmp = 0;
        for(int i = 0; i < N; i += BLOCK_SIZE){
            // copy data from global memory to shared memory

            buf_a[sub_row * BLOCK_SIZE + sub_col] = A[(blockIdx.x * BLOCK_SIZE + sub_row) * N + i + sub_col];
            buf_b[sub_row * BLOCK_SIZE + sub_col] = B[(i + sub_row) * N + (blockIdx.y * BLOCK_SIZE + sub_col)];

            __syncthreads();

            for(int j = 0; j < BLOCK_SIZE; ++j){
                tmp += buf_a[sub_row * BLOCK_SIZE + j] * buf_b[(j) * BLOCK_SIZE + sub_col];
            }

            __syncthreads();

        }
        C[x * N + y] = tmp;
    }
}

void matrix_multiplication_cpu(const float *A, const float *B, float *C, int N){
    for(int i = 0; i < N; ++i){
        for(int k = 0; k < N; ++k){
            for(int j = 0; j < N; ++j){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void matrix_init(float * M, int size){
    for(int i = 0; i < size; ++i){
        M[i] = ((float) rand() * 3) / RAND_MAX;
    }
}

void verifygpu(float * C, float * D, int size){
    for(int i = 0; i < size; ++i){
        // printf("%d %f %f\r\n", i, C[i], D[i]);
        if(fabs(C[i] - D[i]) > 1e-3){
            printf("%d %f %f mismatch\r\n", i, C[i], D[i]);
            printf("Test Failed!\r\n");
            return;
        }
    }
    printf("Test Passed!\r\n");
}

int main(int argc, char *argv[]){

    int size = 1024;
    int repeat = 1;
    int verify = 1;

    if(argc > 3){
        size = atoi(argv[1]);
        repeat = atoi(argv[2]);
        verify = atoi(argv[3]);
    }

    float * A = (float *) malloc(sizeof(float) * size * size);
    float * B = (float *) malloc(sizeof(float) * size * size);
    float * C = (float *) malloc(sizeof(float) * size * size);
    float * C_cuda = (float *) malloc(sizeof(float) * size * size);
    srand(time(0));
    matrix_init(A, size * size);
    matrix_init(B, size * size);
    memset(C, 0, sizeof(float) * size * size);

    dim3 gridDim(CEIL_DIV(size, BLOCK_SIZE), CEIL_DIV(size, BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size * size * sizeof(float));
    cudaMalloc(&d_B, size * size * sizeof(float));
    cudaMalloc(&d_C, size * size * sizeof(float));

    cudaMemcpy(d_A, A, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * size * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start, end;
    double time_elapse = 0;
    double perf = 0;

    if(verify){
        start = clock();
        matrix_multiplication_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
        cudaDeviceSynchronize();
        end = clock();
        time_elapse = ((double) end - start) / CLOCKS_PER_SEC;
        perf = (double) size * size * size * 2 / time_elapse / 1e9;
        printf("gpu execution time: %.5f sec, performance: %.3f GOPs\r\n", time_elapse, perf);
        cudaMemcpy(C_cuda, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);
        matrix_multiplication_cpu(A, B, C, size);
        verifygpu(C, C_cuda, size * size);
    }else{
        start = clock();

        for(int rep = 0; rep < repeat; ++rep){
            matrix_multiplication_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
            cudaDeviceSynchronize();
        }

        end = clock();
        time_elapse = ((double) end - start) / CLOCKS_PER_SEC;
        perf = (double) size * size * size * 2. * repeat / time_elapse / 1e9;
        
        cudaError_t err{cudaGetLastError()};
        if (err != cudaSuccess){
            printf("error: %s\r\n", cudaGetErrorString(err));
            // We don't exit when we encounter CUDA errors in this example.
            // std::exit(EXIT_FAILURE);
        }else{
            printf("gpu execution time: %.5f sec, performance: %.3f GOPs\r\n", time_elapse, perf);
        }

    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);
    free(C_cuda);


    return 0;
}