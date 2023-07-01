/**
 * 2D block tiling, 9648.260 GOPs
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>

#define CEIL_DIV(x, y) ((int) ((x + 0.5) / y))

template <int BLK0, int BLK1, int BLK2, int TM0, int TM1>
__global__ void matrix_multiplication_gpu(const float *A, const float *B, float *C, int N){

    int inner_row_A = threadIdx.x / BLK2;
    int inner_col_A = threadIdx.x % BLK2;

    int inner_row_B = threadIdx.x / BLK1;
    int inner_col_B = threadIdx.x % BLK1;


    __shared__ float buf_a[BLK2 * BLK0];
    __shared__ float buf_b[BLK2 * BLK1];

    float thread_res[TM0 * TM1] = {0.0};

    int out_row = threadIdx.x / TM1;
    int out_col = threadIdx.x % TM1;

    for(int i = 0; i < N; i += BLK2){
        // load a
        for(int j = 0; j < BLK0; j += TM0){
            buf_a[j * BLK2 + threadIdx.x] = A[(blockIdx.y * BLK0 + inner_row_A + j) * N + i + inner_col_A];
        }
        
        for(int j = 0; j < BLK2; ++j){
            buf_b[j * BLK1 + threadIdx.x] = B[(j + inner_row_B + i) * N + blockIdx.x * BLK1 + inner_col_B];
        }
        __syncthreads();
        for(int j = 0; j < BLK2; ++j){
            for(int k = 0; k < TM0; ++k){
                for(int l = 0; l < TM1; ++l){
                    thread_res[k * TM1 + l] += buf_a[out_row * TM0 * BLK2 + j + k * BLK2] * buf_b[(j) * BLK1 + out_col * TM1 + l];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM0; ++i){
        for(int j = 0; j < TM1; ++j){
            C[(blockIdx.y * BLK0 + out_row * TM0 + i) * N + blockIdx.x * BLK1 + out_col * TM1 + j] = thread_res[i * TM1 + j];
        }
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
        M[i] = ((float) rand() * 2) / RAND_MAX;
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

    dim3 gridDim(size / 64, size / 64, 1);
    dim3 blockDim(8 * 8);

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
        matrix_multiplication_gpu<64, 64, 8, 8, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
        cudaDeviceSynchronize();
        end = clock();
        time_elapse = ((double) end - start) / CLOCKS_PER_SEC;
        perf = (double) size * size * size * 2 / time_elapse / 1e9;
        cudaError_t err{cudaGetLastError()};
        if (err != cudaSuccess){
            printf("error: %s\r\n", cudaGetErrorString(err));
        }else{
            printf("gpu execution time: %.5f sec, performance: %.3f GOPs\r\n", time_elapse, perf);
            cudaMemcpy(C_cuda, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);
            matrix_multiplication_cpu(A, B, C, size);
            verifygpu(C, C_cuda, size * size);
        }
    }else{
        start = clock();
        for(int rep = 0; rep < repeat; ++rep){
            matrix_multiplication_gpu<64, 64, 8, 8, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
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