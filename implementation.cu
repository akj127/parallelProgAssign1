#include "implementation.h"
#include "stdio.h"

void printSubmissionInfo()
{
    // This will be published in the leaderboard on piazza
    // Please modify this field with something interesting
    char nick_name[] = "default-team";

    // Please fill in your information (for marking purposes only)
    char student_first_name[] = "John";
    char student_last_name[] = "Doe";
    char student_student_number[] = "00000000";

    // Printing out team information
    printf("*******************************************************************************************************\n");
    printf("Submission Information:\n");
    printf("\tnick_name: %s\n", nick_name);
    printf("\tstudent_first_name: %s\n", student_first_name);
    printf("\tstudent_last_name: %s\n", student_last_name);
    printf("\tstudent_student_number: %s\n", student_student_number);
}
#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 1024  // Number of threads per block

/**
 * Warp-level inclusive scan using shuffle instructions.
 */
__inline__ __device__
int32_t warp_scan(int32_t val) {
    // Inclusive scan within a warp using shfl_up_sync
    for (int offset = 1; offset < 32; offset <<= 1) {
        int32_t n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset)
            val += n;
    }
    return val;
}

/**
 * Kernel for block-level inclusive scan using shared memory and warp-level primitives.
 */
__global__
void inclusive_scan_kernel(const int32_t* d_in, int32_t* d_out, int32_t* d_block_sums, size_t n) {
    __shared__ int32_t s_data[BLOCK_SIZE / 32];  // Shared memory to hold warp sums

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t val = 0;

    if (gid < n)
        val = d_in[gid];

    // Perform warp-level inclusive scan
    val = warp_scan(val);

    // Write the sum of each warp to shared memory
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 31)
        s_data[warp_id] = val;

    __syncthreads();

    // Let the first thread of each warp read the sums from shared memory
    int32_t warp_sum = 0;
    if (warp_id == 0 && threadIdx.x < blockDim.x / 32) {
        warp_sum = s_data[threadIdx.x];
        warp_sum = warp_scan(warp_sum);
        s_data[threadIdx.x] = warp_sum;
    }

    __syncthreads();

    // Each thread adds the sum of previous warps
    if (warp_id > 0)
        val += s_data[warp_id - 1];

    // Write the result to global memory
    if (gid < n)
        d_out[gid] = val;

    // Save the total sum of this block to d_block_sums
    if (threadIdx.x == blockDim.x - 1)
        d_block_sums[blockIdx.x] = val;
}

/**
 * Kernel to add scanned block sums to each element in the block.
 */
__global__
void add_block_sums(int32_t* d_data, const int32_t* d_block_sums, size_t n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && gid < n) {
        d_data[gid] += d_block_sums[blockIdx.x - 1];
    }
}

/**
 * Recursive function to perform multi-level inclusive scan.
 */
void scan_impl(const int32_t* d_in, int32_t* d_out, size_t n) {
    int num_threads = BLOCK_SIZE;
    int num_blocks = (n + num_threads - 1) / num_threads;

    int32_t* d_block_sums = nullptr;
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int32_t));

    // First level scan
    inclusive_scan_kernel<<<num_blocks, num_threads>>>(d_in, d_out, d_block_sums, n);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_block_sums);
        return;
    }

    // If there are multiple blocks, perform recursive scan on block sums
    if (num_blocks > 1) {
        int32_t* d_block_sums_scan = nullptr;
        cudaMalloc(&d_block_sums_scan, num_blocks * sizeof(int32_t));

        // Recursively call scan_impl on block sums
        scan_impl(d_block_sums, d_block_sums_scan, num_blocks);

        // Add scanned block sums to each element
        add_block_sums<<<num_blocks, num_threads>>>(d_out, d_block_sums_scan, n);

        // Check for errors in kernel launch
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            cudaFree(d_block_sums_scan);
            cudaFree(d_block_sums);
            return;
        }

        cudaFree(d_block_sums_scan);
    }

    cudaFree(d_block_sums);
}

/**
 * Implement your CUDA inclusive scan here. Feel free to add helper functions, kernels or allocate temporary memory.
 * However, you must not modify other files. CAUTION: make sure you synchronize your kernels properly and free all
 * allocated memory.
 *
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */
void implementation(const int32_t *d_input, int32_t *d_output, size_t size) {
    // Perform inclusive scan
    scan_impl(d_input, d_output, size);

    // Ensure all kernels have finished executing
    cudaDeviceSynchronize();
}
