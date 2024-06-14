#include "ex1.h"

#define COLOR_VALUES 256
#define THREADS_PER_BLOCK 512

__device__
void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    int increment;
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        if(tid >= stride){
            increment = arr[tid - stride];
        }
        __syncthreads();
        if(tid >= stride){
            arr[tid] += increment;
        }
        __syncthreads();
    }
    return;
}

/**
 * Calculates the histogram of a single tile of an image
 * @param histogram int array of size [COLOR_VALUES].
 * @param image_in 2D char array of the input image.
 * @param tile_row_rum the index of the current tile's row in the image.
 * @param tile_col_num the index of the current tile's column in the image.
 */
__device__
void calc_tile_histogram(int* histogram, uchar* image_in, int tile_row_rum, int tile_col_num)
{
    int pixel_value = 0;
    int index_in_img = 0;

    const int tid = threadIdx.x;
    const int rows_group_size = blockDim.x / TILE_WIDTH;
    const int row_index = tile_row_rum * TILE_WIDTH + tid / TILE_WIDTH;
    const int col_index = tile_col_num * TILE_WIDTH + tid % TILE_WIDTH;
    
    for(int i = 0 ; i < TILE_WIDTH ; i+=rows_group_size)
    {
        index_in_img = (row_index + i) * IMG_WIDTH + col_index;
        pixel_value = image_in[index_in_img];
        if (row_index + i < (tile_row_rum + 1) * TILE_WIDTH)
            atomicAdd(&histogram[pixel_value], 1);
    }  
}

/**
 * Calculates the map for the current tile of the histogram equalization
 * @param maps 3D array of size [TILE_COUNT][TILE_COUNT][COLOR_VALUES] that maps each tiles 
 *             gray values from before the equalization to after it.
 * @param tile_row_rum the index of the current tile's row in the image.
 * @param tile_col_num the index of the current tile's column in the image.
 * @param CDF_func int array of the CDF function of the tile's histogram.
 */
__device__
void calc_tile_map(uchar* maps, int tile_row_rum, int tile_col_num, int* CDF_func)
{
    int index_in_map;
    const int tid = threadIdx.x;
    const int numThreads = blockDim.x;
    int work_per_thread = COLOR_VALUES / numThreads;
    if (COLOR_VALUES % numThreads != 0)
        work_per_thread++;
    
    for(int i = 0; i < work_per_thread; i++)
        if(tid + i * numThreads < COLOR_VALUES){
            index_in_map = tile_row_rum * TILE_COUNT * COLOR_VALUES + tile_col_num * COLOR_VALUES + tid + i * numThreads;
            maps[index_in_map] = float(CDF_func[tid + i * numThreads]) * (COLOR_VALUES - 1) / (TILE_COUNT * TILE_COUNT);
        }
}

/**
 * Initiate array with an input value
 * @param arr int array
 * @param length the length of the array
 * @param value the value to initiate the array with
 */
__device__
void array_initiate(int* arr, int length, int value)
{
    const int tid = threadIdx.x;
    const int numThreads = blockDim.x;
    int work_per_thread = COLOR_VALUES / numThreads;
    if (COLOR_VALUES % numThreads != 0)
        work_per_thread++;
    
    for(int i = 0; i < work_per_thread; i++)
        if(tid + i * numThreads < length)
            arr[tid + i * numThreads] = value;
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    
    __shared__ int histogram[sizeof(int) * COLOR_VALUES];
    const int offset_in_img = IMG_HEIGHT * IMG_WIDTH * blockIdx.x;
    const int offset_in_maps = COLOR_VALUES * TILE_COUNT * TILE_COUNT * blockIdx.x;

    for(int tile_row = 0; tile_row < TILE_COUNT; tile_row++)
    {
        for(int tile_col = 0; tile_col < TILE_COUNT; tile_col++){
            array_initiate(histogram, COLOR_VALUES, 0);
            __syncthreads();

            calc_tile_histogram(histogram, all_in + offset_in_img, tile_row, tile_col);
            __syncthreads();

            prefix_sum(histogram, COLOR_VALUES);
            __syncthreads();

            calc_tile_map(maps + offset_in_maps, tile_row, tile_col, histogram);
            __syncthreads();
        }
    }

    interpolate_device(maps + offset_in_maps, all_in + offset_in_img, all_out + offset_in_img);
    __syncthreads();

    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar* image_in; // size of [IMG_HEIGHT][IMG_WIDTH]
    uchar* image_out; // size of [IMG_HEIGHT][IMG_WIDTH]
    uchar* tiles_maps; // size of [TILE_COUNT][TILE_COUNT][COLOR_VALUES]
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    //Allocate GPU memory for a single input image, a single output image, and maps
    CUDA_CHECK(cudaMalloc((void**)&context->image_in, sizeof(char) * IMG_HEIGHT * IMG_WIDTH));
    CUDA_CHECK(cudaMalloc((void**)&context->image_out, sizeof(char) * IMG_HEIGHT * IMG_WIDTH));
    CUDA_CHECK(cudaMalloc((void**)&context->tiles_maps, sizeof(char) * TILE_COUNT * TILE_COUNT * COLOR_VALUES));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    uchar* current_image_in;
    uchar* current_image_out;

    for(int i = 0; i < N_IMAGES; i++){
        current_image_in = images_in + IMG_HEIGHT * IMG_WIDTH * i;
        current_image_out = images_out + IMG_HEIGHT * IMG_WIDTH * i;

        cudaMemcpy(context->image_in, current_image_in, sizeof(uchar) * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice);
        process_image_kernel<<<1, THREADS_PER_BLOCK>>>(context->image_in, context->image_out, context->tiles_maps);
        cudaMemcpy(current_image_out, context->image_out, sizeof(uchar) * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyDeviceToHost);
    }

}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    cudaFree(context->image_in);
    cudaFree(context->image_out);
    cudaFree(context->tiles_maps);

    delete(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    delete(context);
}
