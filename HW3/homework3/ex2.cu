/* This file should be almost identical to ex2.cu from homework 2. */
/* once the TODOs in this file are complete, the RPC version of the server/client should work correctly. */

#include "ex3.h"
#include "ex2.h"
#include <cuda/atomic>


#define COLOR_VALUES 256
#define THREADS_PER_BLOCK 1024
#define REGISTERS_PER_THREAD 32
#define EMPTY_STREAM -1
#define EMPTY_QUEUE -1
#define TERMINATE -2
#define SHARED_MEM_USAGE 2048

/****************** Implementation of Image Processing taken from hw2 - Start ******************/

// Requires atleast <size of arr> threads
__device__ void prefix_sum(int arr[], int arr_size) {
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)
    int tid = threadIdx.x;
    int increment;

    for(int stride = 1; stride < arr_size; stride *= 2){
        if(tid >= stride & tid < arr_size){
            increment = arr[tid - stride];
        }
        __syncthreads();
        if(tid >= stride & tid < arr_size){
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
 * @param tile_row_num the index of the current tile's row in the image.
 * @param tile_col_num the index of the current tile's column in the image.
 */
__device__
void calc_tile_histogram(int* histogram, uchar* image_in, int tile_row_num, int tile_col_num)
{
    int pixel_value = 0;
    int index_in_img = 0;

    const int tid = threadIdx.x;
    const int rows_group_size = blockDim.x / TILE_WIDTH;
    const int row_index = tile_row_num * TILE_WIDTH + tid / TILE_WIDTH;
    const int col_index = tile_col_num * TILE_WIDTH + tid % TILE_WIDTH;
    
    for(int i = 0 ; i < TILE_WIDTH ; i+=rows_group_size)
    {
        index_in_img = (row_index + i) * IMG_WIDTH + col_index;
        pixel_value = image_in[index_in_img];
        if (row_index + i < (tile_row_num + 1) * TILE_WIDTH)
            atomicAdd(&histogram[pixel_value], 1);
    }  
}

/**
 * Calculates the map for the current tile of the histogram equalization
 * @param maps 3D array of size [TILE_COUNT][TILE_COUNT][COLOR_VALUES] that maps each tiles 
 *             gray values from before the equalization to after it.
 * @param tile_row_num the index of the current tile's row in the image.
 * @param tile_col_num the index of the current tile's column in the image.
 * @param CDF_func int array of the CDF function of the tile's histogram.
 */
__device__
void calc_tile_map(uchar* maps, int tile_row_num, int tile_col_num, int* CDF_func)
{
    const int tid = threadIdx.x;
    int start_index_in_map = tile_row_num * TILE_COUNT * COLOR_VALUES + tile_col_num * COLOR_VALUES;
    
    const int numThreads = blockDim.x;
    int work_per_thread = COLOR_VALUES / numThreads;
    if (COLOR_VALUES % numThreads != 0)
        work_per_thread++;
    
    for(int i = 0; i < work_per_thread; i++)
        if(tid + i * numThreads < COLOR_VALUES){
            maps[start_index_in_map + tid + i * numThreads] = float(CDF_func[tid + i * numThreads]) * (COLOR_VALUES - 1) / (TILE_WIDTH * TILE_WIDTH);
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
 *             the tilesâ€™ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__
 void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);


__device__ void process_image(uchar *all_in, uchar *all_out, uchar* maps) {
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)
    __shared__ int histogram[COLOR_VALUES];

    for(int tile_row = 0; tile_row < TILE_COUNT; tile_row++)
    {
        for(int tile_col = 0; tile_col < TILE_COUNT; tile_col++){
            array_initiate(histogram, COLOR_VALUES, 0);
            __syncthreads();

            calc_tile_histogram(histogram, all_in, tile_row, tile_col);
            __syncthreads();

            prefix_sum(histogram, COLOR_VALUES);
            __syncthreads();

            calc_tile_map(maps, tile_row, tile_col, histogram);
            __syncthreads();
        }
    }

    interpolate_device(maps, all_in, all_out);
    __syncthreads();

    return; 
}


__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar* maps)
{
    process_image(all_in, all_out, maps);
}

/****************** Implementation of Image Processing taken from hw2 - End ******************/


// TODO complete according to HW2:
//          implement a lock,
//          implement a MPMC queue,
//          implement the persistent kernel,
//          implement a function for calculating the threadblocks count
// (This file should be almost identical to ex2.cu from homework 2.)

/****************** Implementation of Queues taken from hw2 - Start ******************/

// Implement a lock
class GPU_Lock{
    private:
        cuda::atomic<int, cuda::thread_scope_device> _lock;

    public:
        __device__
        GPU_Lock() : _lock(0){}

        ~GPU_Lock() = default;

        __device__
        void lock() {
            while (_lock.exchange(1, cuda::memory_order_relaxed)) { ; }
            cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
        }

        __device__
        void unlock(){
            _lock.store(0, cuda::memory_order_release);
        }
};

__device__ GPU_Lock* gpu_lock;

__global__ void init_gpu_lock(){
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    
    if(tid == 0 && bid == 0){
        gpu_lock = new GPU_Lock();
    }
}

__global__ void destroy_gpu_lock(){
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    
    if(tid == 0 && bid == 0){
        delete gpu_lock;
        gpu_lock = nullptr;
    }
}

// Implement a MPMC queue
struct RequestItem
{
    int image_id;
    uchar* image_in;
    uchar* image_out;

    RequestItem() = default;
    RequestItem(const RequestItem&) = default;
    RequestItem(int id, uchar* in, uchar* out) : image_id(id), image_in(in), image_out(out) {}
};

template <typename T>
class RingBuffer {
    private:
        T* _mailbox;
        size_t N;
        cuda::atomic<size_t> _head = 0, _tail = 0;

    public:
        bool terminate;

        RingBuffer() = default;
        explicit RingBuffer(size_t size) : _mailbox(nullptr), N(size), _head(0), _tail(0), terminate(false){
            CUDA_CHECK(cudaMallocHost(&_mailbox, sizeof(T) * size));
        }

        ~RingBuffer(){
            CUDA_CHECK(cudaFreeHost(_mailbox));
        }

        __device__ __host__
        bool push(const T &data) {
            size_t tail = _tail.load(cuda::memory_order_relaxed);
            if((tail - _head.load(cuda::memory_order_acquire)) % (2 * N) == N) {
                return false;
            }

            _mailbox[_tail % N] = data;
            _tail.store(tail + 1, cuda::memory_order_release);

            return true;
        }

        __device__ __host__
        T pop() {
            T item = RequestItem();
            size_t head = _head.load(cuda::memory_order_relaxed);
            if((_tail.load(cuda::memory_order_acquire) - head) % (2 * N) == 0){
                item.image_id = EMPTY_QUEUE;
                return item;
            }
            else{
                item = _mailbox[_head % N];
            }

            _head.store(head + 1, cuda::memory_order_release);
            return item;
        }

        
        __device__ __host__
        bool is_empty(){
            size_t head = _head.load(cuda::memory_order_relaxed);
            if((_tail.load(cuda::memory_order_acquire) - head) % (2 * N) == 0){
                return true;
            }
            return false;
        }
};


// Implement the persistent kernel
__global__
void persistent_kernel_image_process(RingBuffer<RequestItem>* cpu_gpu_q, RingBuffer<RequestItem>* gpu_cpu_q, uchar* maps){
    __shared__ RequestItem request;
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;

    int map_offset = bid * TILE_COUNT * TILE_COUNT * COLOR_VALUES;
    uchar* cur_tb_maps = maps + map_offset;

    while(true){
        if(cpu_gpu_q->terminate || gpu_cpu_q->terminate){
            return;
        }

        if(tid == 0){
            gpu_lock->lock();
            if(!cpu_gpu_q->is_empty()){
                request = cpu_gpu_q->pop();
            }
            else{
                request.image_id = EMPTY_QUEUE;
            }
            gpu_lock->unlock();
        }
        __syncthreads();

        if(request.image_id == EMPTY_QUEUE){
            continue;
        }

        process_image(request.image_in, request.image_out, cur_tb_maps);
        __syncthreads();

        if(tid == 0){
            gpu_lock->lock();
            while(!gpu_cpu_q->push(request)) {;}
            gpu_lock->unlock();
        }
    }
}


// Implement a function for calculating the threadblocks count
int calculate_threadblocks_amount(int threads_per_block){
    cudaDeviceProp GPU_Properties;
    CUDA_CHECK(cudaGetDeviceProperties(&GPU_Properties, 0));

    // Device constrains
    int share_mem_usage_per_tb = SHARED_MEM_USAGE;
    int regs_per_thread = REGISTERS_PER_THREAD;

    int shared_mem_per_SM = GPU_Properties.sharedMemPerMultiprocessor;
    int max_threads_per_SM = GPU_Properties.maxThreadsPerMultiProcessor;
    int regs_per_SM = GPU_Properties.regsPerMultiprocessor;

    int sm_amount = GPU_Properties.multiProcessorCount;

    // Calculated constrains
    // For each SM we calculate the constrain according to amount of registers,
    // amount of shared memory, and amount of threads.
    int tb_regs_constrain = regs_per_SM / (threads_per_block * regs_per_thread);
    int tb_smem_constrain = shared_mem_per_SM / share_mem_usage_per_tb;
    int tb_threads_constrain = max_threads_per_SM / threads_per_block;

    // printf("The constrains are: reg_per_SM %d, mem_per_SM %d, threads_per_SM %d\n", regs_per_SM, shared_mem_per_SM, max_threads_per_SM);
    // printf("The constrains are: Regs - %d, Memory - %d, Threads - %d\n", tb_regs_constrain, tb_smem_constrain, tb_threads_constrain);

    int max_tb_per_sm = min(tb_regs_constrain, min(tb_smem_constrain, tb_threads_constrain));

    // We return the amount of Threadblocks per SM times the amount of SM's for the total amount of TB.
    return max_tb_per_sm * sm_amount;
}

int calculate_queue_size(int threadblocks_amount){
    return (int)pow(2, ceil(log2(16 * threadblocks_amount) / log2(2)));
}

/****************** Implementation of Queues taken from hw2 - End ******************/


class queue_server : public image_processing_server
{
private:
    RingBuffer<RequestItem>* cpu_gpu_q;
    RingBuffer<RequestItem>* gpu_cpu_q;
    uchar* maps;
public:
    //TODO complete according to HW2
    //(This file should be almost identical to ex2.cu from homework 2.)

    queue_server(int threads)
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        int tb_amount = calculate_threadblocks_amount(threads);
        int q_size = calculate_queue_size(tb_amount);
        
        // printf("Amount of threadblocks is: %d\n", tb_amount);
        // printf("Size of queues is: %d\n", q_size);
        
        // Allocate queues and maps memory
        CUDA_CHECK(cudaMallocHost(&cpu_gpu_q, sizeof(RingBuffer<RequestItem>)));
        CUDA_CHECK(cudaMallocHost(&gpu_cpu_q, sizeof(RingBuffer<RequestItem>)));
        CUDA_CHECK(cudaMalloc(&maps, tb_amount * TILE_COUNT * TILE_COUNT * COLOR_VALUES));

        // Initialize the queues
        new (cpu_gpu_q) RingBuffer<RequestItem>(q_size);
        new (gpu_cpu_q) RingBuffer<RequestItem>(q_size);

        // Launch GPU persistent kernel with given number of threads, and calculated number of threadblocks
        init_gpu_lock<<<1,1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        persistent_kernel_image_process<<<tb_amount, threads>>>(cpu_gpu_q, gpu_cpu_q, maps);
    }

    ~queue_server() override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        cpu_gpu_q->terminate = true;
        gpu_cpu_q->terminate = true;
        CUDA_CHECK(cudaDeviceSynchronize());

        cpu_gpu_q->~RingBuffer<RequestItem>();
        gpu_cpu_q->~RingBuffer<RequestItem>();
        CUDA_CHECK(cudaFreeHost(cpu_gpu_q));
        CUDA_CHECK(cudaFreeHost(gpu_cpu_q));
        CUDA_CHECK(cudaFree(maps));

        destroy_gpu_lock<<<1,1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        return cpu_gpu_q->push(RequestItem(img_id, img_in, img_out));
    }

    bool dequeue(int *img_id) override
    {
        //TODO complete according to HW2
        //(This file should be almost identical to ex2.cu from homework 2.)
        
        // query (don't block) the producer-consumer queue for any responses.
        RequestItem response = gpu_cpu_q->pop();
        if(response.image_id == EMPTY_QUEUE){
            return false;
        }

        // return the img_id of the request that was completed.
        *img_id = response.image_id; 
        return true;
    }
};


std::unique_ptr<queue_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}