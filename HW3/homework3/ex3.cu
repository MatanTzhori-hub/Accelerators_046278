/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>

#include <stddef.h>  // for offsetof

/****************** RCP Implementation - Start ******************/

class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;

public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_in;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];

                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    post_rdma_read(
                        img_in,             // local_src
                        req->input_length,  // len
                        mr_images_in->lkey, // lkey
                        req->input_addr,    // remote_dst
                        req->input_rkey,    // rkey
                        wc.wr_id);          // wr_id
                    break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    req = &requests[wc.wr_id];
                    img_in = &images_in[wc.wr_id * IMG_SZ];
                    img_out = &images_out[wc.wr_id * IMG_SZ];

                    // Step 3: Process on GPU
                    while(!gpu_context->enqueue(wc.wr_id, img_in, img_out)){};
		    break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

		    if (terminate)
			got_last_cqe = true;

                    break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_img_id;
            if (gpu_context->dequeue(&dequeued_img_id)) {
                req = &requests[dequeued_img_id];
                img_out = &images_out[dequeued_img_id * IMG_SZ];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_img_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_LOCAL_WRITE);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = img_id;
        req->input_rkey = img_in ? mr_images_in->rkey : 0;
        req->input_addr = (uintptr_t)img_in;
        req->input_length = IMG_SZ;
        req->output_rkey = img_out ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)img_out;
        req->output_length = IMG_SZ;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = img_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *img_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL)) ;
        int img_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&img_id);
        } while (!dequeued || img_id != -1);
    }
};

/****************** RCP Implementation - End ******************/

/****************** RDMA Implementation - Start ******************/

struct remote_information{
    // Server images buffers
    uint64_t images_in_addr;
    uint32_t images_in_rkey;
    uint64_t images_out_addr;
    uint32_t images_out_rkey;

    // CPU to GPU
    uint64_t cpu_gpu_q_addr;
    uint32_t cpu_gpu_q_rkey; 
    uint64_t cpu_gpu_q_mbx_addr;
    uint32_t cpu_gpu_q_mbx_rkey; 

    // GPU to CPU
    uint64_t gpu_cpu_q_addr;
    uint32_t gpu_cpu_q_rkey; 
    uint64_t gpu_cpu_q_mbx_addr;
    uint32_t gpu_cpu_q_mbx_rkey; 

    int q_size;
};


class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> server;

    /* TODO: add memory region(s) for CPU-GPU queues */
    struct ibv_mr *cpu_gpu_q_mr;
    struct ibv_mr *cpu_gpu_q_mbx_mr;
    struct ibv_mr *gpu_cpu_q_mr;
    struct ibv_mr *gpu_cpu_q_mbx_mr;

    // Remote Information
    remote_information remote_info;
public:
    explicit server_queues_context(uint16_t tcp_port) : rdma_server_context(tcp_port), server(create_queues_server(1024))
    {
        /* TODO Initialize additional server MRs as needed. */
        cpu_gpu_q_mr = ibv_reg_mr(pd, server->cpu_gpu_q, sizeof(RingBuffer<RequestItem>), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        cpu_gpu_q_mbx_mr = ibv_reg_mr(pd, server->cpu_gpu_q->_mailbox, sizeof(RequestItem)*server->cpu_gpu_q->N, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        gpu_cpu_q_mr = ibv_reg_mr(pd, server->gpu_cpu_q, sizeof(RingBuffer<RequestItem>), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        gpu_cpu_q_mbx_mr = ibv_reg_mr(pd, server->gpu_cpu_q->_mailbox, sizeof(RequestItem)*server->gpu_cpu_q->N, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);


        if (!cpu_gpu_q_mr || !gpu_cpu_q_mr) 
        {
            fprintf(stderr, "Error, ibv_reg_mr() failed\n");
            exit(1);
        }

        this->remote_info.images_in_addr = (uint64_t) mr_images_in->addr;
        this->remote_info.images_in_rkey = mr_images_in->rkey;
        this->remote_info.images_out_addr = (uint64_t) mr_images_out->addr;
        this->remote_info.images_out_rkey = mr_images_out->rkey;

        this->remote_info.cpu_gpu_q_addr = (uint64_t) cpu_gpu_q_mr->addr;
        this->remote_info.cpu_gpu_q_rkey = cpu_gpu_q_mr->rkey;
        this->remote_info.cpu_gpu_q_mbx_addr = (uint64_t) cpu_gpu_q_mbx_mr->addr;
        this->remote_info.cpu_gpu_q_mbx_rkey = cpu_gpu_q_mbx_mr->rkey;
        
        this->remote_info.gpu_cpu_q_addr = (uint64_t) gpu_cpu_q_mr->addr;
        this->remote_info.gpu_cpu_q_rkey = gpu_cpu_q_mr->rkey;
        this->remote_info.gpu_cpu_q_mbx_addr = (uint64_t) gpu_cpu_q_mbx_mr->addr;
        this->remote_info.gpu_cpu_q_mbx_rkey = gpu_cpu_q_mbx_mr->rkey;

        this->remote_info.q_size = server->cpu_gpu_q->N;

        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
        send_over_socket(&remote_info, sizeof(remote_info));
    }

    ~server_queues_context()
    {
        /* TODO destroy the additional server MRs here */
        ibv_dereg_mr(cpu_gpu_q_mr);
        ibv_dereg_mr(gpu_cpu_q_mr);
    }

    virtual void event_loop() override
    {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
        rpc_request* req;

        bool terminate = false;

        while (!terminate) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		        VERBS_WC_CHECK(wc);

                if (wc.opcode == IBV_WC_RECV){
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];

                    /* Terminate signal */
                    if (req->request_id == TERMINATE) {
                        printf("Terminating...\n");
                        terminate = true;

                        post_rdma_write(
                                req->output_addr,                       // remote_dst
                                0,                                      // len
                                req->output_rkey,                       // rkey
                                0,                                      // local_src
                                mr_images_out->lkey,                    // lkey
                                (uint64_t)TERMINATE,                    // wr_id
                                (uint32_t *)&req->request_id);          // immediate
                    }
                }
            }
        }
    }
};

struct local_buffers{
    cuda::atomic<size_t> cpu_gpu_head;
    cuda::atomic<size_t> cpu_gpu_tail;

    cuda::atomic<size_t> gpu_cpu_head;
    cuda::atomic<size_t> gpu_cpu_tail;
    
    RequestItem request;
};

class client_queues_context : public rdma_client_context {
private:
    /* TODO add necessary context to track the client side of the GPU's
     * producer/consumer queues */
    struct local_buffers local_buffer;
    struct ibv_mr *local_buffer_mr;

    // Remote Information
    remote_information remote_info;

    struct ibv_mr *mr_images_in; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
    /* TODO define other memory regions used by the client here */

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        recv_over_socket(&remote_info, sizeof(remote_info));

        local_buffer_mr = ibv_reg_mr(pd, &local_buffer, sizeof(local_buffers), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
        
        if (!local_buffer_mr) 
        {
            fprintf(stderr, "Error, ibv_reg_mr() failed\n");
            exit(1);
        }

    }

    ~client_queues_context()
    {
	/* TODO terminate the server and release memory regions and other resources */
        send_termination();
        while (!get_termination()){};

        ibv_dereg_mr(local_buffer_mr);
    }

    virtual void set_input_images(uchar *images_in, size_t bytes) override
    {
        // TODO register memory

        /* register a memory region for the input images. */
        mr_images_in = ibv_reg_mr(pd, images_in, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_in) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        // TODO register memory
        
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */

        struct ibv_wc wc; 
        int ncqes;
        
        int image_adjust = img_id % OUTSTANDING_REQUESTS * IMG_SZ;
        uint64_t head_adjust = offsetof(RingBuffer<RequestItem>, _head);
        uint64_t tail_adjust = offsetof(RingBuffer<RequestItem>, _tail);

        /* Enqueu flow:
        *  1. Check if cpu->gpu queue is empty with rdma read request on remote queue
        *  2. If empty return false.
        *  3. Else, copy image to remote
        *  4. Update local cpu->gpu by pushing request
        *  5. Write updated queue to remote
        */

        // Read remote queue and check if full
        post_rdma_read(&local_buffer.cpu_gpu_head,                           // local_dst
                       sizeof(cuda::atomic<size_t>),     // len
                       local_buffer_mr->lkey,                  // lkey
                       remote_info.cpu_gpu_q_addr + head_adjust,          // remote_src
                       remote_info.cpu_gpu_q_rkey,          // rkey
                       0);                                  // wr_id

        while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        post_rdma_read(&local_buffer.cpu_gpu_tail,                           // local_dst
                       sizeof(cuda::atomic<size_t>),     // len
                       local_buffer_mr->lkey,                  // lkey
                       remote_info.cpu_gpu_q_addr + tail_adjust,          // remote_src
                       remote_info.cpu_gpu_q_rkey,          // rkey
                       0);                                  // wr_id

        while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        if (local_buffer.cpu_gpu_tail.load() - local_buffer.cpu_gpu_head.load() == (size_t)remote_info.q_size) {
            return false;
        }

        // Write image to remote
        post_rdma_write(remote_info.images_in_addr + image_adjust, // remote_dst
                    IMG_SZ,                                        // len
                    remote_info.images_in_rkey,                    // rkey
                    img_in,                                        // local_src
                    mr_images_in->lkey,                            // lkey
                    img_id);                                       // wr.id

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        // Update cpu-gpu queue and write to remote
        local_buffer.request = RequestItem(img_id,
                                    (uchar*)(remote_info.images_in_addr + image_adjust),
                                    (uchar*)(remote_info.images_out_addr + image_adjust));

        post_rdma_write((uint64_t)(remote_info.cpu_gpu_q_mbx_addr + sizeof(RequestItem)*(local_buffer.cpu_gpu_tail % remote_info.q_size)),     // remote_dst
                    sizeof(RequestItem),    // len
                    remote_info.cpu_gpu_q_mbx_rkey,         // rkey
                    &local_buffer.request,                          // local_src
                    local_buffer_mr->lkey,                 // lkey
                    img_id);
        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        local_buffer.cpu_gpu_tail++;
        post_rdma_write(remote_info.cpu_gpu_q_addr + tail_adjust,     // remote_dst
                    sizeof(cuda::atomic<size_t>),    // len
                    remote_info.cpu_gpu_q_rkey,         // rkey
                    &local_buffer.cpu_gpu_tail,                          // local_src
                    local_buffer_mr->lkey,                 // lkey
                    img_id);
        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */

        struct ibv_wc wc; 
        int ncqes;
        
        uint64_t head_adjust = offsetof(RingBuffer<RequestItem>, _head);
        uint64_t tail_adjust = offsetof(RingBuffer<RequestItem>, _tail);

        /* Enqueu flow:
        *  1. Check if cpu->gpu queue is empty with rdma read request on remote queue
        *  2. If empty return false.
        *  3. Update local cpu->gpu by popping
        *  4. Write updated queue to remote
        *  5. Read the image from remote
        */

        // Read remote queue and check if full
        post_rdma_read(&local_buffer.gpu_cpu_head,                           // local_dst
                       sizeof(cuda::atomic<size_t>),     // len
                       local_buffer_mr->lkey,                  // lkey
                       remote_info.gpu_cpu_q_addr + head_adjust,          // remote_src
                       remote_info.gpu_cpu_q_rkey,          // rkey
                       0);                                  // wr_id

        while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        post_rdma_read(&local_buffer.gpu_cpu_tail,                           // local_dst
                       sizeof(cuda::atomic<size_t>),     // len
                       local_buffer_mr->lkey,                  // lkey
                       remote_info.gpu_cpu_q_addr + tail_adjust,          // remote_src
                       remote_info.gpu_cpu_q_rkey,          // rkey
                       0);                                  // wr_id

        while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        if (local_buffer.gpu_cpu_tail.load() - local_buffer.gpu_cpu_head.load() == 0) {
            return false;
        }


        // Read the request
        post_rdma_read(&local_buffer.request,     // local_dst
                       sizeof(RequestItem),     // len
                       local_buffer_mr->lkey,                  // lkey
                       remote_info.gpu_cpu_q_mbx_addr + sizeof(RequestItem)*(local_buffer.gpu_cpu_head % remote_info.q_size),          // remote_src
                       remote_info.gpu_cpu_q_mbx_rkey,          // rkey
                       0);                                  // wr_id

        while ((ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        *img_id = local_buffer.request.image_id;

        // Read image from remote
        post_rdma_read((char*)mr_images_out->addr + *img_id % OUTSTANDING_REQUESTS * IMG_SZ,      // local_dst
                       IMG_SZ,                                                  // len
                       mr_images_out->lkey,                                     // lkey
                       remote_info.images_out_addr + *img_id % OUTSTANDING_REQUESTS * IMG_SZ,     // remote_src
                       remote_info.images_out_rkey,                             // rkey
                       *img_id);                                           // wr_id

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        local_buffer.gpu_cpu_head++;
        post_rdma_write(remote_info.gpu_cpu_q_addr + head_adjust,     // remote_dst
                    sizeof(cuda::atomic<size_t>),    // len
                    remote_info.gpu_cpu_q_rkey,         // rkey
                    &local_buffer.gpu_cpu_tail,                          // local_src
                    local_buffer_mr->lkey,                 // lkey
                    *img_id);
        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) {}
        if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
        }
        VERBS_WC_CHECK(wc);

        return true;
    }


    

    void send_termination(){
        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[0];
        req->request_id = TERMINATE;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = 0; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }
    }

    bool get_termination(){
        rpc_request* req;
        bool terminated = false;

        struct ibv_wc wc;
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes > 0) {
            VERBS_WC_CHECK(wc);

            if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM){
                /* Received a new request from the client */
                req = &requests[wc.wr_id];

                /* Terminate signal */
                if (req->request_id == TERMINATE) {
                    printf("Terminating...\n");
                    terminated = true;
                }
            }
        }

        return terminated;
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
    
}

/****************** RDMA Implementation - End ******************/
