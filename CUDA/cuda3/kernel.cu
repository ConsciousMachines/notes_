#include "helper_cuda.h"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16


__global__ void kernel2(uchar4* dst, const int imageW, const int imageH,
                        int param) {

  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

  if ((ix < imageW) && (iy < imageH)) {
    unsigned int idx = imageW * iy + ix;
    dst[idx].x = blockIdx.x * 16;
    dst[idx].y = blockIdx.y * 16;
    dst[idx].z = warpSize + param;
  }
}

__global__ void kernel(uchar4* dst, const int imageW, const int imageH,
                            const int frame, const int gridWidth,
                            const int gridSize, int param) {
  // TODO: compare this against normal method (is this bad for cache?)
  // each kernel loops 133 times. 6 kernels looping simultaneously. 
  // and 256 such kernels need to be launched (same blockIdx each time = "same" block)
                              
  // we manually allocate 800 blocks over 6 SMs
  // blockIndex = blockIdx.x is 0..5 (start value, one into each SM)
  // gridSize is 40*20 = 800         (num blocks = grid size)
  // gridDim.x is 6                  (step size, next block for that SM)
  for (unsigned int blockIndex = blockIdx.x; blockIndex < gridSize - 10;
       blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth; // 0..39 
    unsigned int blockY = blockIndex / gridWidth; // 0..19

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      int m = blockIdx.x;         // uncomment to see scheduling order
      dst[imageW * iy + ix].x = m * 42;//ix;
      dst[imageW * iy + ix].y = m * 42;//iy + param;
      dst[imageW * iy + ix].z = m * 42;//frame;
    }
  }
}

extern "C" void thread_spawner(uchar4 *dst, const int imageW, const int imageH,
                    const int frame, const int numSMs, int param) {
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y); // # 256 threads per block
  dim3 grid(40, 20); // 800 blocks in 1 grid

  // grid size: 40 x 20 = 800 blocks
  // -> blockIdx 0..39, 0..19
  // block size: 16 x 16 = 256 threads
  // -> threadIdx 0..15, 0..15 

  // kernel<<< 800, 256 >>>();
  // kernel<<< (40,20), (16,16) >>>();
  //kernel<<<numSMs, threads>>>(dst, imageW, imageH, frame, grid.x, grid.x * grid.y, param);
  kernel2<<<grid, threads>>>(dst, imageW, imageH, param);

  getLastCudaError("kernel execution failed.\n");
}