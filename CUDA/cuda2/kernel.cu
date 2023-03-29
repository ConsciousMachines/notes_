#include "helper_cuda.h"

#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define ABS(n) ((n) < 0 ? -(n) : (n))

// Increase the grid size by 1 if the image width or height does not divide
// evenly by the thread block dimensions
inline int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void kernel(uchar4* dst, const int imageW, const int imageH,
                            const int animationFrame, const int gridWidth,
                            const int numBlocks, int param) {
  // loop until all blocks completed
  for (unsigned int blockIndex = blockIdx.x; blockIndex < numBlocks;
       blockIndex += gridDim.x) {
    unsigned int blockX = blockIndex % gridWidth;
    unsigned int blockY = blockIndex / gridWidth;

    // process this block
    const int ix = blockDim.x * blockX + threadIdx.x;
    const int iy = blockDim.y * blockY + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
      // int m = blockIdx.x;         // uncomment to see scheduling order
      dst[imageW * iy + ix].x = ix;
      dst[imageW * iy + ix].y = iy + param;
      dst[imageW * iy + ix].z = animationFrame;
    }
  }
}

extern "C" void thread_spawner(uchar4 *dst, const int imageW, const int imageH,
                    const int animationFrame, const int numSMs, int param) {
  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

  // USUALLY i have some random number on LHS but it seems its been grouped into a 
  // "big task" to keep one SM busy (in a loop) without re-scheduling stuff.
  kernel<<<numSMs, threads>>>(dst, imageW, imageH, animationFrame, grid.x, grid.x * grid.y, param);

  getLastCudaError("kernel execution failed.\n");
}