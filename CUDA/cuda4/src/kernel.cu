#include "helper_cuda.h"
#include "helper_math.h"

#include "definitions.h"


// the DE must be in this translation unit so NVCC can optimize.
#include "DE_Hybrid1.cuh"
// #include "DE_Hybrid2.cuh"
// #include "DE_Mandelbox.cuh"
// #include "DE_Menger.cuh"


// COOL FRACTALS:
// Hybrid1 := loop 4 {4 mandelbox, 1 menger} - has bulbs
#define max_iter 120.f
#define bone make_float3(0.89f, 0.855f, 0.788f)
__device__ inline float mix(float v1, float v2, float a)
{
    return v1 * (1.f - a) + v2 * a;
}
__device__ inline float3 abs(float3 a)
{
    return make_float3(abs(a.x), abs(a.y), abs(a.z));
}
// https://developer.download.nvidia.com/cg/pow.html
__device__ inline float3 pow(float3 x, float3 y)
{
    float3 rv;
    rv.x = pow(x.x, y.x);//exp(x.x * log(y.x));
    rv.y = pow(x.y, y.y);//exp(x.y * log(y.y));
    rv.z = pow(x.z, y.z);//exp(x.z * log(y.z));
    return rv;
}
// https://www.reddit.com/r/opengl/comments/6nghtj/glsl_mix_implementation_incorrect/
__device__ inline float3 mix(float3 v1, float3 v2, float a)
{
    float3 result;
    result.x = v1.x * (1.f - a) + v2.x * a;
    result.y = v1.y * (1.f - a) + v2.y * a;
    result.z = v1.z * (1.f - a) + v2.z * a;
    return result;
}


__device__ float intersect(float3 ro, float3 rd, float& iter, Params params)
{
    float res;
    float t = 0.f;
    iter = max_iter;
    for (float i = 0.f; i < max_iter; i += 1.f) {
        float3 p = ro + rd * t;
        res = DE(p, params);
        if (res < params.min_distance * t || res > 20.f) {
            iter = i;
            break;
        }
        t += res * params.step_size;
    }
    if (res > 20.f) t = -1.f;
    return t;
}


__device__ float ambientOcclusion(float3 p, float3 n, Params params) {
    float stepSize = 0.012f;
    float t = stepSize;
    float oc = 0.0f;
    for (int i = 0; i < 12; i++) {
        float d = DE(p + n * t, params);
        oc += t - d;
        t += stepSize;
    }
    return clamp(oc, 0.0f, 1.0f);
}


__device__ float3 normal(float3 p, Params params)
{
    float3 e0 = make_float3(params.EPS, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, params.EPS, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, params.EPS);
    float3 n = make_float3(
        DE(p + e0, params) - DE(p - e0, params),
        DE(p + e1, params) - DE(p - e1, params),
        DE(p + e2, params) - DE(p - e2, params));
    return normalize(n);
}


__device__ float3 lighting(float3 p, float3 rd, float iter, Params params) 
{
    float3 n = normal(p, params);
    float fake = iter / max_iter;
    float fakeAmb = exp(-fake * fake * 9.0f);
    float amb = ambientOcclusion(p, n, params);
    float3 col = make_float3(mix(1.0f, 0.125f, pow(amb, 3.0f))) * make_float3(fakeAmb) * bone;
    return col;
}


__global__ void kernel(uchar4* map, float3 ro, float3 lookat, Params params)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // 0..WIDTH-1
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // 0..HEIGHT-1
    int idx = ix + iy * WIDTH;

    //   C O O R D I N A T E S 
    float qx = ((float)ix) / ((float)WIDTH); // 0..1
    float qy = ((float)iy) / ((float)(HEIGHT - HEIGHT_OFFSET)); // 0..1
    float uvx = ((qx * 2.0f) - 1.0f) * (((float)(WIDTH)) / ((float)(HEIGHT - HEIGHT_OFFSET))); // [-1..1]*aspect
    float uvy = ((qy * 2.0f) - 1.0f); // -1..1

    //   C A M E R A 
    float3 f = normalize(lookat - ro);
    float3 s = normalize(cross(f, make_float3(0.0f, 1.0f, 0.0f)));
    float3 u = normalize(cross(s, f));
    float3 rd = normalize(uvx * s + uvy * u + 2.8f * f);  // transform from view to world

    //   B A C K G R O U N D
    float3 bg = mix(bone * 0.5f, bone, smoothstep(-1.0f, 1.0f, uvy));
    float3 col = bg;
    float3 p = ro;

    //   R A Y M A R C H 
    float iter = 0.f;// # of steps 
    float t = intersect(ro, rd, iter, params);

    //   L I G H T I N G 
    if (t > -0.5) {
        p = ro + t * rd;
        col = lighting(p, rd, iter, params);
        col = mix(col, bg, 1.0 - exp(-0.001 * t * t));
    }
    col = pow(clamp(col, 0.0f, 1.0f), make_float3(0.65f)); // POST
    col = col * 0.6f + 0.4f * col * col * (3.0f- 2.0f * col);  // contrast
    col = mix(col, make_float3(dot(col, make_float3(0.33f))), -0.5f);  // satuation
    col *= 0.5f + 0.5f * pow(19.0f * qx * qy * (1.0f - qx) * (1.0f - qy), 0.7f);  // vigneting

    col = clamp(col, 0.f, 1.f); // there is a bit of overflow somewhere but this fixes it.
    map[idx].x = (unsigned char)(255.0f * col.x);
    map[idx].y = (unsigned char)(255.0f * col.y);
    map[idx].z = (unsigned char)(255.0f * col.z);
    map[idx].w = (unsigned char)255; // need this otherwise photos are transparent
}


extern "C" void thread_spawner(uchar4 *dst, float3 ro, float3 lookat, Params params) {
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y); // # 256 threads per block
    dim3 grid((WIDTH / BLOCKDIM_X), ((HEIGHT - HEIGHT_OFFSET)/ BLOCKDIM_Y)); // 800 blocks in 1 grid

    // TODO: try the 6 SMs grid approach
    kernel<<<grid, threads>>>(dst, ro, lookat, params);
    getLastCudaError("kernel execution failed.\n");
}
