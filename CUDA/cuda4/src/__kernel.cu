// AmazingBox2 + Menger3
// i got Menger3 seemingly correct, now to try AmazingBox2, which cant be done alone.

#include "helper_cuda.h"
#include "helper_math.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "camera.h"
#include "definitions.h"


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






////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
// https://www.shadertoy.com/view/Mdf3z7


#define MaxSteps 30
#define MinimumDistance 0.0009
#define normalDistance 0.0002

#define Jitter 0.05f
#define FudgeFactor 0.7f


__device__ float getLight(float3 color, float3 normal, float3 dir) {
	float3 lightDir = make_float3(1.0f);
	float diffuse = max(0.0f,dot(-normal, lightDir)); // Lambertian
	return diffuse*0.5f;
}


// DE: Infinitely tiled Menger IFS.
// For more info on KIFS, see:
// http://www.fractalforums.com/3d-fractal-generation/kaleidoscopic-%28escape-time-ifs%29/
__device__ float DE_(float3 z, Params params)
{
    // mandelbox
    float Mandelbox_fixed_radius2 = params.p[0];
    float Mandelbox_min_radius2 = params.p[1];
    float Mandelbox_scale = params.p[2];
    float Mandelbox_folding_limit = params.p[3];
    // menger
    float3 Menger_Offset = make_float3(2.225f, 2.8f, 1.025f);
    float Menger_Scale = 2.175f;
    

    float3 Mandelbox_offset = z;
    float dr = 1.0f;

    float d = 1000.0f; // for menger
    float temp; // used for temp storage
    for (int soy = 1; soy < 15; soy++)
    {
        /*
        for(int n = 0; n < 2; ++n) {
            // box fold 
            z = clamp(z, -Mandelbox_folding_limit, Mandelbox_folding_limit) * 2.0f - z;

            // sphere fold
            float r2 = dot(z, z);
            if(r2 < Mandelbox_min_radius2) {
                float temp = (Mandelbox_fixed_radius2 / Mandelbox_min_radius2);
                z *= temp;
                dr *= temp;
            }else if(r2 < Mandelbox_fixed_radius2) {
                float temp = (Mandelbox_fixed_radius2 / r2);
                z *= temp;
                dr *= temp;
            }

            z = Mandelbox_scale * z + Mandelbox_offset;
            dr = dr * abs(Mandelbox_scale) + 1.0f;
        }
        float Mandel_result = length(z) / abs(dr);
        */

        // Folding 'tiling' of 3D space;
        //z  = abs(1.0-mod(z,2.0));
        
        for (int n = 0; n < 1; n++) {	
            z = abs(z);
            if (z.x < z.y) {temp = z.x; z.x = z.y; z.y = temp;}
            if (z.x < z.z) {temp = z.x; z.x = z.z; z.z = temp;}
            if (z.y < z.z) {temp = z.y; z.y = z.z; z.z = temp;}
            z = Menger_Scale * z - Menger_Offset * (Menger_Scale-1.0f);
            if( z.z < - 0.5f * Menger_Offset.z * (Menger_Scale-1.0f))  
                z.z += Menger_Offset.z * (Menger_Scale-1.0f);
            d = min(d, length(z) * pow(Menger_Scale, float(-n)-1.0f)); // AO?

        }
    }
    
	return d-0.001f;
}


__device__ float DE(float3 z, Params params)
{
    // mandelbox params
    const float Mandelbox_scale = 1.0f;//params.p[2];
    const float Mandelbox_min_radius2 = 0.0f;//params.p[1];
    const float Mandelbox_folding_limit = 0.98f;//params.p[3];
    // menger params
    float3 Menger_Offset = make_float3(2.225f, 2.8f, 1.025f);
    float Menger_Scale = 2.175f;

    float Mandelbox_fixed_radius2 = params.p[0];

    float3 Mandelbox_offset = z;
    float dr = 1.0f; // mandelbox scale accumulation to correct distance later?

    //float d = 1000.0f;
	float temp;


    for (int soy = 0; soy < 5; soy++)
    {
        for(int n = 0; n < 2; ++n) {
            // box fold 
            z = clamp(z, -Mandelbox_folding_limit, Mandelbox_folding_limit) * 2.0f - z;

            // sphere fold
            float r2 = dot(z, z);
            if(r2 < Mandelbox_min_radius2) {
                float temp = (Mandelbox_fixed_radius2 / Mandelbox_min_radius2);
                z *= temp;
                dr *= temp;
            }else if(r2 < Mandelbox_fixed_radius2) {
                float temp = (Mandelbox_fixed_radius2 / r2);
                z *= temp;
                dr *= temp;
            }

            z = Mandelbox_scale * z + Mandelbox_offset;
            dr = dr * abs(Mandelbox_scale) + 1.0f;
        }
        //return length(z) / abs(dr);
        
        for (int n = 0; n < 1; n++) {	
            z = abs(z);
            if (z.x < z.y) {temp = z.x; z.x = z.y; z.y = temp;}
            if (z.x < z.z) {temp = z.x; z.x = z.z; z.z = temp;}
            if (z.y < z.z) {temp = z.y; z.y = z.z; z.z = temp;}
            z = Menger_Scale * z - Menger_Offset * (Menger_Scale-1.0f);
            if( z.z<-0.5f * Menger_Offset.z * (Menger_Scale-1.0f))  
                z.z += Menger_Offset.z * (Menger_Scale-1.0f);
            //d = min(d, length(z) * pow(Menger_Scale, float(-n)-1.0f));
            dr *= abs(Menger_Scale);
            
        }

    }


    
    float d = length(z) / (abs(dr) + Menger_Scale);
	return d-0.001f;
}


// Finite difference normal
__device__ float3 getNormal(float3 p, Params params)
{
    float3 e0 = make_float3(normalDistance, 0.0f, 0.0f);
    float3 e1 = make_float3(0.0f, normalDistance, 0.0f);
    float3 e2 = make_float3(0.0f, 0.0f, normalDistance);

    return normalize(make_float3(
        DE(p + e0, params) - DE(p - e0, params),
        DE(p + e1, params) - DE(p - e1, params),
        DE(p + e2, params) - DE(p - e2, params)));
}


__device__ float fract(float x)
{
	return x - floor(x);
}


// Pseudo-random number
// From: lumina.sourceforge.net/Tutorials/Noise.html
__device__ float rand(float2 co){
	float2 soy = make_float2(4.898f,7.23f);
	return fract(cos(dot(co,soy)) * 23421.631f);
}

__device__ float3 rayMarch(float3 from, float3 dir, float uvx, float uvy, int iTime, Params params) {
	// Add some noise to prevent banding
	float totalDistance = 0.f;//Jitter*rand(make_float2(uvx, uvy) + make_float2(iTime));
	//float3 dir2 = dir;
	float distance;
	int steps = 0;
	float3 pos;
	for (int i=0; i < MaxSteps; i++) {
		
		pos = from + totalDistance * dir;
		distance = DE(pos, params)*FudgeFactor;
		totalDistance += distance;
		if (distance < MinimumDistance) break;
		steps = i;
	}
	
	// 'AO' is based on number of steps.
	// Try to smooth the count, to combat banding.
	float smoothStep =   float(steps) + distance/MinimumDistance;
	float ao = 1.1f - smoothStep/float(MaxSteps);
	
	// Since our distance field is not signed,
	// backstep when calc'ing normal
	float3 normal = getNormal(pos-dir*normalDistance*3.0f, params);
	
	float3 color = make_float3(1.0f);
	float light = getLight(color, normal, dir);
	color = (color * 0.32184f + light)*ao;
	return color;
}




__global__ void kernel(uchar4* map, float3 ro, float3 lookat, int iTime, Params params)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // 0..WIDTH-1
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // 0..HEIGHT-1
    int idx = ix + iy * WIDTH;

    //   C O O R D I N A T E S 
    float qx = ((float)ix) / ((float)WIDTH); // 0..1
    float qy = ((float)iy) / ((float)(HEIGHT - HEIGHT_OFFSET)); // 0..1
    float uvx = ((qx * 2.0f) - 1.0f) * (((float)(WIDTH)) / ((float)(HEIGHT - HEIGHT_OFFSET))); // [-1..1]*aspect
    float uvy = ((qy * 2.0f) - 1.0f); // -1..1

	// Camera position (eye), and camera target
	/*
	float3 camPos = ro;//make_float3(0.0,0.0,3.0);
	float3 target = lookat;//make_float3(0.0,0.0,0.0);
	float3 camUp  = make_float3(0.0f,1.0f,0.0f);
	float3 camDir   = normalize(target-camPos); // direction for center ray
	camUp = normalize(camUp-dot(camDir,camUp)*camDir); // orthogonalize
	float3 camRight = normalize(cross(camDir,camUp));
	float3 rayDir = normalize(camDir + uvx*camRight + uvy*camUp);
	*/
	    //   C A M E R A 
    float3 f = normalize(lookat - ro);
    float3 s = normalize(cross(f, make_float3(0.0f, 1.0f, 0.0f)));
    float3 u = normalize(cross(s, f));
    float3 rayDir = normalize(uvx * s + uvy * u + 2.8f * f);  // transform from view to world
	float3 camPos = ro;

	float3 col = rayMarch(camPos, rayDir, uvx, uvy, iTime, params);

    //col = clamp(col, 0.f, 1.f); // there is a bit of overflow somewhere but this fixes it.
    map[idx].x = (unsigned char)(255.0f * col.x);
    map[idx].y = (unsigned char)(255.0f * col.y);
    map[idx].z = (unsigned char)(255.0f * col.z);
    //map[idx].w = (unsigned char)255; // need this otherwise photos are transparent
}


extern "C" void thread_spawner(uchar4 *dst, float3 ro, float3 lookat, Params params) {
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y); // # 256 threads per block
    dim3 grid((WIDTH / BLOCKDIM_X), ((HEIGHT - HEIGHT_OFFSET)/ BLOCKDIM_Y)); // 800 blocks in 1 grid

	static int iTime = 0;
    // TODO: try the 6 SMs grid approach
    //kernel<<<grid, threads>>>(dst, ro, lookat, params);
    kernel<<<grid, threads>>>(dst, ro, lookat, iTime++, params);
    getLastCudaError("kernel execution failed.\n");
}



void reset_options(Camera& camera)
{
    camera.position = make_float3(0.f, 0.f, 6.f); 
    camera.lookat = make_float3(0.0f);

    camera.params.EPS = 0.001f;
    camera.params.step_size = 1.f;
    camera.params.min_distance = 0.001f;
    
    camera.params.p[0]  = 0.f;                    
    camera.params.p[1]  = 0.f;                    
    camera.params.p[2]  = 0.f;                    
    camera.params.p[3]  = 0.f;                 
    camera.params.p[4]  = 0.f;                 
    camera.params.p[5]  = 0.f;                 
    camera.params.p[6]  = 0.f;                 
    camera.params.p[7]  = 0.f;           
    camera.params.p[8]  = 0.f;          
    camera.params.p[9]  = 0.f;        
    camera.params.p[10] = 0.f;          
    camera.params.p[11] = 0.f;    
    camera.params.p[12] = 0.f;        
    camera.params.p[13] = 0.f;
    camera.params.p[14] = 0.f; 
    camera.params.p[15] = 0.f;  
    camera.params.p[16] = 0.f; 
}

void render_options(Camera& camera) 
{
    int interaction = 0; // becomes 1 if any slider was used, so we know to render

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    //ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    ImGui::Begin("V/B inc/dec min_dist, N/M inc/dec EPS, K/L inc/dec step_size");
    if (ImGui::Button("Reset"))
    {
        reset_options(camera);
        camera.launch_kernel();
    }
    interaction |= ImGui::SliderFloat("p0",  &camera.params.p[0],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p1",  &camera.params.p[1],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p2",  &camera.params.p[2],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p3",  &camera.params.p[3],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p4",  &camera.params.p[4],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p5",  &camera.params.p[5],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p6",  &camera.params.p[6],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p7",  &camera.params.p[7],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p8",  &camera.params.p[8],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p9",  &camera.params.p[9],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p10", &camera.params.p[10], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p11", &camera.params.p[11], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p12", &camera.params.p[12], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p13", &camera.params.p[13], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p14", &camera.params.p[14], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p15", &camera.params.p[15], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("p16", &camera.params.p[16], -10.f, 10.f);
    ImGui::End();

    if (interaction) camera.launch_kernel();
}