#include "helper_cuda.h"
#include "helper_math.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "camera.h"
#include "definitions.h"

__device__ inline float mix(float v1, float v2, float a);
__device__ inline float3 abs(float3 a);
__device__ inline float3 pow(float3 x, float3 y);
__device__ inline float3 mix(float3 v1, float3 v2, float a);

__device__ float3 max(float3 a, float b)
{
    float c1 = max(a.x, b);
    float c2 = max(a.y, b);
    float c3 = max(a.z, b);
    return make_float3(c1,c2,c3);
}

__device__ float mod(float x, float y) // https://stackoverflow.com/questions/7610631/glsl-mod-vs-hlsl-fmod
{
    return x - y * floor(x/y);
}


__device__ float3 mod(float3 a, float b)
{
    float c1 = mod(a.x, b);
    float c2 = mod(a.y, b);
    float c3 = mod(a.z, b);
    return make_float3(c1,c2,c3);
}



// Inigo Quilez menger made from sdCrosses https://iquilezles.org/articles/menger/
__device__ float maxcomp(float3 p ) { return max(p.x,max(p.y,p.z));}
__device__ float sdBox(float3 p, float3 b)
{
    float3  di = abs(p) - b;
    float mc = maxcomp(di);
    return min(mc,length(max(di, 0.0f)));
}
__device__ float DE_(float3 p, Params params)
{
    float d = sdBox(p, make_float3(1.0f));

    float s = 1.0f;
    for( int m=0; m<1; m++)
    {
        float3 a = mod( p*s, 2.0f )-1.0f;
        s *= 3.0f;
        float3 r = abs(1.0f - 3.0f*abs(a));
        float da = max(r.x,r.y);
        float db = max(r.y,r.z);
        float dc = max(r.z,r.x);
        float c = (min(da,min(db,dc))-1.0f)/s;

        if( c>d )
        {
          d = c;
        }
    }
    return d;
}


// Syntopia menger - https://www.shadertoy.com/view/Mdf3z7
__device__ float DE__(float3 z, Params params)
{
	// Folding 'tiling' of 3D space;
	//z  = abs(1.0f - mod(z, 2.0f));

    //make_float3(1.0f);//make_float3(0.92858f, 0.92858f, 0.32858f);
    float3 Offset = make_float3(2.225f, 2.8f, 1.025f);
    float Scale = 2.175f;
	float d = 1000.0f;
    float temp;
	for (int n = 0; n < 15; n++) {
		z = abs(z);
		if (z.x<z.y)
        {
            temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
		if (z.x< z.z)
        {
            temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
		if (z.y<z.z)
        {
            temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
		z = Scale * z - Offset * (Scale - 1.0f);
		if( z.z < -0.5f * Offset.z * (Scale - 1.0f))  
        {
            z.z += Offset.z * (Scale - 1.0f);
        }
		d = min(d, length(z) * pow(Scale, float(-n) - 1.0f));
	}
	return d - 0.001f;
}


void reset_options(Camera& camera)
{
    camera.position = make_float3(0.f, 0.f, -6.f); 
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