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




__device__ float DE(float3 z, Params params)
{
    // mandelbox params
    float Mandelbox_scale = params.p[0];
    float Mandelbox_min_radius2 = params.p[1];
    float Mandelbox_folding_limit = params.p[2];
    float Mandelbox_fixed_radius2 = params.p[3];
    // menger params
    float3 Menger_Offset;
    Menger_Offset.x = params.p[4];
    Menger_Offset.y = params.p[5];
    Menger_Offset.z = params.p[6];
    float Menger_Scale = params.p[7];

    float XZ_plane_pos = params.p[8];
    float XY_plane_pos = params.p[9];
    // NEW : XY - plane for cutting crap out 
    float dist_to_XZ_plane = XZ_plane_pos - z.y;
    float dist_to_XY_plane = z.z - XY_plane_pos;


    float3 Mandelbox_offset = z;
    float dr = 1.0f; // mandelbox scale accumulation to correct distance later?

	float temp, r2;

    for (int soy = 0; soy < 15; soy++)
    {
        // mandelbox steps
        for(int n = 0; n < 1; ++n) {
            // box fold 
            z = clamp(z, -Mandelbox_folding_limit, Mandelbox_folding_limit) * 2.0f - z;
            // sphere fold
            r2 = dot(z, z);
            if(r2 < Mandelbox_min_radius2) {
                temp = (Mandelbox_fixed_radius2 / Mandelbox_min_radius2);
                z *= temp;
                dr *= temp;
            }else if(r2 < Mandelbox_fixed_radius2) {
                temp = (Mandelbox_fixed_radius2 / r2);
                z *= temp;
                dr *= temp;
            }

            z = Mandelbox_scale * z + Mandelbox_offset;
            dr *= abs(Mandelbox_scale);

            // menger step
            z = abs(z);
            if (z.x < z.y) {temp = z.x; z.x = z.y; z.y = temp;}
            if (z.x < z.z) {temp = z.x; z.x = z.z; z.z = temp;}
            if (z.y < z.z) {temp = z.y; z.y = z.z; z.z = temp;}
            z = Menger_Scale * z - Menger_Offset * (Menger_Scale-1.0f);
            if( z.z<-0.5f * Menger_Offset.z * (Menger_Scale-1.0f))  
                z.z += Menger_Offset.z * (Menger_Scale-1.0f);
            dr *= abs(Menger_Scale);
        }
        
    }
    return max(max(length(z) / abs(dr) -0.001f, -dist_to_XY_plane), -dist_to_XZ_plane);;
}


void reset_options(Camera& camera)
{
    camera.position = make_float3(0.f, 0.f, 6.f); 
    camera.lookat = make_float3(0.0f);

    camera.params.EPS = 0.001f;
    camera.params.step_size = 1.f;
    camera.params.min_distance = 0.001f;

    camera.params.p[0]  = 2.5f;    // Mandelbox_scale                
    camera.params.p[1]  = 0.f;     // Mandelbox_min_radius2           
    camera.params.p[2]  = 0.98f;   // Mandelbox_folding_limit              
    camera.params.p[3]  = 1.0f;    // Mandelbox_fixed_radius2         
    camera.params.p[4]  = 2.225f;  // Menger_Offset x           
    camera.params.p[5]  = 2.8f;    // Menger_Offset y
    camera.params.p[6]  = 1.025f;  // Menger_Offset z
    camera.params.p[7]  = 2.175f;  // Menger_Scale         
    camera.params.p[8]  = 10.f;          
    camera.params.p[9]  = -10.f;        
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
    interaction |= ImGui::SliderFloat("Mandelbox_scale",  &camera.params.p[0],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Mandelbox_min_radius2",  &camera.params.p[1],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Mandelbox_folding_limit",  &camera.params.p[2],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Mandelbox_fixed_radius2",  &camera.params.p[3],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger_Offset x",  &camera.params.p[4],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger_Offset y",  &camera.params.p[5],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger_Offset z",  &camera.params.p[6],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger_Scale",  &camera.params.p[7],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("dist_to_XY_plane",  &camera.params.p[8],  -10.f, 10.f);
    interaction |= ImGui::SliderFloat("dist_to_XZ_plane",  &camera.params.p[9],  -10.f, 10.f);
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