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


__device__ float DE(float3 z, Params params) {

    float fixed_radius2 = params.p[0];
    float min_radius2 = params.p[1];
    float scale = params.p[2];
    float folding_limit = params.p[3];

    float3 offset = z;
    float dr = 1.0f;
    for(int n = 0; n < 15; ++n) {
        // box fold 
        z = clamp(z, -folding_limit, folding_limit) * 2.0f - z;

        // sphere fold
        float r2 = dot(z, z);
        if(r2 < min_radius2) {
            float temp = (fixed_radius2 / min_radius2);
            z *= temp;
            dr *= temp;
        }else if(r2 < fixed_radius2) {
            float temp = (fixed_radius2 / r2);
            z *= temp;
            dr *= temp;
        }

        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0f;
    }
    float r = length(z);
    return r / abs(dr);
}


void reset_options(Camera& camera)
{
    camera.position = make_float3(0.f, 0.f, 6.f); 
    camera.lookat = make_float3(0.0f);

    camera.params.EPS = 0.001f;
    camera.params.step_size = 1.f;
    camera.params.min_distance = 0.001f;
    camera.params.p[0] = 1.9f; // fixed_radius2 = 1.9f;
    camera.params.p[1] = 0.1f; // min_radius2 = 0.1f;
    camera.params.p[2] =-2.8f; // scale = -2.8f;
    camera.params.p[3] = 1.0f; // folding_limit = 1.0f;
    camera.params.p[4] = 0.f;
    camera.params.p[5] = 0.f;
    camera.params.p[6] = 0.f;
    camera.params.p[7] = 0.f;
    camera.params.p[8] = 0.f;
    camera.params.p[9] = 0.f;
    camera.params.p[10]= 0.f;
    camera.params.p[11]= 0.f;
    camera.params.p[12]= 0.f;
    camera.params.p[13]= 0.f;
    camera.params.p[14]= 0.f;
    camera.params.p[15]= 0.f;
    camera.params.p[16]= 0.f;
}

void render_options(Camera& camera) 
{
    int interaction = 0; // becomes 1 if any slider was used, so we know to render

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowBgAlpha(0.f); // Transparent background
    ImGui::Begin("V/B inc/dec min_dist, N/M inc/dec EPS, K/L inc/dec step_size");
    if (ImGui::Button("Reset"))
    {
        reset_options(camera);
        camera.launch_kernel();
    }
    interaction |= ImGui::SliderFloat("fixed_radius2", &camera.params.p[0], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("min_radius2", &camera.params.p[1], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("scale", &camera.params.p[2], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("folding_limit", &camera.params.p[3], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p4", &camera.params.p[4], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p5", &camera.params.p[5], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p6", &camera.params.p[6], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p7", &camera.params.p[7], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p8", &camera.params.p[8], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p9", &camera.params.p[9], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p10", &camera.params.p[10], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p11", &camera.params.p[11], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p12", &camera.params.p[12], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p13", &camera.params.p[13], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p14", &camera.params.p[14], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p15", &camera.params.p[15], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("p16", &camera.params.p[16], -5.f, 5.f);
    ImGui::End();

    if (interaction) camera.launch_kernel();
}
