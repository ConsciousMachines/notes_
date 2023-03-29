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



__device__ inline float DE(float3 z, Params params)
{
    float3 MandelboxOffset = make_float3(params.p[0], params.p[1], params.p[2]);
    float3 MengerOffset = make_float3(params.p[3], params.p[4], params.p[5]);
    float MengerScale = params.p[6];
    float box_mult = params.p[7];
    float FixedR2 = params.p[8];
    float MinR2 = params.p[9];
    float fold = params.p[10];
    float MandelboxScale = params.p[11];
    float Menger_Scale_Offset = params.p[12];
    float Menger_Z_thing = params.p[13];
    float dr_offset_Mandel = params.p[14];
    float XZ_plane_pos = params.p[15];
    float XY_plane_pos = params.p[16];

    // NEW : XY - plane for cutting crap out 
    float dist_to_XZ_plane = XZ_plane_pos - z.y;
    float dist_to_XY_plane = z.z - XY_plane_pos;


    MandelboxOffset = z + MandelboxOffset;
    const int Iterations = 5;
    float dr = 1.0f;
    float r2, temp; // local

    for (int n = 0; n < Iterations; n++)
    {
        for (int soy = 0; soy < 4; soy++)
        {
            // mandelbox step 
            z = clamp(z, -fold, fold) * box_mult - z;
            r2 = dot(z, z);
            if (r2 < MinR2) {
                temp = (FixedR2 / MinR2);
                z *= temp;
                dr *= temp;
            }
            else if (r2 < FixedR2) {
                temp = (FixedR2 / r2);
                z *= temp;
                dr *= temp;
            }
            z = MandelboxScale * z + MandelboxOffset;
            dr = dr * abs(MandelboxScale) + dr_offset_Mandel;
        }

        // menger step
        z = abs(z);
        if (z.x < z.y) {
            temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - Menger_Scale_Offset); // same space transform as tetrahedron
        if (z.z < -Menger_Z_thing * MengerOffset.z * (MengerScale - Menger_Scale_Offset))
        {
            z.z += MengerOffset.z * (MengerScale - Menger_Scale_Offset);
        }
        dr = dr * abs(MengerScale) + 1.f;// dr_offset_Menger;

        // menger step
        z = abs(z);
        if (z.x < z.y) {
            temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - Menger_Scale_Offset); // same space transform as tetrahedron
        if (z.z < -Menger_Z_thing * MengerOffset.z * (MengerScale - Menger_Scale_Offset))
        {
            z.z += MengerOffset.z * (MengerScale - Menger_Scale_Offset);
        }
        dr = dr * abs(MengerScale) + 1.f;// dr_offset_Menger;

        // menger step
        z = abs(z);
        if (z.x < z.y) {
            temp = z.x;
            z.x = z.y;
            z.y = temp;
        }
        if (z.x < z.z) {
            temp = z.x;
            z.x = z.z;
            z.z = temp;
        }
        if (z.y < z.z) {
            temp = z.y;
            z.y = z.z;
            z.z = temp;
        }
        z = MengerScale * z - MengerOffset * (MengerScale - Menger_Scale_Offset); // same space transform as tetrahedron
        if (z.z < -Menger_Z_thing * MengerOffset.z * (MengerScale - Menger_Scale_Offset))
        {
            z.z += MengerOffset.z * (MengerScale - Menger_Scale_Offset);
        }
        dr = dr * abs(MengerScale) + 1.f;// dr_offset_Menger;

        // i removed an additional menger step bc the bulb is already here 
    }
    float fractal = length(z) / abs(dr);// +dr_offset_final);
    float result = max(fractal, -dist_to_XY_plane);
    result = max(result, -dist_to_XZ_plane);
    return result;
}



void reset_options(Camera& camera)
{
    // camera.position = make_float3(0.f, 0.f, 6.f); 
    // camera.lookat = make_float3(0.0f);

    camera.position = make_float3(-3.75598f, 1.7928e-08f, -4.69161f); 
    camera.lookat = make_float3(-3.51827f, 1.7928e-08f, -3.72027);

    camera.params.EPS = 0.001f;
    camera.params.step_size = 1.f;
    camera.params.min_distance = 0.001f;

    camera.params.p[0] = 0.f; // MandelboxOffset.x
    camera.params.p[1] = 0.f; // MandelboxOffset.y
    camera.params.p[2] = 0.f; // MandelboxOffset.z
    camera.params.p[3] = 1.f; // MengerOffset.x
    camera.params.p[4] = 1.f; // MengerOffset.y
    camera.params.p[5] = 1.f; // MengerOffset.z

    camera.params.p[6] = 3.475f; // MengerScale
    camera.params.p[7] = 2.f; // box_mult
    camera.params.p[8] = 1.f; // FixedR2
    camera.params.p[9] = 0.f; // MinR2
    camera.params.p[10] = 1.65f; // fold
    camera.params.p[11] = 1.2f; // MandelboxScale
    camera.params.p[12] = 1.f; // Menger_Scale_Offset
    camera.params.p[13] = 0.5f; // Menger_Z_thing
    camera.params.p[14] = 1.f; // dr_offset_Mandel
    camera.params.p[15] = 10.f; // dr_offset_Menger
    camera.params.p[16] = -4.f; // dr_offset_final
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
    interaction |= ImGui::SliderFloat("Mandelbox Offset X", &camera.params.p[0], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Mandelbox Offset Y", &camera.params.p[1], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Mandelbox Offset Z", &camera.params.p[2], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger Offset X", &camera.params.p[3], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger Offset Y", &camera.params.p[4], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger Offset Z", &camera.params.p[5], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger Scale", &camera.params.p[6], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("Box Mult", &camera.params.p[7], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("Fixed R2", &camera.params.p[8], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("Min R2", &camera.params.p[9], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("Fold", &camera.params.p[10], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("Mandelbox Scale", &camera.params.p[11], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("Menger Scale Offset", &camera.params.p[12], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("Menger Z thing", &camera.params.p[13], -5.f, 5.f);
    interaction |= ImGui::SliderFloat("dr_offset_Mandel", &camera.params.p[14], 0.f, 5.f);
    interaction |= ImGui::SliderFloat("XZ cut plane", &camera.params.p[15], -10.f, 10.f);
    interaction |= ImGui::SliderFloat("XY cut plane", &camera.params.p[16], -10.f, 10.f);
    ImGui::End();

    if (interaction) camera.launch_kernel();
}