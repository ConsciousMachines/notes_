#include "imgui.h" // version 1.89.2
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>


#include <glad/glad.h>
#include <GLFW/glfw3.h> 


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>


#define HEIGHT 40*16
#define WIDTH 40*16


extern "C" void thread_spawner(uchar4*, const int, const int, const int, const int, int);


int main(int, char**)
{
    // GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "C U D A", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync


    // GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;
    glDisable(GL_DEPTH_TEST);


    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");


    // CUDA: check for hardware 
    int dev = findCudaDevice(0, (const char **)0);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    int numSMs = deviceProp.multiProcessorCount;     // number of multiprocessors


    // create PBO
    GLuint gl_PBO;                                   // OpenGL PBO 
    struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
    uchar4 *d_dst = NULL;                            // Destination image on the GPU side
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));


    static float f1, f2;
    size_t num_bytes;
    int display_w, display_h, frame = 0;
    while (1)
    {
        // render image
        frame++;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
        thread_spawner(d_dst, WIDTH, HEIGHT, frame, numSMs, int(f1 * 255.0));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Hello, world!");
            ImGui::SliderFloat("float1", &f1, 0.0f, 1.0f);
            ImGui::SliderFloat("float2", &f2, 0.0f, 1.0f);
            ImGui::End();
        }


        // Rendering
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    // Cleanup
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &gl_PBO);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}