#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "glad.h"
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>



int imageW = 800, imageH = 600; // Original image width and height
int frame = 0;
int numSMs = 0;   // number of multiprocessors

GLuint gl_PBO; // OpenGL PBO 
struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
uchar4 *d_dst = NULL;  // Destination image on the GPU side

extern "C" void thread_spawner(uchar4 *dst, const int imageW, const int imageH,
                    const int animationFrame, const int numSMs);




int main(void)
{
    if (!glfwInit()) return -1;
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow* window = glfwCreateWindow(imageW, imageH, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;
		

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);


    // 1. check for hardware 
    int dev = findCudaDevice(0, (const char **)0);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    numSMs = deviceProp.multiProcessorCount;

    glDisable(GL_DEPTH_TEST);

    // create PBO
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, imageW * imageH * 4, 0, GL_DYNAMIC_DRAW);
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));



    while (!glfwWindowShouldClose(window))
    {
        // launch kernel
        frame++;
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        size_t num_bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, &num_bytes, cuda_pbo_resource));
        thread_spawner(d_dst, imageW, imageH, frame, numSMs);
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
        glDrawPixels(imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        
        



        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        //glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        //glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
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
