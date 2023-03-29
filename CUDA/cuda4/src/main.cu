#include "imgui.h" // version 1.89.2
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>


#include <glad/glad.h>
#include <GLFW/glfw3.h> 


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_math.h>


#include "definitions.h"
#include "camera.h"


/*
https://fractalforums.org/programming/11
https://www.fractalforums.com/3d-fractal-generation/a-mandelbox-distance-estimate-formula/
http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/
https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Historical%203D%20Fractals/Mandelbox.frag
https://en.wikipedia.org/wiki/Mandelbox
https://en.wikibooks.org/wiki/Mandelbulb3D/Reference/Formulas/Formula_types
http://blog.hvidtfeldts.net/index.php/2011/08/distance-estimated-3d-fractals-iii-folding-space/
https://fractal.batjorge.com/2022/02/02/matinal-menger/ <- recreate this (failed to match AmazingBox2)
https://www.deviantart.com/batjorge
*/

// TODO: 
// ADD BANDING FROM SYNTOPIA MENGER !!!
// https://community.vcvrack.com/t/vcv-rack-on-linux-detailed-guide-faq/15663
// - cinematic lighting?
// - imgui based file loader 

// - things that are amazing:
// 1. wine - how does it work?
// 2. primusrun - how does it work?
// 3. vcv rack
// 4. xrandr zoom/unzoom

/*

position
-0.643060810344573
-8.11384890984497
0.920192221093853
0.230905271378835
8.67005524524585
5.3294492370031
rotation
-59.714639988761
179.899686298201
-127.223223450444


-1.10995197408204E-17



Mandelbulb3Dv18{
g…..E/…o0…w….26….daLcnBtMnzY1gdgL2L30E6p7i1FzYYzPJk1sN8d16.DiBoHjBmpyD
…………………………..Qhb0YgJIJ.2……..Y./……….OaNaNaNap.2…wD
…Uzg/…kd…./Q.1/….2ki5…23…2Ej…..Mj3.DJPLhoD/Q..Aq3E10.G0dkpXm1.Xfe8
z2U0LDOD12../2………………….aNaNaNaN/.A………..U0…..y1…sD…../..
.z1…kDh.Kq7OmnswHpxQhVZE3Ez4wsLfYlV7fDaenX6wcPkwvTRisyZwF8zi3o99/6p4ojc9Y6hyCC
ww1Jim45.IKBzonbfn0miGnjp…..I39…v….2….sD.6….sD..G…………………
………….oAnAt1…sD….z6CARpjZTiA………………………..Aq3EX….k1.
….H83iyz1……….wzzz1.U..6.8/…M2…EB….m….E2….F….6/…In4….SFz7
…U.0Ak6zDk.Xwz.1Amz1Ak616.22pzvLij..E92FpyVYyD6YXppZd1..2……….2.28.kFrA0.
.Ub96aAIVz9.1se7Umvxz0………./EU0.wzzz1………..s/……………….E.2c..
zzzz……………………………2./8.kzzzD…………8………………..
/EU0.wzzz1……………………………..kAnA1UtbTCEzz/nAnA.zzzz.tzDAnAn.sef
i0IzTlAnA1kzzzDEQs5nAnA.zzzz./bTAnAn.wzzz12lylAnA1knE1BEMw5nAnA.V5Ss.lqTAnAn.wzz
z12kzlAnA1kefi8E…cU08czz/cU08cyz1cU08cxz3cU08c…………………………..
E….6E2F2.2….I….k….EEh3aSdtqN0x4Sm.UQ……………………..k/.MU/4…
.ok.1…………..wz………..QxckpX0Ljz1……………………………….
…………………..adlvAnAnAnAnuz…………………………………….
…………………2…..3….8….oINiRKNmB1.jtKG4B3…………………….
…..MU/4MU/4……..IaNaNaNa3.EBnAnAnAn/.oNaNaNaNa/.NaNaNaNa/zD…………….
……………………………………………………………………..
…………………………..}
{Titel: cafeconmenger}

*/


// the GUI options are in the respectful fractal's file
extern void reset_options(Camera& camera);
extern void render_options(Camera& camera);


// paint the memory with 255 in 4th int so it's not transparent
__global__ void pre_kernel(uchar4* map)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x; // 0..WIDTH-1
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // 0..HEIGHT-1
    int idx = ix + iy * WIDTH;
    map[idx].w = (unsigned char)255; // need this otherwise photos are transparent
}


int main(int, char**)
{
    // camera
    Camera camera;
    camera.render_options = render_options;
    camera.reset_options = reset_options;


    // GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "C U D A", NULL, NULL);
    if (window == NULL) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    // set keyboard callback to camera's function using a closure
    glfwSetWindowUserPointer(window, &camera);
    auto func = [](GLFWwindow* w, int key, int scancode, int action, int mods)
    {
        static_cast<Camera*>(glfwGetWindowUserPointer(w))->keypress_callback(key, action);
    };
    glfwSetKeyCallback(window, func);


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
    io.ConfigWindowsMoveFromTitleBarOnly = true; // make window not move when i accidentally click it


    // CUDA: check for hardware 
    int dev = findCudaDevice(0, (const char **)0);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    //int numSMs = deviceProp.multiProcessorCount;     // number of multiprocessors


    // create PBO
    GLuint gl_PBO;                                   // OpenGL PBO 
    struct cudaGraphicsResource *cuda_pbo_resource;  // handles OpenGL-CUDA exchange
    uchar4 *d_dst = NULL;                            // Destination image on the GPU side
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, 0, GL_DYNAMIC_DRAW);
    //checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO, cudaGraphicsRegisterFlagsWriteDiscard));
    // TODO: apparently this flag is more efficient?


    // render first frame
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, 0, cuda_pbo_resource));
    camera.dst = d_dst;
    camera.cuda_pbo_resource = cuda_pbo_resource;

    // prepare the drawing memory
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y); // # 256 threads per block
    dim3 grid((WIDTH / BLOCKDIM_X), ((HEIGHT - HEIGHT_OFFSET)/ BLOCKDIM_Y)); // 800 blocks in 1 grid
    pre_kernel<<<grid, threads>>>(d_dst); 

    camera.launch_kernel();
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    

    camera.reset_options(camera); // initialize fractal params
    camera.launch_kernel(); // render first frame


    int display_w, display_h;
    while (camera.should_render)
    {
        // render image
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_dst, 0, cuda_pbo_resource));
        camera.move();
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
        glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


        // ImGui
        camera.render_options(camera);
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


/*

#define MAX_STEPS 100
#define MAX_DIST 20.
#define SURF_DIST 0.001

// NOTES
// 1. ray direction must be normalized.
// 2. move objects by moving p in opposite direction.
// 3. step size adjusted by max scale in any dimension
// 4. rotate p.xz
// 5. menger fractal requires interior distance of box


float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdBox( vec2 p, vec2 b )
{
  vec2 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,q.y),0.0);
}

float sdCross( in vec3 p )
{
  // float da = sdBox(p.xy,vec2(1));
  // float db = sdBox(p.yz,vec2(1));
  // float dc = sdBox(p.zx,vec2(1));
  p = abs(p) - vec3(1.0) / 3.0;
  float da = max(p.x, p.y);
  float db = max(p.y, p.z);
  float dc = max(p.x, p.z);
  return min(da,min(db,dc));
}

mat2 r(float a)
{
    float s = sin(a);
    float c = cos(a);
    return mat2(c,-s,s,c);
}

float GetDist(vec3 p)
{
  float d = sdBox(p,vec3(1.0));
  float dc = sdCross(p);
  d = min(d, dc);

      
    /*

   vec3 a = mod( p, 2.0 )-1.0; // repeat space
  vec3 r = 1.0 - 3.0*abs(a); // work in poz quadrant, make 3 times smol, offset by 1,1,1
  float c = sdCross(r)/3.0; // distance to cross, divided for conservatism
  d = max(d,c);             // combine
  
  
     a = mod( p*3.0, 2.0 )-1.0;
  r = 1.0 - 3.0*abs(a);
  c = sdCross(r)/9.0;
  d = max(d,c);

    return d;
}

float RayMarch(vec3 ro, vec3 rd)
{
    float d0 = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++)
    {
        vec3 p = ro + rd * d0;
        
        float d = GetDist(p);
        d0 += d;
        
        if (d < SURF_DIST || d0 > MAX_DIST) break;
    }
    
    return d0;
}


vec3 GetNormal(vec3 p)
{
    vec2 e = vec2(0.01, 0.0);
    float d = GetDist(p);
    vec3 n = d - vec3(GetDist(p - e.xyy),
                  GetDist(p - e.yxy),
                  GetDist(p - e.yyx)); // apparently +- makes diff
    return normalize(n);
}

float GetLight(vec3 p)
{
    vec3 lightPos = vec3(0.0,10.0,-6.0);
    //lightPos.xz += vec2(sin(iTime), cos(iTime)) * 5.0;
    
    vec3 normal = GetNormal(p);
    vec3 lightDir = normalize(lightPos - p);
    float l = dot(normal, lightDir);
    
    // shadow 
    float d = RayMarch(p + SURF_DIST * normal * 2.0, lightDir);
    if (d < length(lightPos - p)) l *= 0.1;
    
    return clamp(l, 0.0, 1.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
    
    vec3 rd = normalize(vec3(uv.x, uv.y, 1.0));
    vec3 ro = vec3(0., 1.6, -6.0);
    
    float d = RayMarch(ro, rd);

    vec3 col = vec3(0.0);
    if (d < 20.0f)
    {
      vec3 p = ro + rd * d;
      float l = GetLight(p);
      
      col = vec3(l);
    }
    
    

    fragColor = vec4(col,1.0);
}

https://www.youtube.com/watch?v=Dzan85c3TiM
https://iquilezles.org/articles/distfunctions/
https://iquilezles.org/articles/menger/
https://programming.vip/docs/shadertoy-tutorial-part-15-channels-textures-and-buffers.html
https://learnopengl.com/Guest-Articles/2022/Compute-Shaders/Introduction
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture
https://docs.nvidia.com/cuda/cuda-c-programming-guide/
https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

A CUDA core is a arithmetic pipeline capable of performing one single precision floating point operation per cycle. CUDA core count and frequency can be used to compare the theoretical single precision performance of two different NVIDIA GPUs.

As a CUDA programmer you should completely avoid the notion of CUDA coers as they are not relevant to the design, implementation, or performance of a kernel.

A NVIDIA GPUs contains 1-N Streaming Multiprocessors (SM). Each SM has 1-4 warp schedulers. Each warp scheduler has a register file and multiple execution units. The execution units may be exclusive to the warp scheduler or shared between schedulers. Execution units include CUDA cores (FP/INT), special function units, texture, and load store units. The Fermi and Kepler white papers provide additional information.

*/