
#define GLEW_NO_GLU // i dont think i need this
#define GLEW_STATIC
#include "GL/glew.h"
#include "glfw3.h"

#define WIDTH 1000 
#define HEIGHT 1000

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <assert.h>

#include <math.h> 



class float3
{
public:
    float x,y,z;
};

float3 make_float3(float x, float y, float z)
{
    float3 f;
    f.x = x;
    f.y = y;
    f.z = z;
    return f;
}
float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}
float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
float3 operator*(float a, float3 b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}
float3 operator/(float a, float3 b)
{
    return make_float3(a / b.x, a / b.y, a / b.z);
}
float3 operator+(float a, float3 b)
{
    return make_float3(a + b.x, a + b.y, a + b.z);
}
float3 operator-(float a, float3 b)
{
    return make_float3(a - b.x, a - b.y, a - b.z);
}
float3 cross(float3 a, float3 b) // https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h
{
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}


#define ASSERT(x) if (!(x)) assert(false)

#define GLCall(x) x
// #define GLCall(x) GLClearError();\
//     x;\
//     ASSERT(GLCheckError())

static void GLClearError()
{
    while (glGetError() != GL_NO_ERROR);
}

static bool GLCheckError()
{
    while (GLenum error = glGetError())
    {
        
        std::cout << "[OpenGL Error] ";
          std::cout << std::endl;
          return false;
    }
    return true;
}

struct ShaderProgramSource
{
    std::string VertexSource;
    std::string FragmentSource;
};

static struct ShaderProgramSource ParseShader(const std::string& filepath)
{
    enum class ShaderType
    {
        NONE = -1, VERTEX = 0, FRAGMENT = 1
    };

    std::ifstream stream(filepath);
    std::string line;
    std::stringstream ss[2];
    ShaderType type = ShaderType::NONE;

    while (getline(stream, line))
    {
        if (line.find("#shader") != std::string::npos)
        {
            if (line.find("vertex") != std::string::npos)
                type = ShaderType::VERTEX;
            else if (line.find("fragment") != std::string::npos)
                type = ShaderType::FRAGMENT;
        }
        else
        {
            ss[(int)type] << line << '\n';
        }
    }

    return { ss[0].str(), ss[1].str() };
}

static unsigned int CompileShader(unsigned int type, const std::string& source)
{
    GLCall( unsigned int id = glCreateShader(type) );
    const char* src = source.c_str();
    GLCall( glShaderSource(id, 1, &src, nullptr) );
    GLCall( glCompileShader(id) );

    // Error handling
    int result;
    GLCall( glGetShaderiv(id, GL_COMPILE_STATUS, &result) );
    std::cout << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader compile status: " << result << std::endl;
    if ( result == GL_FALSE )
    {
        int length;
        GLCall( glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length) );
        char* message = (char*) alloca(length * sizeof(char));
        GLCall( glGetShaderInfoLog(id, length, &length, message) );
        std::cout 
            << "Failed to compile "
            << (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
            << "shader"
            << std::endl;
        std::cout << message << std::endl;
        GLCall( glDeleteShader(id) );
        return 0;
    }

    return id;
}

static unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    // create a shader program
    unsigned int program = glCreateProgram();
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);

    GLCall( glAttachShader(program, vs) );
    GLCall( glAttachShader(program, fs) );

    GLCall( glLinkProgram(program) );

    GLint program_linked;

    GLCall( glGetProgramiv(program, GL_LINK_STATUS, &program_linked) );
    std::cout << "Program link status: " << program_linked << std::endl;
    if (program_linked != GL_TRUE)
    {
        GLsizei log_length = 0;
        GLchar message[1024];
        GLCall( glGetProgramInfoLog(program, 1024, &log_length, message) );
        std::cout << "Failed to link program" << std::endl;
        std::cout << message << std::endl;
    }

    GLCall( glValidateProgram(program) );

    GLCall( glDeleteShader(vs) );
    GLCall( glDeleteShader(fs) );

    return program;
}


class Camera{
public:
    float3 position = make_float3(0.f, 0.f, -7.f);
    float3 lookat = make_float3(0.f, 0.f, -6.f);
    float3 right = make_float3(1.0, 0.0, 0.0);
    float3 up = make_float3(0.0, 1.0, 0.0);
    float3 mov_direction = make_float3(0.0, 0.0, 0.0);
    float MOV_AMT = 0.1f; // how slowly we are moving (zooming)
    float ROT_AMT = 0.03f;
    int is_moving = 0; // only needed bc it feels better when you can freely press 2 buttons and still move
    enum MOVE_TYPE { MOVE_FORWARD, MOVE_BACKWARD, MOVE_UP, MOVE_DOWN, MOVE_RIGHT, MOVE_LEFT, ROTATE_RIGHT, ROTATE_LEFT, };
    MOVE_TYPE move_type;
    GLFWwindow* window;

    GLFWwindow* Init()
    {
        glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
        // standard OpenGL
        if (!glfwInit()){fprintf( stderr, "Failed to initialize GLFW\n" );exit(420);}
        GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "C U D A", NULL, NULL);
        if (!window){fprintf( stderr, "Failed to open GLFW window" ); glfwTerminate(); exit(420);}
        glfwMakeContextCurrent(window);
        //glfwSwapInterval(0); // VSYNC
        if (glewInit() != GLEW_OK){fprintf(stderr, "Failed to initialize GLEW\n"); glfwTerminate();exit(420);}
        std::cout << glGetString(GL_VERSION) << std::endl;


        // funky glfw + class mix:
        glfwSetWindowUserPointer(window, this);
        auto func = [](GLFWwindow* w, int key, int scancode, int action, int mods)
        {
            static_cast<Camera*>(glfwGetWindowUserPointer(w))->keyboardPressed(w, key, scancode, action, mods);
        };
        glfwSetKeyCallback(window, func);
        
        this->window = window; // need it for rendering
        return window;
    }

    void keyboardPressed(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            switch (key)
            {
                case GLFW_KEY_O: // move slower (zoom in)
                    MOV_AMT *= 0.5f; break;
                case GLFW_KEY_P: // move faster (zoom out)
                    MOV_AMT *= 2.0f; break;
                case GLFW_KEY_RIGHT: move_type = ROTATE_RIGHT; is_moving++; break;
                case GLFW_KEY_LEFT: move_type = ROTATE_LEFT; is_moving++; break;
                case GLFW_KEY_W: move_type = MOVE_FORWARD; is_moving++; break;
                case GLFW_KEY_S: move_type = MOVE_BACKWARD; is_moving++; break;
                case GLFW_KEY_D: move_type = MOVE_RIGHT; is_moving++; break;
                case GLFW_KEY_A: move_type = MOVE_LEFT; is_moving++; break;
                case GLFW_KEY_E: move_type = MOVE_UP; is_moving++; break;
                case GLFW_KEY_Q: move_type = MOVE_DOWN; is_moving++; break;
                case GLFW_KEY_ESCAPE: exit(420); break;
                default:break;
            }
        }
        if (action == GLFW_RELEASE)
        {
            switch (key)
            {
            case GLFW_KEY_RIGHT:
            case GLFW_KEY_LEFT:
            case GLFW_KEY_W:
            case GLFW_KEY_S:
            case GLFW_KEY_D:
            case GLFW_KEY_A:
            case GLFW_KEY_E:
            case GLFW_KEY_Q: is_moving--; break;
            default:break;;
            }

            // these 2 lines fix the problem where we stop but the next frame is already drawn, so going back gives a jerk
            GLCall( glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr));
            glfwSwapBuffers(window);
        }
    }

    void move()
    {
        if (is_moving)
        {
            switch (move_type) // this effectively lets you either move or turn at once. should i do both?
            {
            case MOVE_FORWARD:
                mov_direction = lookat - position;
                position = position + mov_direction * MOV_AMT;
                lookat = lookat + mov_direction * MOV_AMT;
                break;
            case MOVE_BACKWARD:
                mov_direction = position - lookat;
                position = position + mov_direction * MOV_AMT;
                lookat = lookat + mov_direction * MOV_AMT;
                break;
            case ROTATE_RIGHT:
            {
                float theta = ROT_AMT; // TODO: this is stuck in XZ plane, need to make it relative to current orientation
                // move position to origin 
                float x1 = lookat.x - position.x;
                float z1 = lookat.z - position.z;
                // rotate 
                float x2 = cos(theta) * x1 - sin(theta) * z1;
                float z2 = sin(theta) * x1 + cos(theta) * z1;
                // move position back 
                lookat.x = position.x + x2;
                lookat.z = position.z + z2;
                break;
            }
            case ROTATE_LEFT: // see above 
            {
                float theta = -ROT_AMT;
                float x1 = lookat.x - position.x;
                float z1 = lookat.z - position.z;
                float x2 = cos(theta) * x1 - sin(theta) * z1;
                float z2 = sin(theta) * x1 + cos(theta) * z1;
                lookat.x = position.x + x2;
                lookat.z = position.z + z2;
                break;
            }
            case MOVE_UP:
                position = position + up * MOV_AMT;
                lookat = lookat + up * MOV_AMT;
                break;
            case MOVE_DOWN:
                position = position - up * MOV_AMT;
                lookat = lookat - up * MOV_AMT;
                break;
            case MOVE_RIGHT:
            {
                float3 right = cross(lookat - position, up); // make the right-vector
                lookat = lookat + right * MOV_AMT; // add right-vector to pos & lookat
                position = position + right * MOV_AMT;
                break;
            }
                
            case MOVE_LEFT:
            {
                float3 left = cross(up, lookat - position);
                lookat = lookat + left * MOV_AMT;
                position = position + left * MOV_AMT;
                break;
            }
                
            }

            // Render, only when we are moving to save GPU heat
            GLCall( glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr));
            glfwSwapBuffers(window);
        }
    }
};




int main()
{
    Camera camera;// camera and window are intertwined
    GLFWwindow* window = camera.Init();
	
    

    float positions[] = {-1.0f, -1.0f, 1.0f, -1.0f, 1.0f,  0.2f, -1.0f,  0.2f};
    unsigned int indices[] = {0, 1, 2, 2, 3, 0};

    // VAO
	unsigned int vao;
	GLCall( glGenVertexArrays(1, &vao) );
	GLCall( glBindVertexArray(vao) );
    // VBO
    unsigned int buffer;
    GLCall( glGenBuffers(1, &buffer) );
    GLCall( glBindBuffer(GL_ARRAY_BUFFER, buffer) );
    GLCall( glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(float), positions, GL_STATIC_DRAW) );
    GLCall( glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0) );
    GLCall( glEnableVertexAttribArray(0) );
    // IBO
    unsigned int ibo;
    GLCall( glGenBuffers(1, &ibo) );
    GLCall( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo) );
    GLCall( glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(unsigned int), indices, GL_STATIC_DRAW) );

    // shader
    ShaderProgramSource source = ParseShader("Basic.shader");
    unsigned int shader = CreateShader(source.VertexSource, source.FragmentSource);
    GLCall( glUseProgram(shader) );
    GLCall( unsigned int u_ta = glGetUniformLocation(shader, "ta") );
    GLCall( unsigned int u_ro = glGetUniformLocation(shader, "ro") );
    ASSERT(u_ta != -1);
    ASSERT(u_ro != -1);
    GLCall( glUseProgram(0) );
    GLCall( glBindBuffer(GL_ARRAY_BUFFER, 0) );
    GLCall( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) );
    GLCall( glBindVertexArray(0) );
    // Instead of binding vertex buffer, attrib pointer, just bind Vertex Array Object
    GLCall( glBindVertexArray(vao) );
    // Bind index buffer
    GLCall( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo) );
    // set shader and set uniform color
    GLCall( glUseProgram(shader) );

    while (1)
	{
        camera.move();
        float3 ro = camera.position;
        float3 ta = camera.lookat;
        GLCall( glUniform3f(u_ta, ta.x,ta.y,ta.z) );
        GLCall( glUniform3f(u_ro, ro.x,ro.y,ro.z) );

		glfwPollEvents();
	} 

	GLCall( glDeleteBuffers(1, &buffer) );
	GLCall( glDeleteVertexArrays(1, &vao) );
	GLCall( glDeleteProgram(shader) );
	glfwTerminate();
	return 0;
}

