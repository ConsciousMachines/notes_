IMGUI_DIR = imgui
SOURCES = main.cpp kernel/Mandelbrot_cuda.cu imgui/glad.c
SOURCES += $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SOURCES += $(IMGUI_DIR)/imgui_impl_glfw.cpp $(IMGUI_DIR)/imgui_impl_opengl3.cpp 
OBJS = $(addsuffix .o, $(basename $(notdir $(SOURCES))))


NVCC          := /usr/local/cuda/bin/nvcc -ccbin g++
NVCC_FLAGS    := -m64 --threads 0 --std=c++11 -I$(IMGUI_DIR)
NVCC_INCLUDES := -I/home/chad/Desktop/_backups/py/cuda-samples-11.6/Common

SMS           := 35 37 50 52 60 61 70 75 80 86
HIGHEST_SM    := 86
$(foreach sm,$(SMS),$(eval NVCC_GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
NVCC_GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)


%.o:%.cpp
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -c -o $@ $<

%.o:$(IMGUI_DIR)/%.cpp
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -c -o $@ $<

%.o:$(IMGUI_DIR)/%.c
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -c -o $@ $<

Mandelbrot_cuda.o: kernel/Mandelbrot_cuda.cu
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -o $@ -c $<

all: cuda1

cuda1: $(OBJS)
	$(NVCC) -m64 $(NVCC_GENCODE_FLAGS) -o $@ $+ -lGL `pkg-config --static --libs glfw3`
	mv cuda1 ~

clean:
	rm -f cuda1 $(OBJS)
