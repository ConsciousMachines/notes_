# files split into OBJ (my code) and OBJ_VENDOR which is vendor stuff that doesn't need recompiling
EXE = fractal_explorer
SRC = src/main.cu src/kernel.cu src/camera.cpp src/file_ops.cpp 
IMGUI_DIR = vendor/imgui
SRC_VENDOR = glad/src/glad.c  $(IMGUI_DIR)/imgui.cpp $(IMGUI_DIR)/imgui_demo.cpp $(IMGUI_DIR)/imgui_draw.cpp $(IMGUI_DIR)/imgui_tables.cpp $(IMGUI_DIR)/imgui_widgets.cpp
SRC_VENDOR += $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
OBJ = $(addprefix build/, $(addsuffix .o, $(basename $(notdir $(SRC)))))
OBJ_VENDOR = $(addprefix build/, $(addsuffix .o, $(basename $(notdir $(SRC_VENDOR)))))


NVCC          := /usr/local/cuda/bin/nvcc -ccbin g++
NVCC_FLAGS    := -m64 --threads 0 --std=c++11
NVCC_INCLUDES := -I/home/chad/Desktop/_backups/py/cuda-samples-11.6/Common -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends -Ivendor/glad/include -Iinclude -Ivendor/stb

# Gencode arguments
SMS           := 35 37 50 52 60 61 70 75 80 86
HIGHEST_SM    := 86
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval NVCC_GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
NVCC_GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)


all: $(EXE)

build/kernel.o: src/kernel.cu 
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -o $@ -c $<

build/main.o: src/main.cu 
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -o $@ -c $<

build/%.o: src/%.cpp
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -c -o $@ $<

build/%.o:$(IMGUI_DIR)/%.cpp
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -c -o $@ $<

build/%.o:$(IMGUI_DIR)/backends/%.cpp
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -c -o $@ $<

build/glad.o: vendor/glad/src/glad.c 
	$(NVCC) $(NVCC_INCLUDES) $(NVCC_FLAGS) $(NVCC_GENCODE_FLAGS) -c -o $@ $<

$(EXE): $(OBJ) $(OBJ_VENDOR)
	$(NVCC) -m64 $(NVCC_GENCODE_FLAGS) -o $@ $+ -lGL `pkg-config --static --libs glfw3`
	mv $@ ~/$@

clean:
	rm -f $(EXE) $(OBJ)
	rm ~/$(EXE)
