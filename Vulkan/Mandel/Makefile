#
#  This builds a Vulkan version of a Mandelbrot program that uses
#  Vulkan code for the GPU generation of the Mandelbrot images,
#  and Vulkan code to display them. It assumes all the source code
#  for the program itself is in the default directory.
#
#  The code uses the glfw window library and the glm math library,
#  as well as Vulkan itself. On some systems you may need to set
#  the CPATH and LIBRARY_PATH environment variables to pick these
#  up.
#
#  To build and run:
#
#  make
#  ./Mandel
#
#  To remove all the built files:
#
#  make clean

LIBRARIES = -lglfw -lvulkan

INCLUDES =

OBJECTS = Main.o RendererVulkan.o WindowHandler.o KVVulkanFramework.o \
    MandelController.o MandelComputeHandlerVulkan.o TcsUtil.o \
	Wildcard.o CommandHandler.o ReadFilename.o
    
SHADERS = MandelFrag.spv MandelVert.spv MandelComp.spv \
                                               MandelDComp.spv

target : Mandel $(SHADERS)

Mandel : Main.o $(OBJECTS)
	c++ -Wall $(OBJECTS) $(LIBRARIES) -o Mandel

Main.o : Main.cpp WindowHandler.h RendererVulkan.h MandelController.h
	c++ -c -Wall -std=c++17 $(INCLUDES) Main.cpp

KVVulkanFramework.o : KVVulkanFramework.cpp KVVulkanFramework.h DebugHandler.h
	c++ -c -Wall -std=c++17 KVVulkanFramework.cpp

MandelController.o : MandelController.cpp MandelController.h \
	MandelComputeHandlerVulkan.h RendererVulkan.h KVVulkanFramework.h
	c++ -c -Wall -std=c++17 $(INCLUDES) MandelController.cpp

RendererVulkan.o : RendererVulkan.cpp RendererVulkan.h \
		  KVVulkanFramework.h
	c++ -c -Wall -std=c++17 -O3  $(INCLUDES) RendererVulkan.cpp

WindowHandler.o : WindowHandler.cpp WindowHandler.cpp
	c++ -c -Wall -std=c++17 $(INCLUDES) WindowHandler.cpp
	
MandelComputeHandlerVulkan.o : \
          MandelComputeHandlerVulkan.cpp \
		  MandelComputeHandlerVulkan.h \
		  KVVulkanFramework.h
	c++ -c -Wall -std=c++17 -O3 $(INCLUDES) MandelComputeHandlerVulkan.cpp
	   
TcsUtil.o : TcsUtil.cpp TcsUtil.h
	c++ -c -Wall -std=c++17 TcsUtil.cpp
	
Wildcard.o : Wildcard.cpp Wildcard.h
	c++ -c -Wall -std=c++17 Wildcard.cpp

ReadFilename.o : ReadFilename.cpp ReadFilename.h
	c++ -c -Wall -std=c++17 ReadFilename.cpp

CommandHandler.o : CommandHandler.cpp CommandHandler.h
	c++ -c -Wall -std=c++17 CommandHandler.cpp

MandelVert.spv : Mandel.vert
	glslc Mandel.vert -O -o MandelVert.spv

MandelFrag.spv : Mandel.frag
	glslc Mandel.frag -O -o MandelFrag.spv
	
MandelComp.spv : Mandel.comp
	glslc Mandel.comp -O -o MandelComp.spv
	
MandelDComp.spv : MandelD.comp
	glslc MandelD.comp -O -o MandelDComp.spv

clean :
	@rm -f Mandel $(SHADERS) $(OBJECTS)
