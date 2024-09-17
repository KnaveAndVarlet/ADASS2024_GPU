#
#  This builds a Vulkan version of Adder, a demonstration program
#  that uses Vulkan code to add values to a floating point array
#  using a GPU, and tests the result and compares the timings with
#  doing the same operation using the CPU. It assumes all the
#  source code is in the default directory.
#
#  The code assumes that Vulkan has been installed on the machine
#  and can be found by default. It also assumes that glslc - the
#  GLSL shader code compiler - has been installed.
#
#  To build and run:
#
#  make clean
#  make
#  ./Adder
#
#  ./Adder help     provides a description of the command line
#                   parameters.
     
Target : Adder Adder.spv

LIBRARIES = -lvulkan

INCLUDES =

OBJ_FILES = AdderVulkan.o TcsUtil.o Wildcard.o CommandHandler.o \
                                ReadFilename.o KVVulkanFramework.o
                        
Adder : $(OBJ_FILES)
	c++ -Wall -std=c++17 $(OBJ_FILES) $(LIBRARIES) -o Adder

AdderVulkan.o : AdderVulkan.cpp MsecTimer.h
	c++ -c -Wall -std=c++17 -O3 $(INCLUDES) AdderVulkan.cpp
	   	
TcsUtil.o : TcsUtil.cpp TcsUtil.h
	c++ -c -Wall -ansi -pedantic -std=c++17 TcsUtil.cpp

Wildcard.o : Wildcard.cpp Wildcard.h
	c++ -c -Wall -ansi -pedantic -std=c++17 Wildcard.cpp

CommandHandler.o : CommandHandler.cpp CommandHandler.h
	c++ -c -Wall -ansi -pedantic -std=c++17 CommandHandler.cpp

ReadFilename.o : ReadFilename.cpp ReadFilename.h
	c++ -c -Wall -ansi -pedantic -std=c++17 ReadFilename.cpp

KVVulkanFramework.o : KVVulkanFramework.cpp KVVulkanFramework.h \
					                          DebugHandler.h
	c++ -c -Wall -std=c++17 KVVulkanFramework.cpp

Adder.spv : Adder.comp
	glslc Adder.comp -O -o Adder.spv

clean :
	@rm -f Adder Adder.spv $(OBJ_FILES)
