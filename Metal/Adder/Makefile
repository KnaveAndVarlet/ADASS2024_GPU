#
#  This builds a Metal version of Adder, a demonstration program
#  that uses Vulkan code to add values to a floating point array
#  using a GPU, and tests the result and compares the timings with
#  doing the same operation using the CPU. It assumes all the
#  source code is in the default directory.
#
#  This code should build and run on any recent Mac that has the
#  usual C++ programming tools installed. If not, install XCode
#  from the App Store. All the source code is provided in the
#  default directory.
#
#  To build and run:
#
#  make clean
#  make
#  ./Adder
#
#  ./Adder help     provides a description of the command line
#                   parameters.

LIBRARIES =

INCLUDES =

METAL_CPP_DIR = Metal-cpp
     
Target : Adder Compute.metallib

OBJ_FILES = AdderMetal.o TcsUtil.o Wildcard.o CommandHandler.o \
                                               ReadFilename.o
                        
Adder : $(OBJ_FILES)
	clang++ -Wall -std=c++17  -framework Metal \
		-framework Foundation -framework Cocoa \
		-framework CoreGraphics -framework MetalKit  \
		$(LIBRARIES) $(OBJ_FILES) -o Adder

AdderMetal.o : AdderMetal.cpp MsecTimer.h
	clang++ -c -Wall -std=c++17 \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -fno-objc-arc -O3 $(INCLUDES) AdderMetal.cpp
	   	
TcsUtil.o : TcsUtil.cpp TcsUtil.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 TcsUtil.cpp

Wildcard.o : Wildcard.cpp Wildcard.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 Wildcard.cpp

CommandHandler.o : CommandHandler.cpp CommandHandler.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 CommandHandler.cpp

ReadFilename.o : ReadFilename.cpp ReadFilename.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 ReadFilename.cpp

Compute.metallib : Adder.metal
	xcrun -sdk macosx metal -c -Ofast Adder.metal -o Adder.air
	xcrun -sdk macosx metallib Adder.air -o Compute.metallib

clean :
	@rm -f Adder Compute.metallib Adder.air $(OBJ_FILES)
