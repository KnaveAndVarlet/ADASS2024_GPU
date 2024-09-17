#
#          M a k e f i l e  (for Mandel, Metal version)
#
#  	This Makefile builds the Metal version of the Mandel GPU
#   demonstration program. The default target builds the program,
#   and the 'clean' target cleans out all but the source files.
#
#	All the code needed for the program should be in the default
#   directory. This assumes the program is being built on a
#   Mac that has the usual command line program development
#   tools (clang, make, xcrun), usually installed as part of XCode.
#   The program uses Apple's metal-cpp layer to call Metal from
#   C++, but since this is not a standard package found on most
#   Macs, this Makefile assumes the default directory has a copy
#   of the metal-cpp header files in a Metal-cpp sub-directory.

LIBRARIES =

INCLUDES =

METAL_CPP_DIR = Metal-cpp

OBJECTS = main.o RendererMetal.o ControllerAdaptor.o \
   MyMTKViewDelegate.o MandelComputeHandlerMetal.o MandelArgs.o \
     MandelController.o MtkPrivate.o TcsUtil.o Wildcard.o \
    CommandHandler.o ReadFilename.o

target : Mandel compute.metallib

Mandel : Main.o $(OBJECTS)
	clang++ -Wall -framework Metal \
		-framework Foundation -framework Cocoa \
		-framework CoreGraphics -framework MetalKit  \
		$(OBJECTS) -o Mandel

Main.o : Main.m
	clang -c -Wall \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -O2 Main.m

MandelArgs.o : MandelArgs.cpp MandelArgs.h
	clang -c -Wall -std=c++17 $(INCLUDES) \
	   -DUSE_METAL_COMPUTE -DUSE_METAL_GRAPHICS \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -O2 MandelArgs.cpp

MandelController.o : MandelController.cpp MandelController.h \
					MandelComputeHandlerMetal.h RendererMetal.h
	clang++ -c -Wall -std=c++17 $(INCLUDES) \
	   -DUSE_METAL_COMPUTE -DUSE_METAL_GRAPHICS \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -fno-objc-arc -O2  MandelController.cpp

RendererMetal.o : RendererMetal.cpp RendererMetal.h
	clang++ -c -Wall -std=c++17 $(INCLUDES) \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -fno-objc-arc -O2  RendererMetal.cpp

MandelComputeHandlerMetal.o : MandelComputeHandlerMetal.cpp MandelComputeHandlerMetal.h
	clang++ -c -Wall -std=c++17 $(INCLUDES) \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -fno-objc-arc -O3  MandelComputeHandlerMetal.cpp

ControllerAdaptor.o : ControllerAdaptor.mm ControllerAdaptor.h
	clang -c -Wall -std=c++17 -DUSE_METAL_COMPUTE -DUSE_METAL_GRAPHICS $(INCLUDES) \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -O2  ControllerAdaptor.mm
	   
MyMTKViewDelegate.o : MyMTKViewDelegate.m MyMTKViewDelegate.h
	clang -c -Wall  $(INCLUDES) \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -O2  MyMTKViewDelegate.m

MtkPrivate.o : MtkPrivate.cpp
	clang++ -c -Wall -std=c++17 $(INCLUDES) \
	   -I$(METAL_CPP_DIR)/metal-cpp \
	   -I$(METAL_CPP_DIR)/metal-cpp-extensions \
	   -O2  MtkPrivate.cpp

TcsUtil.o : TcsUtil.cpp TcsUtil.h
	clang++ -c -Wall -std=c++17 TcsUtil.cpp
	
Wildcard.o : Wildcard.cpp Wildcard.h
	clang++ -c -Wall -std=c++17 Wildcard.cpp

ReadFilename.o : ReadFilename.cpp ReadFilename.h
	clang++ -c -Wall -std=c++17 ReadFilename.cpp

CommandHandler.o : CommandHandler.cpp CommandHandler.h
	clang++ -c -Wall -std=c++17 CommandHandler.cpp

compute.metallib : mandel.metal
	xcrun -sdk macosx metal -c mandel.metal -o mandel.air
	xcrun -sdk macosx metallib mandel.air -o compute.metallib

clean :
	@rm -f Mandel compute.metallib mandel.air $(OBJECTS)
