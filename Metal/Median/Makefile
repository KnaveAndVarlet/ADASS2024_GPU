#
#  This builds a Metal version of Median, a demonstration program
#  that runs a median filter through a floating point array using
#  a GPU, and tests the result and compares the timings with doing
#  the same operation using the CPU. It assumes all the source code
#  is in the default directory.
#
#  This code should build and run on any recent Mac that has the
#  usual C++ programming tools installed, together with the cfitsio
#  library. If not, install XCode from the App Store and cfitsio
#  can be downloaded from: http://heasarc.gsfc.nasa.gov/fitsio
#  If you have cfitsio in a non-standard location, you may need
#  to modify INCLUDES and LIBRARIES.
#  All the source code is provided in the default directory.
#
#  To build and run:
#
#  make clean
#  make
#  ./Median
#
#  ./Median help    provides a description of the command line
#                   parameters.

LIBRARIES = -lcfitsio

INCLUDES =

METAL_CPP_DIR = Metal-cpp

Target : Median Compute.metallib

OBJ_FILES = MedianMetal.o TcsUtil.o Wildcard.o CommandHandler.o \
                                               ReadFilename.o
                        
Median : $(OBJ_FILES)
	clang++ -Wall -std=c++17  -framework Metal \
		-framework Foundation -framework Cocoa \
		-framework CoreGraphics -framework MetalKit  \
		$(OBJ_FILES) $(LIBRARIES) -o Median

MedianMetal.o : MedianMetal.cpp MsecTimer.h
	clang++ -c -Wall -std=c++17 \
	   -I $(METAL_CPP_DIR)/metal-cpp \
	   -I $(METAL_CPP_DIR)/metal-cpp-extensions \
	   -fno-objc-arc -O3  $(INCLUDES) MedianMetal.cpp
	   	
TcsUtil.o : TcsUtil.cpp TcsUtil.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 TcsUtil.cpp

Wildcard.o : Wildcard.cpp Wildcard.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 Wildcard.cpp

CommandHandler.o : CommandHandler.cpp CommandHandler.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 CommandHandler.cpp

ReadFilename.o : ReadFilename.cpp ReadFilename.h
	clang++ -c -Wall -ansi -pedantic -std=c++17 ReadFilename.cpp

Compute.metallib : Median.metal
	xcrun -sdk macosx metal -c -O3 Median.metal -o Median.air
	xcrun -sdk macosx metallib Median.air -o Compute.metallib

clean :
	@rm -f Median Compute.metallib Median.air $(OBJ_FILES) Median_*.fits
