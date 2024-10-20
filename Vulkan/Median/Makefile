#
#  This builds a Vulkan version of Median, a demonstration program
#  that runs a median filter through a floating point array using
#  a GPU, and tests the result and compares the timings with doing
#  the same operation using the CPU. It assumes all the source code
#  is in the default directory.
#
#  The code assumes that Vulkan has been installed on the machine
#  and can be found by default. It also assumes that glslc - the
#  GLSL shader code compiler - has been installed. It also uses the
#  cfitsio library for reading and writing FITS files. cfitsio
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
#
#  This can also build a version of Median called Medianx which
#  does not use Cfitsio (and so does not have the option of reading
#  and writing FITS files).
#
#  Modified:
#     16th Oct 2024. Added an explicit -lpthread for systems that
#                    still need it. KS.
#     18th Oct 2024. Clean no longer deletes .spv files. Cleanup
#                    does. Also added Medianx target. KS.

#  Median is the default target, and builds Median using Cfitsio.

Target : Median Median.spv

#  Medianx builds a version of Median that does not need Cfitsio,
#  but as a result cannot work with data read from FITS files.

Medianx : Median.spv

LIBRARIES = -lvulkan -lcfitsio -lpthread

LIBRARIESX = -lvulkan -lpthread

INCLUDES =

OBJ_FILES = TcsUtil.o Wildcard.o CommandHandler.o \
								ReadFilename.o KVVulkanFramework.o
                        
Median : MedianVulkan.o $(OBJ_FILES)
	c++ -Wall -std=c++17 MedianVulkan.o \
		$(OBJ_FILES) $(LIBRARIES) -o Median

MedianVulkan.o : MedianVulkan.cpp MsecTimer.h
	c++ -c -Wall -std=c++17 -O3 $(INCLUDES) MedianVulkan.cpp

Medianx : MedianVulkanx.o $(OBJ_FILES)
	c++ -Wall -std=c++17 \
		MedianVulkanx.o $(OBJ_FILES) $(LIBRARIESX) -o Medianx

MedianVulkanx.o : MedianVulkan.cpp MsecTimer.h
	c++ -c -Wall -std=c++17 -DNO_CFITSIO -O3 $(INCLUDES) \
	-o MedianVulkanx.o MedianVulkan.cpp

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

Median.spv : Median.comp
	glslc Median.comp -Os -o Median.spv

clean :
	@rm -f Median *.o Median_*.fits Medianx

cleanup :
	@rm -f Median Median.spv *.o Median_*.fits Medianx
