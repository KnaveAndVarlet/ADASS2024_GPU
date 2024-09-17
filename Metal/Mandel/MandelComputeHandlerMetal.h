//  ------------------------------------------------------------------------------------------------

//                  M a n d e l  C o m p u t e  H a n d l e r  M e t a l . h
//
//  A MandelComputeHandler exists to compute an image of the Mandlebrot set, with a specified
//  centre point and a specified magnification, using a specified maximum number of iterations
//  for each point in the image. The intention behind this code was to show an example of
//  how such a calculation can be done using a GPU, and to allow a comparision of the results
//  with doing the same calculation using the CPU. This is written in C++, and uses Apple's
//  metal-cpp to call Metal GPU routines directly. This is a deliberately low-level approach.
//  Note that this code is Apple-specific.
//
//  The constructor has to be passed a reference to the GPU to be used. The first call to the
//  Handler should be to Initialise(), which passes it the string used to selectively enable
//  the various diagnostic levels for the program. It is also passed a 'Validate' parameter
//  which is used by the Vukan version to enable additional diagnostics, but which this
//  version ignores.
//
//  The handler needs to be told the dimensions of the image to be created, using SetImageSize().
//  It needs to be given the centre point for the image, using SetCentre(), the Magnification
//  using SetMagnification() and the maximum number of iterations using SetMaxIter(). Once these
//  have been called, a call to Compute() will cause the image to be generated using the
//  GPU. Alternatively, a call to ComputeInC() will cause the image to be generated using
//  the CPU, with multi-threaded C++ code that uses all available CPU threads. The intention is
//  that the address of the generated image will then be obtained using GetImageData() and
//  the image will then be written out or displayed in some way.
//
//  The image is generated in a float array, Nx by Ny. Although a float array is used, each
//  pixel in the image will be the number of iterations that it took the code to decide
//  whether the point lies within the Mandlebrot set or not. If the code runs more than
//  MaxIter iterations on a given point without showing the point is not outside the set, it
//  assumes it is within the set and sets the corresponding element of the array to zero.
//  Pixels corresponding to points outside the set will have values in the range 1 through
//  (MaxIter - 1). MaxIter defaults to 1024. These images are usually displayed by colour-coding
//  the image based on the pixel values.
//
//  The various image parameters can be changed and Compute() or ComputeInC() called again
//  to produce a succession of images. On most modern machines, the GPU at least will be fast
//  enough to allow a zoom in real-time through a sequence of images.
//
//  The code assumes that when the image is displayed it will be stretched to fill the area
//  of the display. If the display area is not square, this would normally result in a
//  distored image. To compensate for this, a call to SetAspect() can be used to specify
//  the dimensions of the display to be used, and the handler will itself distort the image
//  it generates to compensate. (It's probably more efficient for this distortion to be
//  handled when the image is computed, as it adds no overhead worth mentioning, whereas it
//  might slow down the display to have to do the distortion in the display code.)
//
//  At high magnifications, the single precision floating point used by the GPU will run
//  into significant rounding errors - the difference between the coordinates of adjacent
//  pixels will be too small for single precision to represent properly. Note that Apple
//  GPUs do not support double precision. Depending on the number of display pixels used
//  for the image, this usually becomes noticeable at a magnification of around a hundred
//  thousand. The CPU code uses double precision and is usually good to magnifications of
//  up to around a hundered trillion. The handler has routines FloatOK() and DoubleOK()
//  which can be used to check whether the given parameters will cause rounding error
//  or not.
//
//  History:
//     22 Jul 2023. First version after repackaging of the original testbed code. KS.
//     30 Sep 2023. Introduced USE_VULKAN to allow testing of a Vulkan version of this code. KS.
//      4 Nov 2023. Changed to use USE_VULKAN_COMPUTE instead of USE_VULKAN to allow the compute
//                  and graphics parts of the code to use Vulkan/Metal independently. KS.
//     26 Feb 2024. Rationalising slightly the Metal and Vulkan versions, this has been renamed
//                  from Renderer.cpp to RendererMetal.cpp and use of USE_VULKAN_GRAPHICS has been
//                  dropped. KS.
//     18 Jun 2024. Added Initialise(), ComputeDouble() and GPUSupportsDouble(), to be consistent
//                  with the Vulkan version. Initialise() is also a first step to support for use
//                  of a DebugHandler. KS.
//     30 Aug 2024. Added use of a debug handler, support for GetDebugOptions() and added Validate
//                  parameter to Initialise(), following recent changes to the Vulkan version. KS.


#ifndef __MandelComputeHandler__
#define __MandelComputeHandler__

#include "Metal/Metal.hpp"
#include "MetalKit/MetalKit.hpp"
#include "MsecTimer.h"
#include "DebugHandler.h"

//  The MandelComputeDevice type is defined here so a controller can know what sort
//  of device is expected by the constructor. (A Vulkan version of the controller,
//  for example, would expect a different type of device.)

#define MandelComputeDevice MTL::Device

//  'prec' is used throughout the CPU code to specify the precision to be used for the
//  calculation when performed using the CPU. See programming notes for a short discussion.

#define prec double

class MandelComputeHandler
{
    public:
        MandelComputeHandler(MTL::Device* Device);
        ~MandelComputeHandler();
        void Initialise(bool Validate,const std::string& DebugLevels);
        void SetImageSize (int Nx, int Ny);
        void SetCentre(double XCent,double YCent);
        void SetMagnification(double Magnification);
        void SetAspect(double Width, double Height);
        void SetMaxIter(int MaxIter);
        double GetMagnification();
        bool FloatOK();
        bool DoubleOK();
        bool GPUSupportsDouble();
        void GetCentre(double* XCent,double* YCent);
        void Compute ();
        void ComputeDouble();
        void ComputeInC();
        float* GetImageData();
        static std::string GetDebugOptions (void);
    private:
        //  Single structure to pass the arguments to the compute kernel on the GPU
        //  (This means only using one buffer for all the arguments - you could use
        //  separate buffers for each, but that's more overhead and messier. This
        //  structure could contain the address of the output array as well, but I'm
        //  leaving in two argument buffers just to show that works.) The structure
        //  defined in the .metal code must match this exactly - be wary of alignment
        //  and size issues.
        struct MandelArgs {
            float xCent;
            float yCent;
            float dX;
            float dY;
            int maxIter;
        };
        static void ComputeInCThreads (float* Data,int Nx,int Ny,prec Xcent,prec Ycent,
                                                          prec Dx,prec Dy,int MaxIter);
        static void ComputeRangeInC (float* Data,int Nx,int Ny,int Iyst,int Iyen,
                                     prec Xcent,prec Ycent,prec Dx,prec Dy,int MaxIter);
        void BuildComputeShader ();
        bool FloatOKatXY(int Ix,int Iy);
        bool DoubleOKatXY(int Ix,int Iy);
        void RecomputeArgs();
        static const std::string _debugOptions;
        DebugHandler _debug;
        MTL::CommandQueue* _commandQueue;
        MTL::Size _gridSize;
        MTL::Size _threadGroupDims;
        MTL::Device* _device;
        MTL::ComputePipelineState* _mandelFunction;
        double _xCent;
        double _yCent;
        double _magnification;
        double _width;
        double _height;
        double _dx;
        double _dy;
        MandelArgs _currentArgs;
        MTL::Buffer* _outputBuffer;
        float* _imageData;
        int _maxiter;
        int _nx;
        int _ny;
};

#endif

/*
                           P r o g r a m m i n g   N o t e s

    o   'prec' determines the precision used when the images are computed using the CPU.
        Having it defined here made it easy to experiment to see if there were noticeable
        speed gains to be got by using float instead of double. You can try it for yourself.
        (Spoiler: On most modern processors, there aren't. In fact double is often very
        marginally faster, I think because internally most processors nowadays implement
        float by calculating in double and then taking the time to round the results to
        float's lower precision. That doesn't mean there aren't gains in terms of array
        size to be had by using float, but that's not the issue in this code.)

*/
