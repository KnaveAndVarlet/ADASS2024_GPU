//  ------------------------------------------------------------------------------------------------

//                       M a n d e l  C o m p u t e  H a n d l e r . h
//
//  A MandelComputeHandler exists to compute an image of the Mandelbrot set, with a specified
//  centre point and a specified magnification, using a specified maximum number of iterations
//  for each point in the image. The intention behind this code was to show an example of
//  how such a calculation can be done using a GPU, and to allow a comparision of the results
//  with doing the same calculation using the CPU. This is written in C++, and uses Vulkan.
//  This is a deliberately low-level approach.
//
//  The constructor has to be passed a reference to the GPU to be used. The handler needs to
//  be told the dimensions of the image to be created, using SetImageSize(). It needs to be
//  given the centre point for the image, using SetCentre(), the Magnification using
//  SetMagnification() and the maximum number of iterations using SetMaxIter(). Once these
//  have been called, a call to Compute() will cause the image to be generated using the
//  GPU. If the GPU has double precision support, a call to ComputeDouble() will generate
//  the image using the GPU in double precision. Alternatively, a call to ComputeInC() will
//  cause the image to be generated using the CPU, with multi-threaded C++ code that uses all
//  available CPU threads. The intention is that the address of the generated image will then
//  be obtained using GetImageData() and the image will then be written out or displayed in
//  some way.
//
//  The image is generated in a float array, Nx by Ny. Although a float array is used, each
//  pixel in the image will be the number of iterations that it took the code to decide
//  whether the point lies within the Mandelbrot set or not. If the code runs more than
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
//  pixels will be too small for single precision to represent properly. Note that many
//  GPUs do not support double precision. Depending on the number of display pixels used
//  for the image, this usually becomes noticeable at a magnification of around a hundred
//  thousand. The CPU code uses double precision and is usually good to magnifications of
//  up to around a hundered trillion. The handler has routines FloatOK() and DoubleOK()
//  which can be used to check whether the given parameters will cause rounding error
//  or not. If the GPU does support double precision, it will often be noticeably slower
//  than its single precision.
//
//     26th Feb 2024. General tidying of Metal and Vulkan versions. Now matches the
//                    Metal version by defining the new MandelComputeDevice. KS.
//     14th Mar 2024. Added support for double precision on the GPU, with ComputeDouble()
//                    and GPUSupportsDouble(). KS.
//     12th Jun 2024. Added Initialise(). KS.
//     13th Jun 2024. Added use of a debug handler. KS.
//     23rd Aug 2024. Added support for GetDebugOptions(). KS.
//     27th Aug 2024. Added Validate parameter to Initialise(). KS.
//     14th Sep 2024. Modified following renaming of Framework routines and types. KS.

#ifndef __MandelComputeHandlerVulkan__
#define __MandelComputeHandlerVulkan__

#include "KVVulkanFramework.h"
#include "MsecTimer.h"
#include "DebugHandler.h"

#define prec double

//  The MandelComputeDevice type is defined here so a controller can know what sort
//  of device is expected by the constructor. (A Metal version of the controller,
//  for example, would expect a different type of device. In fact, the Metal version
//  needs to be passed an actual Metal device, whereas this Vulkan version ignores
//  the device argument.)

#define MandelComputeDevice void

class MandelComputeHandler
{
    public:
        MandelComputeHandler(void*);
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
        void Compute();
        void ComputeDouble();
        void ComputeInC();
        float* GetImageData();
        static std::string GetDebugOptions (void);
    private:
        //  Single structure to pass the arguments to the compute kernel on the GPU.
        //  The structure defined in the shader code must match this exactly - be wary
        //  of alignment and size issues. There are two versions of each of the floating
        //  point quantities, one single and one double precision. Not all GPUs support
        //  double precision, so a shader that only supports single precsion will use
        //  the single precision quantities.
        struct MandelArgs {
            float xCent;
            float yCent;
            float dX;
            float dY;
            int maxIter;
            int nx;
            int ny;
            int padding;
            double xCentD;
            double yCentD;
            double dXD;
            double dYD;
        };
        static const std::string _debugOptions;
        static void ComputeInCThreads (float* Data,int Nx,int Ny,prec Xcent,prec Ycent,
                                                          prec Dx,prec Dy,int MaxIter);
        static void ComputeRangeInC (float* Data,int Nx,int Ny,int Iyst,int Iyen,
                                     prec Xcent,prec Ycent,prec Dx,prec Dy,int MaxIter);
        void InitialiseVulkanItems();
        bool FloatOKatXY(int Ix,int Iy);
        bool DoubleOKatXY(int Ix,int Iy);
        void RecomputeArgs();
        bool _statusOK;
        KVVulkanFramework* _vulkanFramework;
        DebugHandler _debug;
        bool _frameworkIsLocal;
        KVVulkanFramework::KVBufferHandle _uniformBufferHndl;
        KVVulkanFramework::KVBufferHandle _imageBufferHndl;
        void* _uniformBufferAddr;
        double _xCent;
        double _yCent;
        double _magnification;
        double _width;
        double _height;
        double _dx;
        double _dy;
        float* _imageData;
        int _maxiter;
        int _nx;
        int _ny;
        MandelArgs _currentArgs;
        VkQueue _computeQueue;
        VkCommandPool _commandPool;
        VkCommandBuffer _commandBuffer;
        VkDescriptorSetLayout _setLayout;
        VkDescriptorPool _descriptorPool;
        VkDescriptorSet _descriptorSet;
        VkPipelineLayout _computePipelineLayout;
        VkPipeline _computePipeline;
        uint32_t _workGroupCounts[3];
        bool _doubleSupportInGPU;
        VkPipelineLayout _computePipelineLayoutD;
        VkPipeline _computePipelineD;
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

    o   This really needs to be commented properly, explaining all the routines and the
        instance variables.
        
    o   On GPUs that only support single precision, a shader that uses 'double' will not
        load. A single precision shader can just ignore the double quantities at the end
        of the MandelArgs structure - that's why they're at the end. The single int called
        padding is just to get the alignment right for those final doubles.
*/
