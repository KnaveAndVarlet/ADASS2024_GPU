
//  ------------------------------------------------------------------------------------------------

//                           R e n d e r e r  V u l k a n . h
//
//  The Renderer is a specialised C++ class designed to display sections of the Mandelbrot
//  set as part of the Mandelbrot display program.
//
//  The Mandelbrot program uses a compute handler to create an image of an area of the
//  Mandelbrot set. This image is created in a floating point array of dimensions Nx by Ny,
//  and it is the job of the Renderer to display this, in a section of the application's window.
//  The dimensions of the view will change from time to time as the window is resized. The
//  dimensions of the image do not have to match the dimensions of the view - the renderer can
//  assume that the contents of the image may change rapidly (perhaps changing at the frame rate
//  of the application, particularly when the image is being zoomed), but the Nx by Ny dimensions
//  will change rarely if at all. This allows the Renderer to set up the geometry of the display
//  in a fixed manner, with usually only the colour values for the various elements it uses
//  being expected to change.
//
//  The constructor for the Renderer has to be passed the graphics device it is to use.
//  The Renderer needs to know the Nx by Ny dimensions of the image to be displayed. These
//  are passed to it using a SetImageSize() call. It needs to know the size of the window it is
//  to use for the display, which is passed to it using SetDrawableSize(). The values in
//  the floating point image has to display are in fact always integers in the range 0 to
//  (MaxIter - 1), where MaxIter is the maximum number of iterations used by the compute
//  handler. Since it helps the Renderer to know the range of data values to expect, it
//  needs to be passed MaxIter, using a call to SetMaxIter(). The Draw() call asks the
//  Renderer to draw a new version of the image, and the image and the window to use
//  are passed in the Draw() parameters.
//
//  This version of the Renderer uses Vulkan.
//
//  History:
//      3rd Nov 2023. First working version, based on the Metal version. KS.
//     30th Jan 2024. Added _commandPool to support use of staged buffers. KS.
//     23rd Feb 2024. Introduced SetVertexPositions() and SetVertexDefaultColours(). KS.
//     26th Feb 2024. General tidying of Metal and Vulkan versions. Now matches the
//                    Metal version by defining the new types MandelRendererDevice
//                    and MandelRendererView. KS.
//     23rd Aug 2024. Added use of a debug handler. KS.
//     23rd Aug 2024. Added support for GetDebugOptions(). KS.
//     14th Sep 2024. Modified following renaming of Framework routines and types. KS.

#ifndef __RendererVulkan__
#define __RendererVulkan__

#include "KVVulkanFramework.h"
#include "MsecTimer.h"
#include "DebugHandler.h"

//  The MandelRenderDevice type is defined here so a controller can know what sort
//  of argument is expected by the constructor. (A Metal version of the controller,
//  for example, actually does expect a Metal device, whereas this Vulkan version
//  needs to be passed a pointer to a basic framework.) Ditto MandelRendererView
//  for the view type that needs to be passed to Draw() - the Vulkan version of
//  Draw() ignores the view argument, but it is needed by the Metal version.

#define MandelRendererDevice KVVulkanFramework
#define MandelRendererView void

//  A position vector contains the X,Y positions for a vertex, and must match the layout
//  expected by the glsl code in the shaders, which expect a vec2 quantity.

typedef struct {
    float X;
    float Y;
} PositionVec;

//  A colour vector contains the R,G,B colours for a vertex, and must match the layout
//  expected by the glsl code in the shaders, which expect a vec3 quantity.

typedef struct {
    float R;
    float G;
    float B;
} ColourVec;

class Renderer
{
    public:
        //  Constructor. For this Vulkan-based version, the setup argument must be the address
        //  of a Vulkan basic framework object that has already had its basic setup performed
        //  (including interaction with the windowing system) and a logical device created.
        Renderer(KVVulkanFramework* SetupArgument);
        ~Renderer();
        void Initialise(const std::string& DebugLevels);
        void SetImageSize (int Nx, int Ny);
        void SetDrawableSize (float width, float height);
        void SetMaxIter (int MaxIter);
        void SetOverlay(float* XPosns,float* YPosns,int NPosns);
        void Draw(void* pView, float* imageData);
        static std::string GetDebugOptions(void);
    private:
        void SetColourData(float* imageData,int Nx,int Ny);
        void SetColourDataHistEq(float* imageData,int Nx,int Ny);
        bool BuildShaders();
        void BuildBuffers();
        void SetVertexPositions (PositionVec positions[],int Nx,int Ny);
        void SetVertexDefaultColours (ColourVec colours[],int Nx,int Ny);
        void GetRGB (int Index, float* R, float* G, float* B);
        void PercentileRange (float* imageData,int Nx,int Ny,float Percentile,
                          float* rangeMin,float* rangeMax);
        static const std::string _debugOptions;
        MsecTimer _frameTimer;
        KVVulkanFramework* _frameworkPtr;
        DebugHandler _debug;
        int _imageCount;
        int _currentImage;
        std::vector<VkCommandBuffer> _commandBuffers;
        VkPipeline _pipeline;
        VkPipeline _overlayPipeline;
        VkCommandPool _commandPool;
        int _positionsIndex;
        int _coloursIndex;
        VkShaderModule _vertexShader;
        VkShaderModule _fragmentShader;
        std::vector<KVVulkanFramework::KVBufferHandle> _bufferHandles;
        std::vector<KVVulkanFramework::KVBufferHandle> _overlayBufferHandles;
        int _maxOverVerts;
        int _overVerts;
        PositionVec* _positionsMemAddr;
        long _positionsBytes;
        ColourVec* _coloursMemAddr;
        long _coloursBytes;
        PositionVec* _overlayPositionsMemAddr;
        ColourVec* _overlayColoursMemAddr;
        long _overlayPositionsBytes;
        long _overlayColoursBytes;
        float _viewWidth;
        float _viewHeight;
        int _frames;
        int _iterLimit;
        int _nx;
        int _ny;
};

#endif

/*
                                P r o g r a m m i n g  N o t e s
      
    o   It's annoying that the constructor needs to be passed a pointer to a semi-initialised
        framework. It would be cleaner if the Renderer could simply be passed nothing and was
        left to do the early initialisation itself. Unfortunately, that initialisation involves
        a slightly messy interaction with the window handler, and I didn't want to have to
        introduce code that depended on the window handler into the Renderer code.
 
    o   In the original Metal version of the program, the Mandel compute handler and the
        Renderer are simply passed the default Metal device in their constructors. This seemed
        neat, but that doesn't work in this Vulkan version because I didn't want to assume
        which window handler was going to be used. (In the Metal version, you know it's
        going to be the standard OS X window system.)
 */
