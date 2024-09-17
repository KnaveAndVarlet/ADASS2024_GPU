
//  ------------------------------------------------------------------------------------------------

//                                  R e n d e r e r  M e t a l . h
//
//  The Renderer is a specialised C++ class designed to display sections of the Mandelbrot
//  set as part of the Mandelbrot display program.
//
//  The Mandelbrot program uses a compute handler to create an image of an area of the
//  Mandelbrot set. This image is created in a floating point array of dimensions Nx by Ny,
//  and it is the job of the Renderer to display this, using Metal, in a section of the
//  application's window set up as an MTKView. The dimensions of the view will change from
//  time to time as the window is resized. The dimensions of the image do not have to match
//  the dimensions of the view - the renderer can assume that the contents of the image may
//  change rapidly (perhaps changing at the frame rate of the application, particularly when
//  the image is being zoomed), but the Nx by Ny dimensions will change rarely if at all.
//  This allows the Renderer to set up the geometry of the display in a fixed manner, with
//  usually only the colour values for the various elements it uses being expected to change.
//
//  The constructor for the Renderer has to be passed the graphics device it is to use.
//  The Renderer needs to know the Nx by Ny dimensions of the image to be displayed. These
//  are passed to it using a SetImageSize() call. It needs to know the size of the MTKView it is
//  to use for the display, which is passed to it using SetDrawableSize(). The values in
//  the floating point image has to display are in fact always integers in the range 0 to
//  (MaxIter - 1), where MaxIter is the maximum number of iterations used by the compute
//  handler. Since it helps the Renderer to know the range of data values to expect, it
//  needs to be passed MaxIter, using a call to SetMaxIter(). The Draw() call asks the
//  Renderer to draw a new version of the image, and the image and the MTKView to use
//  are passed in the Draw() parameters.
//
//  History:
//     22 Jul 2023. First version after repackaging of the original testbed code. KS.
//     30 Sep 2023. Introduced USE_VULKAN to allow testing of a Vulkan version of this code. KS.
//      4 Nov 2023. Changed to use USE_VULKAN_GRAPHICS instead of USE_VULKAN to allow graphics
//                  and compute Vulkan/Metal usage to be different. KS.
//      6 Feb 2024. Introduced _useManagedBuffers to allow testing of shared vs managed buffers. KS.
//     26 Feb 2024. Rationalising slightly the Metal and Vulkan versions, this has been renamed
//                  from Renderer.h to RendererMetal.h and use of USE_VULKAN_GRAPHICS has been
//                  dropped. Definition of MandelRendererDevice and MandelRendererView have
//                  been added. KS.
//     30 Aug 2024. Added use of a debug handler and support for GetDebugOptions(). Also added
//                  Initialise(). This brings this up to date with the latest changes to the
//                  Vulkan version. KS.
//      4 Sep 2024. Added support for SetOverlay(). KS.

#ifndef __RendererMetal__
#define __RendererMetal__

#include "Metal/Metal.hpp"
#include "MetalKit/MetalKit.hpp"
#include "MsecTimer.h"
#include "DebugHandler.h"

//  The MandelRenderDevice type is defined here so a controller can know what sort
//  of device is expected by the constructor. (A Vulkan version of the controller,
//  for example, would expect a different type of device.) Ditto MandelRendererView
//  for the view type that needs to be passed to Draw().

#define MandelRendererDevice MTL::Device
#define MandelRendererView MTK::View

class Renderer
{
    public:
        Renderer(MTL::Device* pDevice );
        ~Renderer();
        void Initialise(const std::string& DebugLevels);
        void SetImageSize (int Nx, int Ny);
        void SetDrawableSize (float width, float height);
        void SetMaxIter (int MaxIter);
        void SetOverlay(float* XPosns,float* YPosns,int NPosns);
        void Draw(MTK::View* pView, float* imageData);
        static std::string GetDebugOptions(void);
    private:
        void SetColourData(float* imageData,int Nx,int Ny);
        void SetColourDataHistEq(float* imageData,int Nx,int Ny);
        bool BuildShaders();
        void BuildBuffers();
        void GetRGB (int Index, float* R, float* G, float* B);
        void PercentileRange (float* imageData,int Nx,int Ny,float Percentile,
                          float* rangeMin,float* rangeMax);
        static const std::string _debugOptions;
        MTL::Device* _pDevice;
        MTL::CommandQueue* _pCommandQueue;
        MTL::RenderPipelineState* _pPSO;
        MTL::Buffer* _pVertexPositionsBuffer;
        MTL::Buffer* _pVertexColorsBuffer;
        MTL::Buffer* _pOverlayVertexBuffer;
        MTL::Buffer* _pOverlayColorsBuffer;
        int _maxOverVerts;
        int _overVerts;
        MsecTimer _frameTimer;
        DebugHandler _debug;
        bool _useManagedBuffers;
        float _viewWidth;
        float _viewHeight;
        int _frames;
        int _iterLimit;
        int _nx;
        int _ny;
};

#endif
