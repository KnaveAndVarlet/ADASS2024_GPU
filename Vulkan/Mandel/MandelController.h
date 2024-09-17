//
//                          M a n d e l  C o n t r o l l e r . h
//
//  A MandelController coordinates the operation of a MandelComputeHandler and a Renderer,
//  using the compute handler to create an image of the Mandelbrot set with specified
//  parameters, and then using the Renderer to display this in an view or surface that has
//  already been created. The controller handles interaction between the compute handler and
//  the renderer and the application in which they are running.
//
//  The renderer and the compute handler both need to know how to access the GPU device
//  to be used, and the renderer also needs to know how to access the view it is to
//  use. These are passed to the controller as arguments to Initialise(), which should be
//  the first call made to the controller. During initialisation, the controller creates
//  the compute handler and the renderer and passes them what they need to know about the
//  the GPU device and view. The controller code is exactly the same for the Vulkan and
//  Metal versions. The details of the device and view parameters passed to Initialise()
//  differ between the two versions, as do the compute handler and renderer, but all the
//  controller does with these parameters is pass them on to the system-dependent handler
//  and renderer. Then a call to SetViewSize() is needed to tell the renderer the view
//  dimensions (ie the size of the display window).
//
//  If the controller has a request to make of the application, it does so through an
//  object that implements the MandelAppContact interface (ie inherits from the
//  MandelAppContact class). The application is expected to create such an object
//  and pass its address to the controller in a call to SetAppContact().
//
//  Once this is done, a call to Draw() will cause an image to be created at a default
//  size (1024 by 1024) and with default parameters for the image centre and magnification,
//  and for the number of iterations to be used.
//
//  Once SetAppContact() has been called, the controller can make calls back to
//  the supplied MandelAppContact object to provide feedback about things such as
//  magnification and whether the CPU or GPU is being used to generate images.
//
//  Note that the compute handler creates images of a specified size and the renderer
//  then stretches these for display in the View - there is no expectation that
//  the two sizes are the same. Indeed, the compute handler uses its knowledge of
//  the aspect ratio of the view to produce an image that compensates for any distortion
//  caused when the view has a non-square aspect ratio. (This may seem an odd way to
//  do it, but as it happens it makes it slightly easier for the renderer, given the way
//  it draws the image, and is simple for the compute handler.)
//
//  User interaction events can be passed on to the controller through calls to MouseUp(),
//  MouseDown(), MouseMoved(), KeyUp(), KeyDown() and ScrollWheel().
//
//  History:
//     22 Jul 2023. First version after repackaging of the original testbed code. KS.
//     25 Sep 2023. Removed the no longer used _imageData pointer. KS.
//      4 Nov 2023. For compatability between Vulkan and Metal, the Device and View
//                  parameters now have types that depend on whether or not USE_VULKAN
//                  is defined. KS.
//     26 Feb 2024. In a slight reworking of the Vulkan/Metal build setups, this is now
//                  all handed separately for graphics and compute using USE_METAL_COMPUTE
//                  and USE_METAL_GRAPHICS. (Note that the default is now to use Vulkan, as
//                  most setups can use this.) The constructor now takes two device parameters,
//                  one for graphics and one for compute. KS.
//     14 Mar 2024. Now supports use of double precision on the GPU if available. KS.
//     12 Jun 2024. Initialisation sequence reworked so constructor simply sets the instance
//                  variables to default values, and the new Initialise() call actually
//                  does the initialisation. This is neater, and now allows passing of a
//                  DebugLevels string to the compute handler. KS.
//     26 Aug 2024. Added Nx,Ny,Iter and Validate to Initialise() arguments. KS.
//      4 Sep 2024. Added support for displaying the Mandelbrot path, singly using the 'd' key
//                  or continuously using the 'e' key. This requires the Renderer to support
//                  the SetOverlay() call, and the windowing sub-system to support continuous
//                  reporting of mouse movements using MouseMoved(). KS.
//      7 Sep 2024. Some comments updated. SetCentre(), SetMagnification() and SetMaxIter()
//                  removed. At the moment the controller and renderer assume that the
//                  maximum iterations will not be changed after initially being set in the
//                  Initialise() call, and the controller handles changes to centre and
//                  magnification in response to cursor and key events. KS.

#ifndef __MandelController__
#define __MandelController__

#ifdef USE_METAL_GRAPHICS
#include "RendererMetal.h"
#else
#include "RendererVulkan.h"
#endif
#ifdef USE_METAL_COMPUTE
#include "MandelComputeHandlerMetal.h"
#else
#include "MandelComputeHandlerVulkan.h"
#endif

#include <string>

class MandelAppContact
{
public:
    ~MandelAppContact() {};
    //  Requests that the following string be displayed to the user (perhaps as the
    //  window title).
    virtual void DisplayString (const char* Title) {};
};

class MandelController
{
public:
    //  Constructor
    MandelController (void);
    //  Destructor
    ~MandelController();
    //  Initialise - the device and view types are defined by the renderer and compute handler
    //                incude files.
    void Initialise(MandelComputeDevice* ComputeDevice,MandelRendererDevice* RendererDevice,
                    MandelRendererView* View,int Nx,int Ny,int Iter,bool Validate,
                    const std::string& DebugLevels);
    //  This specifies the dimensions of the view. The renderer needs to know these.
    //  The compute handler does not need to know the actual dimensions, but it does need
    //  to know the aspect ratio that will be used to display the image.
    void SetViewSize(double Width, double Height);
    //  Introduce the controller to its request handler.
    void SetAppContact (MandelAppContact* Contact);
    //  These are called when user interaction needs to be handled.
    void ScrollWheel (float DeltaX, float DeltaY, float AtX, float AtY);
    void KeyDown (const char* Key, long Flags, float AtX, float AtY);
    void KeyUp (const char* Key, long Flags, float AtX, float AtY);
    void MouseDown (float AtX, float AtY);
    void MouseUp (float AtX, float AtY);
    void MouseMoved (float AtX, float AtY);
    //  Invoked when the view needs to be drawn or redrawn.
    void Draw();
private:
    //  A Setting structure gives the location of a point in the Mandelbrot set and
    //  the magnification to use for it.
    typedef struct {
        double XCent;
        double YCent;
        double Magnification;
    } Setting;
    enum ZoomMode {ZOOM_NONE,ZOOM_IN,ZOOM_OUT,ZOOM_TIMED};
    enum ComputeMode {AUTO_MODE,CPU_MODE,GPU_MODE};
    enum UseMode {USE_NONE,USE_CPU,USE_GPU,USE_GPU_D};
    //  Format the magnification into a suitable window title.
    std::string FormatMagnification (double Magnification);
    //  Display an updated window title.
    void RedisplayTitle();
    //  Convert from view coordinates to Mandelbrot set coordinates.
    void FrameToImageCoord(float AtX, float AtY,double* XCoord,double* YCoord);
    //  Convert from Mandelbrot set coordinates to view coordinates.
    void ImageToFrameCoord(double* XCoord,double* YCoord,float* AtX,float* AtY,int N);
    //  Calculate route of Mandelbrot calculation from given coordinates.
    int CalcRoute (double X0,double Y0,double* XPosns,double* YPosns,int MaxIter);
    //  Set the image size and create the various buffers
    void SetImageSize(int Nx, int Ny);
    //  Output help text
    void PrintHelp();
    //  Set a specific memory to specified positiona and magnification
    void SetMemory (int Memory, double XCent,double YCent,double Magnification);
    //  Set all the memories to their default values
    void SetMemoriesToDefault();
    //  Memory settings for preset displays
    Setting _Memories[10];
    //  The view for the renderer to draw into.
    MandelRendererView* _View;
    //  The computation mode.
    ComputeMode _ComputeMode;
    //  Set if the GPU supports double precision.
    bool _GPUSupportsDouble;
    //  Base size of image in X - set by Initialise() call.
    int _BaseNx;
    //  Base size of image in y - set by Initialise() call.
    int _BaseNy;
    //  Iteration limit for calculation - set by Initialise() call.
    int _Iter;
    //  Compensate for computation delays during zoom by scaling the magnification.
    bool _ScaleMagByTime;
    //  How the last image was calculated.
    UseMode _LastUsedMode;
    //  The current Zoom mode
    ZoomMode _ZoomMode;
    //  Timer used to track Zoom mode
    MsecTimer _ZoomTimer;
    //  Timer used to track Zoom mode render time.
    MsecTimer _ZoomRenderTimer;
    //  Zoom frame count (CPU)
    int _ZoomFramesCPU;
    //  Zoom frame count (GPU double)
    int _ZoomFramesGPU_D;
    //  Zoom frame count (GPU)
    int _ZoomFramesGPU;
    //  Total compute time in last Zoom (CPU)
    float _TotalComputeMsecCPU;
    //  Total compute time in last Zoom (GPU double)
    float _TotalComputeMsecGPU_D;
    //  Total compute time in last Zoom (GPU)
    float _TotalComputeMsecGPU;
    //  Total render time in last Zoom
    float _TotalRenderMsec;
    //  The Zoom timer when the last frame was drawn
    float _LastZoomMsec;
    //  True if drawing of Mandelbrot path is enabled.
    bool _Drawing;
    //  Buffer used to hold last calculated route X coordinates.
    double* _RouteX;
    //  Buffer used to hold last calculated route Y coordinates.
    double* _RouteY;
    //  Number of tracks in last calculated route.
    int _RouteN;
    //  True if mouse is being used to drag image.
    bool _InDrag;
    //  Image position in X below cursor when drag starts.
    double _DragImageX;
    //  Image position in Y below cursor when drag starts.
    double _DragImageY;
    //  The address of the compute handler.
    MandelAppContact* _AppContact;
    //  The address of the compute handler.
    MandelComputeHandler* _ComputeHandler;
    //  The address of the renderer.
    Renderer* _Renderer;
    //  The width of the view into which the image is rendered.
    float _FrameX;
    //  The height of the view into which the image is rendered.
    float _FrameY;
    //  Set internally if the image needs to be redrawn.
    bool _NeedToRedraw;
};

#endif
