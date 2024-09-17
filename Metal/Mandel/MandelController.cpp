//
//                          M a n d e l  C o n t r o l l e r . c p p
//
//  A MandelController coordinates the operation of a MandelComputeHandler and a Renderer,
//  using the compute handler to create an image of the Mandelbrot set with specified
//  parameters, and then using the Renderer to display this in an view or surface that has
//  already been created. The controller handles interaction between the compute handler and
//  the renderer and the application in which they are running. See the comments at the
//  start of MandelController.h for more details.
//
//  History:
//     22 Jul 2023. First version after repackaging of the original testbed code. KS.
//     25 Sep 2023. Removed the no longer used _imageData pointer. KS.
//      4 Nov 2023. For compatability between Vulkan and Metal, the Device and View
//                  parameters now have types that depend on whether USE_VULKAN_GRAPHICS
//                  and/or USE_VULKAN_COMPUTE are defined or not. KS.
//     26 Feb 2024. Constructor changed to use new device and view types, and to have
//                  separate device parameters for the compute handler and renderer. See
//                  comments in .h file. Some float() calls added to bypass warnings when
//                  compiling under Windows. KS.
//     14 Mar 2024. Now can make use of double precision on the GPU if available. KS.
//     12 Jun 2024. Initialisation sequence reworked so constructor simply sets the instance
//                  variables to default values, and the new Initialise() call actually
//                  does the initialisation. This is neater, and now allows passing of a
//                  DebugLevels string to the compute handler. KS.
//      2 Aug 2014  Preset locations 2 and 3 were very similar. Modified 2. KS.
//     26 Aug 2024. Added Nx,Ny,Iter and Validate to Initialise() arguments. Modified help
//                  output to reflect supplied Nx,Ny values and whether or not the GPU has
//                  double precision support. KS.
//      4 Sep 2024. Added support for displaying the Mandelbrot path, singly using the 'd' key
//                  or continuously using the 'e' key. This requires the Renderer to support
//                  the SetOverlay() call, and the windowing sub-system to support continuous
//                  reporting of mouse movements using MouseMoved(). KS.
//      7 Sep 2024. Some comments updated. SetCentre(), SetMagnification() and SetMaxIter()
//                  removed. At the moment the controller and renderer assume that the
//                  maximum iterations will not be changed after initially being set in the
//                  Initialise() call, and the controller handles changes to centre and
//                  magnification in response to cursor and key events. KS.

#include "MandelController.h"

#include <cmath>

MandelController::MandelController (void)
{
    //  Set the instance variables to default values.
    
    _AppContact = nullptr;
    _View = nullptr;
    _ComputeHandler = nullptr;
    _Renderer = nullptr;
    _FrameX = 512.0;
    _FrameY = 512.0;
    _BaseNx = 1024;
    _BaseNy = 1024;
    _Iter = 1024;
    _NeedToRedraw = true;
    _ZoomMode = ZOOM_NONE;
    _LastUsedMode = USE_NONE;
    _ZoomFramesCPU = 0;
    _ZoomFramesGPU = 0;
    _ZoomFramesGPU_D = 0.0;
    _LastZoomMsec = 0.0;
    _TotalComputeMsecCPU = 0.0;
    _TotalComputeMsecGPU = 0.0;
    _TotalComputeMsecGPU_D = 0.0;
    _TotalRenderMsec = 0.0;
    _ComputeMode = AUTO_MODE;
    _GPUSupportsDouble = false;
    _ScaleMagByTime = true;
    _RouteX = nullptr;
    _RouteY = nullptr;
    _RouteN = 0;
    _InDrag = false;
    _DragImageX = 0.0;
    _DragImageY = 0.0;
    _Drawing = false;
    SetMemoriesToDefault();
}

void MandelController::Initialise(MandelComputeDevice* ComputeDevice,
           MandelRendererDevice* RendererDevice, MandelRendererView* View,
                 int Nx,int Ny,int Iter,bool Validate,const std::string& DebugLevels)
{
    //  Create the compute handler and renderer, and initialise them.
    
    _View = View;
    _ComputeHandler = new MandelComputeHandler(ComputeDevice);
    _Renderer = new Renderer(RendererDevice);
    if (_ComputeHandler) {
        _ComputeHandler->Initialise(Validate,DebugLevels);
        _ComputeHandler->SetCentre(-0.5,0.0);
        _ComputeHandler->SetMagnification(1.0);
        _ComputeHandler->SetMaxIter(Iter);
        _GPUSupportsDouble = _ComputeHandler->GPUSupportsDouble();
    }
    if (_Renderer) {
        _Renderer->Initialise(DebugLevels);
        _Renderer->SetMaxIter(Iter);
    }
    
    //  Allocate space for the Mandelbrot track vertex and colour arrays.
    
    _RouteX = (double*) malloc(Iter * sizeof(double));
    _RouteY = (double*) malloc(Iter * sizeof(double));
    
    //  Set the default image size and iteration count to the passed values.
    
    _Iter = Iter;
    _BaseNx = Nx;
    _BaseNy = Ny;
    SetImageSize(Nx,Ny);
    SetViewSize(_FrameX,_FrameY);
    printf ("\nPress 'h' key for help.\n");
}

MandelController::~MandelController()
{
    if (_Renderer) delete(_Renderer);
    if (_ComputeHandler) delete(_ComputeHandler);
}

//  Introduce the controller to its request handler.
void MandelController::SetAppContact (MandelAppContact* Contact)
{
    _AppContact = Contact;
    RedisplayTitle();
}

//  Set the image size to be used by the compute handler, and notify the renderer.
void MandelController::SetImageSize(int Nx, int Ny)
{
    if (_ComputeHandler) _ComputeHandler->SetImageSize(Nx,Ny);
    if (_Renderer) _Renderer->SetImageSize(Nx,Ny);
}

//  This specifies the dimensions of the view. The renderer needs to know these.
//  The compute handler does not need to know the actual dimensions, but it does need
//  to know the aspect ratio that will be used to display the image. This will be called
//  whenever the size of the application window, and hence the view, is changed.
void MandelController::SetViewSize(double Width, double Height)
{
    _FrameX = float(Width);
    _FrameY = float(Height);
    if (_ComputeHandler) _ComputeHandler->SetAspect(Width,Height);
    if (_Renderer) _Renderer->SetDrawableSize(float(Width),float(Height));
    _NeedToRedraw = true;
}

//  Called whenever the scroll wheel has been moved.
void MandelController::ScrollWheel (float DeltaX, float DeltaY, float AtX, float AtY)
{
    if (_ComputeHandler && DeltaY != 0.0) {
        double XCoord = 0.0;
        double YCoord = 0.0;
        FrameToImageCoord(AtX,AtY,&XCoord,&YCoord);
        double Magnification = _ComputeHandler->GetMagnification();
        double NewMagnification = Magnification;
        if (DeltaY > 0.0) NewMagnification = Magnification * (1.0 + 0.01 * DeltaY);
        if (DeltaY < 0.0) NewMagnification = Magnification / (1.0 + 0.01 * -DeltaY);
        Magnification = NewMagnification;
        _ComputeHandler->SetMagnification(Magnification);
        RedisplayTitle();
        
        //  With the new magnification, this reworks the calculation done
        //  by frameToImageCoord to work out what new center coordinates
        //  will keep the point currently under the cursor at the same
        //  point in screen coordinates.
        
        double CoordRangeInX = 2.0;
        double FrameWidth = _FrameX;
        double DistFromFrameCentX = FrameWidth * 0.5 - AtX;
        double XCoordFromCent =
             DistFromFrameCentX * CoordRangeInX / (FrameWidth * Magnification);
        double xCent = XCoord + XCoordFromCent;
        double FrameHeight = _FrameY;
        double CoordRangeInY = 2.0 * FrameHeight / FrameWidth;
        double DistFromFrameCentY = FrameHeight * 0.5 - AtY;
        double YCoordFromCent =
             DistFromFrameCentY * CoordRangeInY / (FrameHeight * Magnification);
        double yCent = YCoord + YCoordFromCent;
        _ComputeHandler->SetCentre(xCent,yCent);
        _NeedToRedraw = true;
    }
}

//  Display a reformatted window title to reflect changes to compute mode or magnification.
void MandelController::RedisplayTitle()
{
    if (_ComputeHandler) {
        double Magnification = _ComputeHandler->GetMagnification();
        std::string Title = FormatMagnification(Magnification);
        if (_AppContact) _AppContact->DisplayString(Title.c_str());
    }
}

//  Called whenever the user presses a key on the keyboard.
void MandelController::KeyDown (const char* Key, long Flags, float AtX, float AtY)
{
    //  Some keys (like 'page down') generate flags but a null character pointer.
    
    if (Key == nullptr) return;
    
    if (*Key == 'h') PrintHelp();
    
    if (_ComputeHandler) {
        double Magnification = _ComputeHandler->GetMagnification();
        
        //  Reset back to initial conditions
        
        if (*Key == 'r') {
            Magnification = 1.0;
            _ComputeHandler->SetCentre(-0.5,0.0);
            _ComputeHandler->SetMagnification(Magnification);
            _RouteN = 0;
            _Drawing = false;
            RedisplayTitle();
            _NeedToRedraw = true;
        }
        
        //  Toggle enabling drawing of Mandelbrot path
        
        if (*Key == 'e') {
            if (_Drawing) {
                _Drawing = false;
                _NeedToRedraw = true;
            } else {
                _Drawing = true;
                MouseMoved(AtX,AtY);
            }
        }
        
        //  Clear any overlay, eg any Mandelbrot path
        
        if (*Key == 'x') {
            _RouteN = 0;
            _NeedToRedraw = true;
        }
        
        //  Show current position
        
        if (*Key == 'p') {
            double XCent = 0.0,YCent = 0.0;
            _ComputeHandler->GetCentre(&XCent,&YCent);
            printf ("Xcent %.16g Ycent %.16g, Magnification %.10g\n",XCent,YCent,Magnification);
        }
        
        //  Center the image on the point under the cursor.
        
        if (*Key == 'j') {
            double XCoord = 0.0;
            double YCoord = 0.0;
            FrameToImageCoord(AtX,AtY,&XCoord,&YCoord);
            _ComputeHandler->SetCentre(XCoord,YCoord);
            _NeedToRedraw = true;
        }
        
        //  Handle changes to CPU/GPU mode
        
        ComputeMode OldMode = _ComputeMode;
        if (*Key == 'a') {
            _ComputeMode = AUTO_MODE;
        } else if (*Key == 'c') {
            _ComputeMode = CPU_MODE;
        } else if (*Key == 'g') {
            _ComputeMode = GPU_MODE;
        }
        if (OldMode != _ComputeMode) {
            _NeedToRedraw = true;
            RedisplayTitle();
        }
        
        //  Handle the Zoom modes
        
        if (*Key == 'z') {
            if (_ZoomMode == ZOOM_TIMED) {
                _ZoomMode = ZOOM_NONE;
                float Msec = _ZoomTimer.ElapsedMsec();
                printf ("Zoom mode cancelled, frame rate = %.2f frames/sec\n",
                        float(_ZoomFramesCPU + _ZoomFramesGPU_D + _ZoomFramesGPU) * 1000.0 /Msec);
            } else {
                _ZoomMode = ZOOM_TIMED;
                _ZoomTimer.Restart();
                _ZoomFramesCPU = 0;
                _ZoomFramesGPU = 0;
                _ZoomFramesGPU_D = 0;
                _TotalComputeMsecCPU = 0.0;
                _TotalComputeMsecGPU = 0.0;
                _TotalComputeMsecGPU_D = 0.0;
                _TotalRenderMsec = 0.0;
                _NeedToRedraw = true;
            }
        }
        if (*Key == 'i' || *Key == 'o') {
            if (_ZoomMode == ZOOM_NONE || _ZoomMode == ZOOM_TIMED) {
                _ZoomTimer.Restart();
                _ZoomFramesCPU = 0;
                _ZoomFramesGPU = 0;
                _ZoomFramesGPU_D = 0;
                _TotalComputeMsecCPU = 0.0;
                _TotalComputeMsecGPU = 0.0;
                _TotalComputeMsecGPU_D = 0.0;
                _TotalRenderMsec = 0.0;
                _NeedToRedraw = true;
            }
            if (*Key == 'i') _ZoomMode = ZOOM_IN;
            else _ZoomMode = ZOOM_OUT;
        }
        
        //  Use a memory setting ('0'..'9')
        
        int NKey = *Key - '0';
        if (NKey >= 0 && NKey <= 9) {
            double XCent = _Memories[NKey].XCent;
            double YCent = _Memories[NKey].YCent;
            double Magnification = _Memories[NKey].Magnification;
            _ComputeHandler->SetCentre(XCent,YCent);
            _ComputeHandler->SetMagnification(Magnification);
            RedisplayTitle();
            _NeedToRedraw = true;
        }
        
        //  Allows the image size to be changed, showing the effect on resolution.
        
        if (*Key == 'l') { SetImageSize(_BaseNx * 2,_BaseNy * 2); _NeedToRedraw = true; }
        if (*Key == 'm') { SetImageSize(_BaseNx,_BaseNy); _NeedToRedraw = true; }
        if (*Key == 's') { SetImageSize(_BaseNx / 2,_BaseNy / 2); _NeedToRedraw = true; }
        if (*Key == 't') { SetImageSize(_BaseNx / 4,_BaseNy / 4); _NeedToRedraw = true; }
        
        //  Enable or disable the weighting of the magnification during zoom to compensate
        //  for the time for the computation.
        
        if (*Key == 'w') {
            printf ("Scaling of magnification to compensate for compute delays ");
            _ScaleMagByTime = !_ScaleMagByTime;
            if (_ScaleMagByTime) printf ("enabled\n");
            else printf ("disabled\n");
        }
        
        //  Calculate the track of a Mandelbrot calculation from the point under the cursor.
        
        if (*Key == 'd') {
            double XCoord = 0.0;
            double YCoord = 0.0;
            FrameToImageCoord(AtX,AtY,&XCoord,&YCoord);
            int NPosns = CalcRoute(XCoord,YCoord,_RouteX,_RouteY,_Iter);
            _RouteN = NPosns;
            _NeedToRedraw = true;
        }
    }
}

int MandelController::CalcRoute (double X0,double Y0,double* XPosns,double* YPosns,int MaxIter)
{
    double X = 0.0;
    double Y = 0.0;
    int Iter = 0;
    double Xtmp = 0.0;
    while (((X * X) + (Y * Y) < 4.0) && (Iter < MaxIter)) {
        Xtmp = (X + Y) * (X - Y) + X0;
        Y = (2.0 * X * Y) + Y0;
        X = Xtmp;
        XPosns[Iter] = X;
        YPosns[Iter] = Y;
        Iter += 1;
    }
    return Iter;
}

//  Called whenever the user releases a key on the keyboard.
void MandelController::KeyUp (const char* Key, long Flags, float AtX, float AtY)
{
    if (Key == nullptr) return;
    
    if (_ComputeHandler) {
        
        //  The only key releases we care about are the 'i' and 'o' keys, which
        //  cancel the zoom in/out modes.
        
        if (*Key == 'i' || *Key == 'o') {
            _ZoomMode = ZOOM_NONE;
            float Msec = _ZoomTimer.ElapsedMsec();
            int ZoomFrames = _ZoomFramesGPU + + _ZoomFramesGPU_D + _ZoomFramesCPU;
            printf ("Frame rate = %.2f frames/sec\n",float(ZoomFrames) * 1000.0 /Msec);
            printf ("Average compute time:");
            if (_ZoomFramesGPU > 0) printf (" %.2f msec (GPU)",
                                            _TotalComputeMsecGPU / float(_ZoomFramesGPU));
            if (_ZoomFramesGPU_D > 0) printf (" %.2f msec (GPU-D)",
                                            _TotalComputeMsecGPU_D / float(_ZoomFramesGPU_D));
            if (_ZoomFramesCPU > 0) printf (" %.2f msec (CPU)",
                                            _TotalComputeMsecCPU / float(_ZoomFramesCPU));
            printf ("\n");
            if (ZoomFrames > 0) printf ("Average render time: %.2f msec\n",
                                                    _TotalRenderMsec / float(ZoomFrames));
        }
    }
}

//  Called whenever the mouse moves within the view, if mouse moves are enabled.
void MandelController::MouseMoved (float AtX, float AtY)
{
    double XCoord = 0.0;
    double YCoord = 0.0;
    if (_InDrag) {
        
        //  Calculate the image coordinate difference between the image point below the
        //  cursor and the image point where the drag started. This is the correction to
        //  the image center needed to reposition the image so the cursor is now above
        //  that start point in the image.
        
        FrameToImageCoord(AtX,AtY,&XCoord,&YCoord);
        double XOffset = _DragImageX - XCoord;
        double YOffset = _DragImageY - YCoord;
        double XCent,YCent;
        _ComputeHandler->GetCentre(&XCent,&YCent);
        _ComputeHandler->SetCentre(XCent + XOffset,YCent + YOffset);
        _NeedToRedraw = true;
    }
    if (_Drawing) {
        
        //  Calculate the Mandelbrot path from the mouse position, and set it as
        //  the overlay arrays, _RouteX, _RouteY and _RouteN. (It's important to
        //  call FrameToImageCoord() here rather than just at the start of this routine,
        //  because if we're dragging and drawing at the same time, the change in image
        //  centre will have a minor effect on the calculated position and the path is
        //  very position-sensitive - that's the whole point of the Mandelbrot set!)
        
        FrameToImageCoord(AtX,AtY,&XCoord,&YCoord);
        int NPosns = CalcRoute(XCoord,YCoord,_RouteX,_RouteY,_Iter);
        _RouteN = NPosns;
        _NeedToRedraw = true;
    }
}

//  Called whenever the user releases the mouse click.
void MandelController::MouseUp (float AtX, float AtY)
{
    if (_InDrag) _InDrag = false;
}

//  Called whenever the user clicks down on the mouse with the cursor in the view.
void MandelController::MouseDown (float AtX, float AtY)
{
    if (_ComputeHandler) {
        double XCoord = 0.0;
        double YCoord = 0.0;
        FrameToImageCoord(AtX,AtY,&XCoord,&YCoord);
        _InDrag = true;
        _DragImageX = XCoord;
        _DragImageY = YCoord;
    }
}

void MandelController::SetMemory(
            int Memory, double XCent,double YCent,double Magnification)
{
    if (Memory >= 0 && Memory <= 9) {
        _Memories[Memory].XCent = XCent;
        _Memories[Memory].YCent = YCent;
        _Memories[Memory].Magnification = Magnification;
    }
}

void MandelController::SetMemoriesToDefault()
{
    //  Initially, set all memories to the default showing the whole range of the set.
    
    for (int I = 0;I < 10; I++) SetMemory(I,-0.5,0.0,1.0);
    
    //  Now set some of them (OK, all bar memory 0) to some interesting values - these
    //  are actually just the results of pressing 'p' when I came to images I liked.
    
    SetMemory(1,0.3868518957329334,0.1346382218151437,4638.938418);
    SetMemory(2,-0.7485981681169396,0.1847233013261255,105707.2469);
    SetMemory(3,-0.6523833435215625,0.3575238849957945,5.589402892e+12);
    SetMemory(4,0.2709702586923193,0.00504822194561597,5000.0);
    SetMemory(5,0.4002654933420453,0.1408816530352049,1154.003232);
    SetMemory(6,0.4006417188140499,0.1408379640285069,22623.25281);
    SetMemory(7,-1.39985867565925,0.001279901488190826,1609014.646);
    SetMemory(8,-0.7478413625068855,0.09125909131712467,1138.784602);
    SetMemory(9,0.270925,0.004725,15000.0);
}

//  Invoked when the view needs to be drawn or redrawn. This will be called first as soon
//  as the program starts up and the window and view are created. Then it will be called
//  at whatever frame rate the program has set - usually 60 frames/sec. It is possible to
//  change this, or even to disable this behaviour in the application, in which case this
//  will usually only be called when the window/view are resized.

void MandelController::Draw()
{
    //  Most of the time, there's nothing that needs doing. If anything has changed that
    //  will have an effect on the image (zooming, changing image center, for example),
    //  _NeedToRedraw should have been set.
    
    if (_NeedToRedraw && _Renderer && _ComputeHandler) {
        
        //  Get the compute handler to recompute the image using the current settings,
        //  using either the CPU or GPU. In auto mode, use the GPU unless floating point
        //  rounding error at the current settings will be a problem, in which case use
        //  the CPU.
        
        bool FloatOK = _ComputeHandler->FloatOK();
        UseMode ModeToUse = USE_NONE;
        if (_ComputeMode == AUTO_MODE) {
            if (FloatOK) {
                ModeToUse = USE_GPU;
            } else {
                if (_GPUSupportsDouble) {
                    ModeToUse = USE_GPU_D;
                } else {
                    ModeToUse = USE_CPU;
                }
            }
        } else if (_ComputeMode == CPU_MODE) {
            ModeToUse = USE_CPU;
        } else {
            ModeToUse = USE_GPU;
            if (!FloatOK && _GPUSupportsDouble) {
                ModeToUse = USE_GPU_D;
            }
        }
        MsecTimer ComputeTimer;
        if (ModeToUse == USE_GPU) {
            _ComputeHandler->Compute();
            _TotalComputeMsecGPU += ComputeTimer.ElapsedMsec();
        } else if (ModeToUse == USE_GPU_D) {
            _ComputeHandler->ComputeDouble();
            _TotalComputeMsecGPU_D += ComputeTimer.ElapsedMsec();
        } else if (ModeToUse == USE_CPU) {
            _ComputeHandler->ComputeInC();
            _TotalComputeMsecCPU += ComputeTimer.ElapsedMsec();
        } else {
            printf ("**Internal error, compute mode unspecified**\n");
        }
                
        //  If the mode (CPU/GPU) has changed, we need to show this in the window title.
        //  Note that RedisplayTitle() uses _LastUsedMode so this has to be set before
        //  it is called. Actually, given that something - probably the magnification -
        //  has changed, or we'd not be redrawing at all, we might as well just always
        //  update the title.
        
        _LastUsedMode = ModeToUse;
        RedisplayTitle();

        //  If a Mandelbrot path has been calculated, using either 'e' or 'd' keys, set
        //  this as the overlay for the Renderer. The Renderer works in window frame
        //  coordinates and the path is in Mandelbrot coordinates, and the conversion
        //  between the two changes, probably with each redraw, so has to be redone at
        //  this point.
        
        if (_RouteN > 0) {
            float* XPosns = (float*)malloc(_Iter * sizeof(float));
            float* YPosns = (float*)malloc(_Iter * sizeof(float));
            ImageToFrameCoord(_RouteX,_RouteY,XPosns,YPosns,_RouteN);
            _Renderer->SetOverlay(XPosns,YPosns,_RouteN);
            free (XPosns);
            free (YPosns);
        } else {
            _Renderer->SetOverlay(nullptr,nullptr,0);
        }
        
        //  Get the renderer to draw the new image into the view, getting its address
        //  from the compute handler.
        
        float* ImageData = _ComputeHandler->GetImageData();
        MsecTimer RenderTimer;
        _Renderer->Draw(_View,ImageData);
        _TotalRenderMsec += RenderTimer.ElapsedMsec();
        _NeedToRedraw = false;
        
        //  Handle Zoom mode by changing the magnification for the next frame.
        
        if (_ZoomMode != ZOOM_NONE) {
            double Magnification =_ComputeHandler->GetMagnification();
            float Msec = _ZoomTimer.ElapsedMsec();
            
            //  Work out whether we are zooming in or out (obvious for ZOOM_IN and
            //  ZOOM_OUT, but in ZOOM_TIMED mode we switch after 5 seconds and stop
            //  after 10.
            
            bool IncreaseMag = false;
            bool DecreaseMag = false;
            if (_ZoomMode == ZOOM_TIMED) {
                if (Msec > 10000.0) {
                    printf ("Zoom mode ends, frame rate = %.2f frames/sec\n",
                       float(_ZoomFramesCPU + _ZoomFramesGPU + _ZoomFramesGPU_D) * 1000.0 /Msec);
                    _ZoomMode = ZOOM_NONE;
                } else {
                    if (Msec < 5000.0) IncreaseMag = true;
                    else DecreaseMag = true;
                }
            } else if (_ZoomMode == ZOOM_IN) {
                IncreaseMag = true;
            } else if (_ZoomMode == ZOOM_OUT) {
                DecreaseMag = true;
            }
            
            //  This aims to change the magnification by a factor 2 every second,
            //  allowing for the fact that we may not be rescheduled at the 60 frames
            //  a second rate that we expect - particularly if the CPU is being used.
            //  If we were rescheduled precisely 60 times a second, and each time
            //  Magnification were multipled by a fixed MagFactor, then after one second
            //  Magnification would have increased by a factor given by MagFactor to the
            //  power 60. If we want that to be a factor 2, then we need MagFactor equal
            //  to the 60th root of 2. Which is the default we use here (it's ~1.0116).
            //  In this, (1.0 / 60.0) is the frame time in seconds. If the measured
            //  frame time is different, we compensate by tweaking the MagFactor value.
            //  FrameSecs is the time taken for the previous frame in this Zoom sequence,
            //  so we need to have had at least one frame so far to work this out. (If
            //  we're running at exactly 60fps, FrameSecs will be exactly 1.0/60.0)
            
            double MagFactor = pow(2.0,(1.0 / 60.0));
            int ZoomFrames = _ZoomFramesGPU + _ZoomFramesGPU_D + _ZoomFramesCPU;
            if (ZoomFrames > 0 && _ScaleMagByTime) {
                float FrameSec = (Msec - _LastZoomMsec) * 0.001f;
                MagFactor = pow(2.0,FrameSec);
            }
            if (IncreaseMag) Magnification = Magnification * MagFactor;
            if (DecreaseMag) Magnification = Magnification / MagFactor;
            
            //  Display the new magnification and apply it,  and remember the time so we can
            //  allow for the frame rate next time round.
            
            _ComputeHandler->SetMagnification(Magnification);
            RedisplayTitle();
            if (ModeToUse == USE_GPU) _ZoomFramesGPU++;
            else if (ModeToUse == USE_GPU_D) _ZoomFramesGPU_D++;
            else _ZoomFramesCPU++;
            _LastZoomMsec = Msec;
            _NeedToRedraw = true;
        }
    }
}

//  Format the magnification into a suitable window title.
std::string MandelController::FormatMagnification (double Magnification)
{
    std::string Device;
    std::string Units = "";
    double Value = Magnification;
    if (_LastUsedMode == USE_GPU) {
        if (_ComputeHandler->FloatOK()) {
            Device = "GPU";
        } else {
            Device = "*GPU*";
        }
    } else if (_LastUsedMode == USE_GPU_D) {
        if (_ComputeHandler->DoubleOK()) {
            Device = "GPU-D";
        } else {
            Device = "*GPU-D*";
        }
    } else {
       if (_ComputeHandler->DoubleOK()) {
            Device = "CPU";
        } else {
            Device = "*CPU*";
        }
    }

    if (Magnification > 1.0e15) {
        Value = Magnification/1.0e15;
        Units = "quadrillion";
    } else if (Magnification > 1.0e12) {
        Value = Magnification/1.0e12;
        Units = "trillion";
    } else if (Magnification > 1.0e9) {
        Value = Magnification/1.0e9;
        Units = "billion";
    } else if (Magnification > 1.0e6) {
        Value = Magnification/1.0e6;
        Units = "million";
    } else if (Magnification > 1.0e3) {
        Value = Magnification/1.0e3;
        Units = "thousand";
    }
    char Number[256];
    snprintf (Number,sizeof(Number),"%.3g",Value);
    std::string TitleString = Number;
    TitleString += (" " + Units + " (" + Device + ")");
    return TitleString;
}

//  Convert from view coordinates to Mandelbrot set coordinates.
void MandelController::FrameToImageCoord(float AtX, float AtY,double* XCoord,double* YCoord)
{
    if (_ComputeHandler) {
        double XCent,YCent;
        _ComputeHandler->GetCentre(&XCent,&YCent);
        double Magnification = _ComputeHandler->GetMagnification();
        double CoordRangeInX = 2.0;
        double FrameWidth = _FrameX;
        double DistFromFrameCentX = FrameWidth * 0.5 - AtX;
        double XCoordFromCent =
             DistFromFrameCentX * CoordRangeInX / (FrameWidth * Magnification);
        *XCoord = XCent - XCoordFromCent;
        double FrameHeight = _FrameY;
        double CoordRangeInY = 2.0 * FrameHeight / FrameWidth;
        double DistFromFrameCentY = FrameHeight * 0.5 - AtY;
        double YCoordFromCent =
             DistFromFrameCentY * CoordRangeInY / (FrameHeight * Magnification);
        *YCoord = YCent - YCoordFromCent;
    }

}

//  Convert from Mandelbrot set coordinates to view coordinates.
void MandelController::ImageToFrameCoord(double* XCoord,double* YCoord,float* AtX,float* AtY,int N)
{
    if (_ComputeHandler) {
        double XCent,YCent;
        _ComputeHandler->GetCentre(&XCent,&YCent);
        double Magnification = _ComputeHandler->GetMagnification();
        double FrameWidth = _FrameX;
        double FrameHeight = _FrameY;
        double CoordRangeInX = 2.0;
        double CoordRangeInY = 2.0 * FrameHeight / FrameWidth;
        for (int I = 0; I < N; I++) {
            double XCoordFromCent = XCent - XCoord[I];
            double DistFromFrameCentX =
            (XCoordFromCent * FrameWidth * Magnification) / CoordRangeInX;
            AtX[I] = (FrameWidth * 0.5) - DistFromFrameCentX;
            double YCoordFromCent = YCent - YCoord[I];
            double DistFromFrameCentY =
            (YCoordFromCent * FrameHeight * Magnification) / CoordRangeInY;
            AtY[I] = (FrameHeight * 0.5) - DistFromFrameCentY;
        }
    }
    
}

void MandelController::PrintHelp()
{
    //  The description of pixelation depends on whether or not the GPU has double precision
    //  support. I thought it looked neater to set this in a string first rather than have
    //  conditional code in the main output section.
    
    std::string DoublePrecText;
    if (_GPUSupportsDouble) {
        DoublePrecText =
            "    This GPU supports double precision support floating point and will use it\n"
            "    at magnifications above about 100,000, where single precision floating point\n"
            "    errors would cause pixelation.";
    } else {
        DoublePrecText =
            "    This GPU does not support double precision floating point and at magnifications\n"
            "    above about 100,000, single precision floating point will cause pixelation.";
    }
    
    //  Output what I hope is useful help information.
    
    printf ("\n");
    printf ("This shows the Mandelbrot set.\n");
    printf ("Zooming or moving this image recalculates the set, using either GPU or CPU.'\n");
    printf ("Dragging on the image moves it around in the display.\n");
    printf ("Scrolling up zooms in around the cursor position.\n");
    printf ("Scrolling down zooms out around the cursor position.\n");
    printf ("As you zoom, the window title shows the current magnification level\n");
    printf ("\n");
    printf ("Centre on an interesting point - usually near the edge of the set boundary\n");
    printf ("and keep zooming in. The set boundary continues to get more complicated as you\n");
    printf ("zoom in on it. You may have to recenter the image occasionally\n");
    printf ("\n");
    printf ("Hitting certain keyboard keys has an effect:\n");
    printf ("'0'..'9' select pre-determined settings for centre point and magnification.\n");
    printf ("'r' resets the display to its starting point\n");
    printf ("'i' hold down the 'i' key to zoom in\n");
    printf ("'o' hold down the 'o' key to zoom out\n");
    printf ("'j' centers the image on the cursor position.\n");
    printf ("'p' outputs the current image centre and magnification on the terminal.\n");
    printf ("'d' displays the Mandelbrot path for the point under the cursor.\n");
    printf ("'e' toggles a continuous display of the Mandelbrot path as the cursor moves.\n");
    printf ("'x' clears any Mandelbrot path from the display\n");
    printf ("'z' does a zoom test. It zooms in for 5 seconds, then out for 5 seconds\n");
    printf ("'a' sets auto mode - the program uses the GPU so long as its floating point\n");
    printf ("    support is accurate enough at the current magnification.\n");
    printf ("'c' forces the program to use the CPU - all available cores.\n");
    printf ("'g' forces the program to use the GPU at all magnifications.\n");
    printf ("%s\n",DoublePrecText.c_str());
    printf ("    (Above about 100 trillion even double precision has problems.)\n");
    printf ("'w' toggles magnification rate compensation for slow compute times during zoom\n");
    printf ("'l' sets size of images to %d by %d (large)\n",_BaseNx * 2,_BaseNy * 2);
    printf ("'m' sets size of images to %d by %d (medium - default)\n",_BaseNx,_BaseNy);
    printf ("'s' sets size of images to %d by %d (small)\n",_BaseNx / 2,_BaseNy / 2);
    printf ("'t' sets size of images to %d by %d (tiny)\n",_BaseNx / 4,_BaseNy / 4);
}

/*
                            P r o g r a m m i n g   N o t e s
 
    o   There are occasional times near the boundary where single precision floating point
        isn't accurate enough, when I've seen the window title briefly display (*GPU*)
        indicating that it's using the GPU but shouldn't be. Since it shouldn't be, this
        seems to be a bug. I think it's connected with just when RedisplayTitle() gets
        called, but haven't worked out why yet.
 
    o   When expanding the window, especially on Linux, I've seen a black band at the right
        and bottom of the window even once the window size has stopped changing, as though
        the program hasn't redisplayed at the final window size. As soon as it redisplays
        - eg as a result of a zoom in or out - this fixes itself up. (Added later: this
        seems to have been a problem with the Vulkan Framework not always recreating the
        swap chain to reflect the final size of the window, and should now be fixed.)
 
    o   The number of keyboard options is getting silly, and I'm running out of sensible
        letters. But I really don't want to get into the complexity of a proper GUI, not in
        what's supposed to be a simple demonstration of GPU programming.
 */
