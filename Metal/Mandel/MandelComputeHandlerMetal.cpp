//  ------------------------------------------------------------------------------------------------

//                    M a n d e l  C o m p u t e  H a n d l e r  M e t a l. c p p
//
//  A MandelComputeHandler exists to compute an image of the Mandelbrot set, with a specified
//  centre point and a specified magnification, using a specified maximum number of iterations
//  for each point in the image. The intention behind this code was to show an example of
//  how such a calculation can be done using a GPU, and to allow a comparision of the results
//  with doing the same calculation using the CPU. This is written in C++, and uses Apple's
//  metal-cpp to call Metal GPU routines directly. This is a deliberately low-level approach.
//  Note that this code is Apple-specific.
//
//  For more details, see the extended comments at the start of MandelComputeHandlerMetal.h
//
//  History:
//     22 Jul 2023. First version after repackaging of the original testbed code. KS.
//     25 Sep 2023. Corrected RecalculateArgs() to allow for use of a non-square data array,
//                  ie where _nx != _ny. Also should now release any existing image buffer when
//                  the image size is changed. KS.
//      2 Dec 2023. Minor commenting changes to Compute(). KS.
//     26 Feb 2024. Rationalising slightly the Metal and Vulkan versions, this has been renamed
//                  from MandelComputeHandler.cpp to MandelComputeHandlerMetal.cpp. KS.
//     18 Jun 2024. Added Initialise(), ComputeDouble() and GPUSupportsDouble(), to be consistent
//                  with the Vulkan version. Initialise() is also a first step to support for use
//                  of a DebugHandler. KS.

#include "MandelComputeHandlerMetal.h"

#include <thread>

using NS::StringEncoding::UTF8StringEncoding;

//  _debugOptions is the comma-separated list of all the diagnostic levels that the built-in debug
//  handler recognises. If a call to _debug.Log() or .Logf() is added with a new level name, this
//  new name must be added to this string.

const std::string MandelComputeHandler::_debugOptions = "Setup,Timing";

//  ------------------------------------------------------------------------------------------------

//                              M a n d e l  C o m p u t e  H a n d l e r

MandelComputeHandler::MandelComputeHandler(MTL::Device* Device )
    : _device( Device->retain() )
{
    _xCent = 0.0;
    _yCent = 0.0;;
    _magnification = 1.0;
    _width = 512.0;
    _height = 512.0;
    _nx = 0;
    _ny = 0;
    _maxiter = 1024;
    _outputBuffer = nullptr;
    _imageData = nullptr;
    _mandelFunction = nullptr;
    _commandQueue = nullptr;
    _debug.SetSubSystem("Compute");
    _debug.LevelsList(_debugOptions);
}

MandelComputeHandler::~MandelComputeHandler()
{
    if (_commandQueue) _commandQueue->release();
    if (_device) _device->release();
    if (_mandelFunction) _mandelFunction->release();
    if (_outputBuffer) _outputBuffer->release();
    _commandQueue = nullptr;
    _device = nullptr;;
    _mandelFunction = nullptr;;
    _outputBuffer = nullptr;;
    _imageData = nullptr;

    //  _gridSize and _threadGroupDims are not pointers, and don't need to be released.
    //  _image data is a pointer to the data underlying _outputBuffer and so is released
    //  along with _outputBuffer.
}

void MandelComputeHandler::Initialise(bool /*Validate*/,const std::string& DebugLevels)
{
    //  The Validate argument is there for the Vulkan version, which uses it to switch
    //  in additional diagnostics. It is not relevant for this Metal version. Note that
    //  the GPU device is passed to the compute handler already selected.
    
    _debug.SetLevels(DebugLevels);
    _debug.Log("Setup","Initialising Metal for compute.");
    RecomputeArgs();
    MsecTimer SetupTimer;
    _commandQueue = _device->newCommandQueue();
    _debug.Logf("Setup","GPU command queue created at %.3f msec",SetupTimer.ElapsedMsec());
    BuildComputeShader();
}

bool MandelComputeHandler::GPUSupportsDouble()
{
    //  Metal GPUs don't support double precision (at least as of June 2024).
    return false;
}

void MandelComputeHandler::BuildComputeShader()
{
    NS::Error* pError = nullptr;
    MsecTimer SetupTimer;

    MTL::Library* library = _device->newLibrary(
                            NS::String::string("compute.metallib",UTF8StringEncoding),&pError);
    if (library == nullptr || pError != nullptr) {
        printf ("Error opening library 'Compute.metallib'.\n");
        if (pError) {
            printf ("Reason: %s\n",pError->localizedDescription()->cString(UTF8StringEncoding));
        }
    } else {
        _debug.Logf("Setup","GPU library created at %.3f msec",SetupTimer.ElapsedMsec());
        _mandelFunction = _device->newComputePipelineState(
                   library->newFunction(NS::String::string("mandel",UTF8StringEncoding)),&pError);
        if (_mandelFunction == nullptr || pError != nullptr) {
            printf ("Unable to find 'mandel' function in library\n");
            if (pError) {
                printf ("Reason: %s\n",pError->localizedDescription()->cString(UTF8StringEncoding));
            }
        } else {
            _debug.Logf("Setup","GPU mandel function created at %.3f msec",
                        SetupTimer.ElapsedMsec());
        }
    }

    if (library) library->release();
}

void MandelComputeHandler::SetImageSize(int Nx,int Ny)
{
    //  If the image size changes, the grid size and thread group sizes have to change
    //  to match it. So does the buffer used by both the GPU and CPU code.
    
    if (_nx != Nx || _ny != Ny) {
        
        _debug.Logf("Setup","Rebuilding image buffer to %d by %d.",Nx,Ny);
        MsecTimer theTimer;

        //  Release any existing buffer. (I believe this is what's required - indeed, it may
        //  be that either of these calls will the do the job.)
        
        if (_outputBuffer) {
            _outputBuffer->setPurgeableState(MTL::PurgeableStateEmpty);
            _outputBuffer->release();
        }
        
        //  We create a Metal buffer for the computed images. This will be used by the GPU
        //  pipeline code (see the code for Compute()) and is made shared so that it can
        //  also be accessed by the CPU (getting its memory address in _imageData using the
        //  contents() call shown below). This allows this shared memory address to be passed
        //  to the Renderer for display, and can also be used by the CPU code to calculate
        //  the image in the same memory. This simplifies things, as both the GPU and the
        //  CPU create an image at the address given by _imageData. (See programming notes
        //  for a little more discussion of buffer allocation.)
        
        int length = Nx * Ny * sizeof(float);
        unsigned int alignment = sysconf(_SC_PAGE_SIZE);
        int allocationSize = (length + alignment - 1) & (~(alignment - 1));
        uint bufferOptions = MTL::StorageModeShared;
        _outputBuffer = _device->newBuffer(allocationSize,bufferOptions);
        _imageData = (float*)_outputBuffer->contents();
        _debug.Logf("Timing","Resized image buffer at %.2f msec",theTimer.ElapsedMsec());

        //  Tweaking the arrangement of GPU threads and thread groups can be tricky, but this
        //  follows the general advice given in the Apple documentation. The pipeline set up
        //  by Draw() will use _gridSize and _threadGroupSize.
        
        int threadGroupSize = _mandelFunction->maxTotalThreadsPerThreadgroup();
        int threadWidth = _mandelFunction->threadExecutionWidth();
        if (threadGroupSize > (Nx * Ny)) threadGroupSize = Nx * Ny;
        _gridSize = MTL::Size(Nx,Ny,1);
        _threadGroupDims = MTL::Size(threadGroupSize / threadWidth,threadWidth,1);

        _nx = Nx;
        _ny = Ny;
        _debug.Log("Setup","Image buffer resized and mapped.");
    }
}

void MandelComputeHandler::SetCentre(double XCent,double YCent)
{
    _xCent = XCent;
    _yCent = YCent;
}

void MandelComputeHandler::SetMagnification(double Magnification)
{
    _magnification = Magnification;
    RecomputeArgs();
}

void MandelComputeHandler::SetAspect(double Width, double Height)
{
    _height = Height;
    _width = Width;
    RecomputeArgs();
}

void MandelComputeHandler::SetMaxIter(int MaxIter)
{
    _maxiter = MaxIter;
    RecomputeArgs();
}

double MandelComputeHandler::GetMagnification(void)
{
    return _magnification;
}

void MandelComputeHandler::GetCentre(double* Xcent,double* Ycent)
{
    *Xcent = _xCent;
    *Ycent = _yCent;
}

float* MandelComputeHandler::GetImageData()
{
    return _imageData;
}

//  GetDebugOptions() returns the comma-separated list of the various diagnostic levels supported
//  by the compute handler. Note that this is a static routine; it can be convenient for a program
//  to have this list available before the compuet handler is constructed, and it is in any case a
//  fixed list.

std::string MandelComputeHandler::GetDebugOptions(void)
{
    return _debugOptions;
}

void MandelComputeHandler::RecomputeArgs()
{
    //  This may seem to be an odd place to take this into account, but it was actually easiest
    //  to do this here. The calculation takes into account the aspect ratio of the way the image
    //  will be displayed, and corrects accordingly, stretching the image scale as needed. And
    //  of course it has to take into account the aspect ratio of the data array it is using for
    //  the image. Of course, the cleanest thing would be to change the aspect ratio of the array
    //  to match changes in the display, but that would mean reallocating the array each time the
    //  display changes (which happens a lot if a window is being resized, say).
    
    double aspect = ((double(_height)/double(_width)) * (double(_nx)/double(_ny)));
    
    //  A change in image center is just passed directly to the computation (GPU or CPU). The main
    //  thing we have to do is calculate the effects of the magnification on the scale values in
    //  X and Y (_dx and _dy), allowing for the overall aspect ratio.
    
    double xRange = 2.0 / _magnification;
    double yRange = aspect * xRange * double(_ny) / double(_nx);
    _dx = xRange/double(_nx);
    _dy = yRange/double(_ny);
    _currentArgs = {float(_xCent),float(_yCent),float(_dx),float(_dy),_maxiter};
}

void MandelComputeHandler::Compute ()
{
    RecomputeArgs();
    
    //  This sets up the command buffer that controls for the GPU calculation.
        
    //  It's good practice to use a separate autorelease pool for separate sections like this.
    //  This way, things get tidied up at the end of this routine.

    NS::AutoreleasePool* pipeAutoreleasePool = NS::AutoreleasePool::alloc()->init();

    //  Create a command buffer, a command encoder, and set the compute shader kernel
    //  that will perform the calculation.

    //MsecTimer TheTimer;
    MTL::CommandBuffer* commandBuffer = _commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(_mandelFunction);

    //  Set the two data buffers, the actual 2D array to be written into by the compute
    //  kernel, and the structure giving the parameters for the calculation.

    encoder->setBuffer(_outputBuffer,0,1);
    encoder->setBytes(&_currentArgs,sizeof(MandelArgs),2);

    // Submit the command buffer for execution and wait for it to complete.

    encoder->dispatchThreads(_gridSize,_threadGroupDims);
    encoder->endEncoding();
    //printf ("Command buffer encoding took %f msec\n",TheTimer.ElapsedMsec());
    //TheTimer.Restart();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    //printf ("Command buffer commit and execution took %f msec\n",TheTimer.ElapsedMsec());

    //  Tidy up
    
    pipeAutoreleasePool->release();
}

void MandelComputeHandler::ComputeDouble ()
{
    //  This is provided for compatability with the Vulkan version, but the controller should
    //  never call it (since GPUSupportsDouble() always returns false for Metal GPUs.)
}

void MandelComputeHandler::ComputeInC ()
{
    RecomputeArgs();
    
    ComputeInCThreads (_imageData,_nx,_ny,_xCent,_yCent,_dx,_dy,_maxiter);
}

void MandelComputeHandler::ComputeInCThreads (
        float* Data,int Nx,int Ny,prec Xcent,prec Ycent,
                                       prec Dx,prec Dy,int MaxIter)
{
    int nThreads = std::thread::hardware_concurrency();
    if (nThreads <= 0) nThreads = 1;

    std::thread threads[nThreads];
    int iY = 0;
    int iYinc = Ny / nThreads;
    for (int iThread = 0; iThread < nThreads; iThread++) {
        threads[iThread] = std::thread (ComputeRangeInC,Data,Nx,Ny,iY,iY+iYinc,
                                                              Xcent,Ycent,Dx,Dy,MaxIter);
        iY += iYinc;
    }
    for (int iThread = 0; iThread < nThreads; iThread++) {
        threads[iThread].join();
    }
    if (iY < Ny) {
        ComputeRangeInC(Data,Nx,Ny,iY,Ny,Xcent,Ycent,Dx,Dy,MaxIter);
    }

    
}

void MandelComputeHandler::ComputeRangeInC (
       float* Data,int Nx,int Ny,int Iyst,int Iyen,prec Xcent,prec Ycent,
                                             prec Dx,prec Dy,int MaxIter)
{
    Data += Iyst * Nx;
    prec gridXcent = Nx * 0.5;
    prec gridYcent = Ny * 0.5;
    for (int Iy = Iyst; Iy < Iyen; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            // Scale
            prec x0 = Xcent + (prec(Ix) - gridXcent) * Dx;
            prec y0 = Ycent + (prec(Iy) - gridYcent) * Dy;

            // Implement Mandelbrot set
           
            prec x = 0.0;
            prec y = 0.0;
            int iteration = 0;
            prec xtmp = 0.0;
            while (((x * x) + (y * y) <= 4.0) && (iteration < MaxIter))
            {
                xtmp = (x + y) * (x - y) + x0;
                y = (2.0 * x * y) + y0;
                x = xtmp;
                iteration += 1;
            }

            // Treat iteration result as a colour value for the image.
            float colour = (float)(iteration);
            if (iteration == MaxIter) colour = 0.0;
            *Data++ = colour;
       }
    }

}

bool MandelComputeHandler::FloatOKatXY(int Ix,int Iy)
{
    //  This checks whether floating point rounding error will not show up at a given
    //  point in the image (given by its pixel coordinates). It calculates the position
    //  in the Mandelbrot set for this pixel given the current X and Y center and the
    //  magnification, then offsets this by roughly the equivalent in image pixel
    //  coordinates of a pixel in the display and recalculates the position. If floating
    //  point precision is such that these two positions appear the same due to rounding
    //  error, then it returns true. Strictly, there still will be rounding error, but
    //  it probably won't be visible with the current display settings.
    
    float Xinc = float(_nx) / float(_width);
    float X0 = _xCent + (float(Ix) - _nx * 0.5) * _dx;
    float X1 = _xCent + (float(Ix) + Xinc - _nx * 0.5) * _dx;
    float Yinc = float(_ny) / float(_height);
    float Y0 = _yCent + (float(Iy) - _ny * 0.5) * _dy;
    float Y1 = _yCent + (float(Iy) + Yinc - _ny * 0.5) * _dy;
    //if (((Y1 - Y0) <= 0.0f) || ((X1 - X0) <= 0.0f)) printf ("%g %g\n",Y1 - Y0,X1 - X0);
    return ((Y1 - Y0) > 0.0f) && ((X1 - X0) > 0.0f);
}

bool MandelComputeHandler::FloatOK(void)
{
    //  This runs a quick test for floating point accuracy at the current display settings
    //  at a set of points that cover the range of the image in both X and Y.
    
    int Ix = 0;
    int Iy = 0;
    int Ixinc = _nx / 10;
    int Iyinc = _ny / 10;
    for (int I = 0; I < 10; I++) {
        if (!FloatOKatXY(Ix,Iy)) return false;
        Ix += Ixinc;
        Iy += Iyinc;
    }
    return true;
}

bool MandelComputeHandler::DoubleOKatXY(int Ix,int Iy)
{
    //  The same as FloatOKatXY(), but using double precision.
    
    double Xinc = double(_nx) / double(_width);
    double X0 = _xCent + (double(Ix) - _nx * 0.5) * _dx;
    double X1 = _xCent + (double(Ix) + Xinc - _nx * 0.5) * _dx;
    double Yinc = double(_ny) / double(_height);
    double Y0 = _yCent + (double(Iy) - _ny * 0.5) * _dy;
    double Y1 = _yCent + (double(Iy) + Yinc - _ny * 0.5) * _dy;
    //if (((Y1 - Y0) <= 0.0) || ((X1 - X0) <= 0.0)) printf ("%g %g\n",Y1 - Y0,X1 - X0);
    return ((Y1 - Y0) > 0.0) && ((X1 - X0) > 0.0);
}

bool MandelComputeHandler::DoubleOK(void)
{
    //  The same as FloatOK() but uses double precision.
    
    int Ix = 0;
    int Iy = 0;
    int Ixinc = _nx / 10;
    int Iyinc = _ny / 10;
    for (int I = 0; I < 10; I++) {
        if (!DoubleOKatXY(Ix,Iy)) return false;
        Ix += Ixinc;
        Iy += Iyinc;
    }
    return true;
}

/*
                                   P r o g r a m m i n g  N o t e s
 
 o  To expand on the comments in SetImageSize() about buffer allocation. The shared memory access
    may have overheads, particularly on machines that don't use unified memory, but this should be
    negligible compared to the rest of the calculations. The device::newBuffer() call used to
    create the buffer is mapped by metal-cpp to the Obj-C newBufferWithLength method of MTLDevice.
    If we wanted to allocate a buffer in this code using malloc() or new[], we could create a
    new Metal buffer using MTLDevice's newBufferWithBytesNoCopy method (which creates a Metal
    buffer that wraps an existing page-aligned memory array), for which metal-cpp also provides
    a wrapper (a variant of the newBuffer() call that supplies the existing memory address and
    an optional deallocator for the memory).
 
 o  When SetImageSize() changes the buffer size, I need to release any memory used by an existing
    buffer. This code does not use automatic reference counting, so the code in SetImageSize() sets
    the buffer state to empty and then releases it. This is what is suggested in:
    https://stackoverflow.com/questions/39158302/how-to-deallocate-a-mtlbuffer-and-mtltexture
    Ideally, I'd do some testing to see if there are any memory leaks in the program, due to this
    or anything else, but I've not done so yet. I have watched its memory usage in ActivityMonitor
    and haven't seen any indication of problems.
 
 o  Apple's documentation about setting threadgroup and grid size is at https://developer.apple.com
                        /documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes
 
 */
