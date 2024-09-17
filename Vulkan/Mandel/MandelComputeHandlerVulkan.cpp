//  ------------------------------------------------------------------------------------------------

//                  M a n d e l  C o m p u t e  H a n d l e r  V u l k a n . c p p
//
//  A MandelComputeHandler exists to compute an image of the Mandelbrot set, with a specified
//  centre point and a specified magnification, using a specified maximum number of iterations
//  for each point in the image. The intention behind this code was to show an example of
//  how such a calculation can be done using a GPU, and to allow a comparision of the results
//  with doing the same calculation using the CPU. This is written in C++, and uses Vulkan.
//  This is a deliberately low-level approach.
//
//  For more details, see the extended comments at the start of MandelComputeHandler.h
//
//  History:
//     26 Sep 2023. First Vulkan version, based on the Metal version. KS.
//      9 Nov 2023. Some tidying of comments. KS.
//     18 Nov 2023. Mandel.spv now found in Shaders/Mandel.spv, not in the default directory. KS.
//     19 Jan 2024. No longer includes the Apple-specific simd.h KS.
//     30 Jan 2024. Now supports the use of a staged buffer for the computed data, as it
//                  includes the necessary SyncBuffer() calls. Tested using both staged and
//                  shared buffers. KS.
//     23 Feb 2024. Minor fixes to get a clean compilation under Windows. KS.
//     14 Mar 2024. Added support for double precision on the GPU, with ComputeDouble()
//                  and GPUSupportsDouble(). KS.
//     12 Jun 2024. Moved Vulkan initialisation from constructor into new Initialise() routine.
//                  Constructor can now accept a null Framework argument (in which case a new
//                  Framework is created locally) or an already initialised Framework, assumed
//                  to have already created a logical device to use. KS.
//     23 Aug 2024. Added additional setup diagnostics. Changed names of shader files. KS.
//     27 Aug 2024. Added Validate parameter to Initialise(). KS.
//      7 Sep 2024. Moved shaders back into default directory, to match the Metal version. KS.
//     14 Sep 2024. Modified following renaming of Framework routines and types. KS.

#include "MandelComputeHandlerVulkan.h"

#include <thread>

//  These have to match the values used by the GPU shader code.

static const uint32_t C_WorkGroupSize = 32;
static const int C_StorageBufferBinding = 0;
static const int C_UniformBufferBinding = 1;

//  _debugOptions is the comma-separated list of all the diagnostic levels that the built-in debug
//  handler recognises. If a call to _debug.Log() or .Logf() is added with a new level name, this
//  new name must be added to this string.

const std::string MandelComputeHandler::_debugOptions = "Setup,Timing";

//  ------------------------------------------------------------------------------------------------

//                              M a n d e l  C o m p u t e  H a n d l e r

MandelComputeHandler::MandelComputeHandler(void* Framework)
{
    _statusOK = true;
    _uniformBufferHndl = KVVulkanFramework::KV_NULL_HANDLE;
    _imageBufferHndl = KVVulkanFramework::KV_NULL_HANDLE;
    _uniformBufferAddr = nullptr;
    _xCent = 0.0;
    _yCent = 0.0;;
    _magnification = 1.0;
    _width = 512.0;
    _height = 512.0;
    _nx = 0;
    _ny = 0;
    _dx = 2.0 / 1024.0;
    _dy = 2.0 / 1024.0;
    _maxiter = 1024;
    _imageData = nullptr;
    _computeQueue = VK_NULL_HANDLE;
    _commandPool = VK_NULL_HANDLE;
    _commandBuffer = VK_NULL_HANDLE;
    _setLayout = VK_NULL_HANDLE;
    _descriptorPool = VK_NULL_HANDLE;
    _descriptorSet = VK_NULL_HANDLE;
    _computePipelineLayout = VK_NULL_HANDLE;
    _computePipeline = VK_NULL_HANDLE;
    _doubleSupportInGPU = false;
    _computePipelineLayoutD = VK_NULL_HANDLE;
    _computePipelineD = VK_NULL_HANDLE;
    _workGroupCounts[0] = _workGroupCounts[1] = _workGroupCounts[2] = 0;
    _frameworkIsLocal = false;
    _debug.SetSubSystem("Compute");
    _debug.LevelsList("Setup,Timing");
    _vulkanFramework = (KVVulkanFramework*)Framework;
}

MandelComputeHandler::~MandelComputeHandler()
{
    _imageData = nullptr;
    if (_frameworkIsLocal && _vulkanFramework) delete _vulkanFramework;
}

void MandelComputeHandler::Initialise(bool Validate,const std::string& DebugLevels)
{
    _debug.SetLevels(DebugLevels);
    if (_vulkanFramework == nullptr) {
        _debug.Log("Setup","Initialising new Vulkan Framework.");
        _vulkanFramework = new(KVVulkanFramework);
        _vulkanFramework->SetDebugSystemName("VulkanCompute");
        _vulkanFramework->EnableValidation(Validate);
        _vulkanFramework->SetDebugLevels(DebugLevels);
        _vulkanFramework->CreateVulkanInstance(_statusOK);
        _vulkanFramework->FindSuitableDevice(_statusOK);
        _vulkanFramework->CreateLogicalDevice(_statusOK);
        _frameworkIsLocal = true;
    }
    InitialiseVulkanItems();
    RecomputeArgs();
}

void MandelComputeHandler::InitialiseVulkanItems()
{
    //  With the physical device selected, we can see if it supports double precision.
    //  (Many devices don't, and even those that do run much slower in double than they do
    //  in float, unlike most modern CPUs, but we want to be able to try it if it's available. 
    
    _doubleSupportInGPU = _vulkanFramework->DeviceSupportsDouble();
    if (_doubleSupportInGPU) _debug.Log("Setup","Device supports double precision.");

    //  And now that we have a Vulkan logical device, we can create all the things we're going to
    //  need that don't depend on knowing the image size to use - which we will eventually
    //  be told by a call to SetImageSize().
    
    //  We can create the uniform buffer used for the arguments here. It never changes - although
    //  its contents do.
    
    _debug.Log("Setup","Setting up uniform buffer for compute arguments.");
    long uniformSizeInBytes = sizeof(MandelArgs);
    _uniformBufferHndl = _vulkanFramework->SetBufferDetails(
                                        C_UniformBufferBinding,"UNIFORM","SHARED",_statusOK);
    _vulkanFramework->CreateBuffer(_uniformBufferHndl,uniformSizeInBytes,_statusOK);
    long bytes;
    _debug.Log("Setup","Mapping and initialising uniform buffer.");
    _uniformBufferAddr = _vulkanFramework->MapBuffer(_uniformBufferHndl,&bytes,_statusOK);
    if (_statusOK && _uniformBufferAddr) memcpy(_uniformBufferAddr,&_currentArgs,bytes);
    
    //  We can even set up the buffer description that will be used to create the main data
    //  buffer, although for the moment we don't actually create the buffer. (To experiment
    //  with a staged buffer, change 'SHARED' to 'STAGED_GPU' That's all that's needed.)

    _debug.Log("Setup","Setting up buffer to store resulting image.");
    _imageBufferHndl = _vulkanFramework->SetBufferDetails(
                                        C_StorageBufferBinding,"STORAGE","SHARED",_statusOK);
       
    //  Given the handles to those two buffer descriptions, we can specify the layout of the
    //  descriptor set that will be needed to describe them to the GPU shader.
    
    std::vector<KVVulkanFramework::KVBufferHandle> handles;
    handles.push_back(_imageBufferHndl);
    handles.push_back(_uniformBufferHndl);
    _vulkanFramework->CreateVulkanDescriptorSetLayout(handles,&_setLayout,_statusOK);
    
     //  We also create a pool that can supply such descriptor sets.
    
    _vulkanFramework->CreateVulkanDescriptorPool(handles,1,&_descriptorPool,_statusOK);

    //  And get that pool to supply the descriptor set that we'll use for the pipeline.
    
    _vulkanFramework->AllocateVulkanDescriptorSet(_setLayout,_descriptorPool,&_descriptorSet,
                                                                                    _statusOK);
    _debug.Log("Setup","Buffers and descriptors set up.");

    //  And, given the set layout, we can specify the layout of the compute pipeline that will
    //  run the shader, and we can create it.
    
    _vulkanFramework->CreateComputePipeline("MandelComp.spv","main",
                       &_setLayout,&_computePipelineLayout,&_computePipeline,_statusOK);
    _debug.Log("Setup","Single precision pipeline created using MandelComp.spv.");

    
    //  If the GPU supports double precision, set up an alternative pipeline that can use it.
    
    if (_doubleSupportInGPU) {    
        _vulkanFramework->CreateComputePipeline("MandelDComp.spv","main",
                       &_setLayout,&_computePipelineLayoutD,&_computePipelineD,_statusOK);
        _debug.Log("Setup","Double precision pipeline created using MandelDComp.spv.");
    }
       
   //  We can also create the one compute queue we will need
    
    _vulkanFramework->GetDeviceQueue(&_computeQueue,_statusOK);
    
    //  And the command pool and command buffer
    
    _vulkanFramework->CreateCommandPool(&_commandPool,_statusOK);
    _vulkanFramework->CreateComputeCommandBuffer(_commandPool,&_commandBuffer,_statusOK);
    _debug.Log("Setup","Command queue and command buffer created.");

 
    //  And that's all we can do to set things up until we're told the size of the image
    //  through a call to SetImageSize(). At that point we can set up the image data buffer
    //  and then set up the descriptor set we've just created, associating its entries with
    //  the actual buffers we're going to use. And then we can run the pipeline when asked to.
    
    _debug.Log("Setup","Initial Vulkan setup completed.");
}

void MandelComputeHandler::SetImageSize(int Nx,int Ny)
{
    //  We need a Vulkan buffer for the computed images. This will be used by the GPU
    //  pipeline code (see the code for Compute()) and is made shared so that it can
    //  also be accessed by the CPU (getting its memory address in _imageData using the
    //  MapBuffer() call shown below). This allows this shared memory address to be passed
    //  to the Renderer for display, and can also be used by the CPU code to calculate
    //  the image in the same memory. This simplifies things, as both the GPU and the
    //  CPU create an image at the address given by _imageData. (See programming notes
    //  for a little more discussion of buffer allocation.)
    
    //  We also need a uniform buffer to pass the parameters for the computation to the
    //  GPU. This is created when this compute handler is itself created, and its size
    //  (but not its contents) never needs to change. However, the image size can change.
    
    //  If the image size changes, the image buffer has to change to match it. So do the the
    //  number of work groups needed to process it. So this routine will be called at the start
    //  of the program to create the buffer in the first place, and also if the program wants
    //  to change the size of the computed images - reducing them for speed, or increasing
    //  them for resolution. The current size of the buffer is held in _nx,_ny, and these are
    //  set to zero at the start of the program. So the initial call to set up the buffer in
    //  the first place can be treated as a resize request. Obviously, we do nothing if the
    //  image size hasn't changed at all.
        
    if (_nx != Nx || _ny != Ny) {
        
        _debug.Logf("Setup","Rebuilding image buffer to %d by %d.",Nx,Ny);
        MsecTimer theTimer;
        
        //  First we need to deal with the image buffer.
        
        //  Note that the first time through this code, we will have a handle for the buffer
        //  (because we needed to set its details so the descriptor sets could describe it for
        //  the pipeline) but the underlying Vulkan buffer and its associated memory won't have been
        //  created yet, in which case we really should create it rather than resize it. However,
        //  ResizeBuffer() can handle the case where the Vulkan buffer doesn't exist yet, so we
        //  can get away with treating all cases as a rezise.
                    
        int sizeInBytes = Nx * Ny * sizeof(float);
        _vulkanFramework->ResizeBuffer(_imageBufferHndl,sizeInBytes,_statusOK);

        _debug.Logf("Timing","Resized image buffer at %.2f msec",theTimer.ElapsedMsec());

        long bytes;
        _imageData = (float*)_vulkanFramework->MapBuffer(_imageBufferHndl,&bytes,_statusOK);
        
        //  Now that we have created both our buffers - the uniform buffer used for the arguments
        //  and the storage buffer used for the image data, we finally set their details into
        //  the descriptor set that's already been created and associated with the pipeline (all
        //  this was done in InitialiseVulkan()).
            
        std::vector<KVVulkanFramework::KVBufferHandle> bufferHandles;
        bufferHandles.push_back(_imageBufferHndl);
        bufferHandles.push_back(_uniformBufferHndl);
        _vulkanFramework->SetupVulkanDescriptorSet(bufferHandles,_descriptorSet,_statusOK);
               
        //  Tweaking the arrangement of GPU threads and thread groups can be tricky, but if
        //  we assume the GPU shader code has set up a hard-coded local workgroup size of
        //  {C_WorkGroupSize,C_WorkGroupSize,1}, then the following values for _workGroupCounts
        //  cover the whole image (with some possible spillover at the edges that the shader
        //  code has to allow for). The pipeline set up by Compute() will use _workGroupCounts.
        
        _workGroupCounts[0] = (uint32_t(Nx) + C_WorkGroupSize - 1)/C_WorkGroupSize;
        _workGroupCounts[1] = (uint32_t(Ny) + C_WorkGroupSize - 1)/C_WorkGroupSize;
        _workGroupCounts[2] = 1;
        
        //  And the pipeline is now ready to go. It will get used in the Compute() routine.

        _nx = Nx;
        _ny = Ny;
        
        _debug.Log("Setup","Image buffer resized and mapped.");

                
    }
}

bool MandelComputeHandler::GPUSupportsDouble()
{
    return _doubleSupportInGPU;
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
    _currentArgs = {float(_xCent),float(_yCent),float(_dx),float(_dy),_maxiter,_nx,_ny,0,
                                                                       _xCent,_yCent,_dx,_dy};
}

void MandelComputeHandler::Compute ()
{
    //printf ("Compute called\n");
    //printf ("%p %d %d %f %f %f %f %d\n",_imageData,_nx,_ny,_xCent,_yCent,_dx,_dy,_maxiter);
        
    RecomputeArgs();
    if (_statusOK && _uniformBufferAddr) memcpy(_uniformBufferAddr,&_currentArgs,
                                                                           sizeof(MandelArgs));

    //  This sets up the pipeline for the GPU calculation and runs it.
        
    _vulkanFramework->RecordComputeCommandBuffer(_commandBuffer,_computePipeline,
                        _computePipelineLayout,&_descriptorSet,_workGroupCounts,_statusOK);
    _vulkanFramework->RunCommandBuffer(_computeQueue,_commandBuffer,_statusOK);
    
    //  If a staged buffer is being used, it needs to be synched at this point. If a non-staged
    //  buffer is being used, this call is simply a null operation, so it can be left in anyway.
    
    _vulkanFramework->SyncBuffer(_imageBufferHndl,_commandPool,_computeQueue,_statusOK);
}

void MandelComputeHandler::ComputeDouble ()
{
    //printf ("Compute double called\n");
    //printf ("%p %d %d %f %f %f %f %d\n",_imageData,_nx,_ny,_xCent,_yCent,_dx,_dy,_maxiter);
    
    //  If double precision isn't supported by the GPU, fall back on the single precision
    //  version.
    
    if (!_doubleSupportInGPU) {
        Compute();
    } else {
    
        //  Otherwise, what follows is just what happens in Compute, but using the double
        //  precision pipeline and its associated layout, descriptor set, etc..
        
        RecomputeArgs();
        if (_statusOK && _uniformBufferAddr) memcpy(_uniformBufferAddr,&_currentArgs,
                                                                           sizeof(MandelArgs));

        //  Set up the pipeline for the GPU calculation and run it.
        
        _vulkanFramework->RecordComputeCommandBuffer(_commandBuffer,_computePipelineD,
                        _computePipelineLayoutD,&_descriptorSet,_workGroupCounts,_statusOK);
        _vulkanFramework->RunCommandBuffer(_computeQueue,_commandBuffer,_statusOK);
    
        //  If a staged buffer is being used, synch it.
    
        _vulkanFramework->SyncBuffer(_imageBufferHndl,_commandPool,_computeQueue,_statusOK);
    }
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

    std::thread threads[64];
    int iY = 0;
    int iYinc = Ny / nThreads;
    for (int iThread = 0; iThread < nThreads; iThread++) {
        threads[iThread] = std::thread (ComputeRangeInC,Data,Nx,Ny,iY,iY+iYinc,Xcent,Ycent,Dx,Dy,
                                                                                         MaxIter);
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
    //  error, then it returns false. Strictly, there still will be rounding error, but
    //  it probably won't be visible with the current display settings.
    
    float Xinc = float(_nx) / float(_width);
    float X0 = float(_xCent + (float(Ix) - _nx * 0.5) * _dx);
    float X1 = float(_xCent + (float(Ix) + Xinc - _nx * 0.5) * _dx);
    float Yinc = float(_ny) / float(_height);
    float Y0 = float(_yCent + (float(Iy) - _ny * 0.5) * _dy);
    float Y1 = float(_yCent + (float(Iy) + Yinc - _ny * 0.5) * _dy);

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
    //  The same as FloatOKatXY(), bur using double precision.
    
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

#ifdef MANDEL_COMPUTE_TEST
#include <stdio.h>
int main()
{
    MandelComputeHandler Handler(nullptr);
    Handler.Initialise("");
    Handler.SetImageSize(1024,1024);
    Handler.SetCentre(0.270925,0.004725);
    Handler.SetMagnification(15000.0);
    float* Image = Handler.GetImageData();
    if (Image) { for (int I = 0; I < 1024 * 1024; Image[I++] = 42.0);}
    Handler.Compute();
    if (Image) {
        float Vmin = 4096.0;
        float Vmax = 0.0;
        for (int I = 0; I < 1024 * 1024; I++) {
            if (I < 50) printf ("[%d] = %f ",I,Image[I]);
            if (I == 50) printf ("\n");
            if (Image[I] > Vmax) Vmax = Image[I];
            if (Image[I] < Vmin) Vmin = Image[I];
        }
        printf ("Min = %f, max = %f\n",Vmin,Vmax);
    }
    return 0;
}
#endif

/*
                                   P r o g r a m m i n g  N o t e s
 
    o   To run the built-in test program:
 
           c++ -std=c++17 -DMANDEL_COMPUTE_TEST -Wall \
                    MandelComputeHandlerVulkan.cpp ../BasicFramework/BasicFramework.cpp -lvulkan
                    
    o   This code needs better comments in a number of places.
 
*/
