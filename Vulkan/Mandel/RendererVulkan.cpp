
//  ------------------------------------------------------------------------------------------------

//                            R e n d e r e r  V u l k a n . c p p
//
//  The Renderer is a specialised C++ class designed to display sections of the Mandelbrot set as
//  part of the Mandelbrot display program. For complete details, see the comments at the start
//  of the Renderer.h header file. This version uses Vulkan.
//
//  History:
//      3rd Nov 2023. First working version, based on the Metal version. KS.
//     19th Jan 2024. Modified after initial testing on a Linux system. Needed to include
//                    math.h, and unused references to simd (which is Apple-only) removed.
//                    Code has been reworked to use triangle strips rather than triangle
//                    lists, which is rather more efficient. KS.
//     30th Jan 2024. Now supports the use of staged buffers for the vertex and colour data,
//                    as it includes the necessary SyncBuffer() calls. Tested using both staged
//                    and shared buffers. KS.
//     21st Feb 2024. Following testing under Windows 11 with Visual Studio, revised the loop 
//                    code in SetColourDataHistEq() to speed up setting the colour buffer
//                    values (see Programming notes). Also some trivial changes to placate
//                    the rather fussy VisualStudio compiler and minimise warnings. KS.
//     23rd Feb 2024. Split code for SetVertexPositions() and SetVertexDefaultColours()
//                    into separate routines. SLightly neater, and gets around an unwanted
//                    warning from the VisualStudio compiler. KS.
//     26th Feb 2024. General tidying of Metal and Vulkan versions. Constructor now explicitly
//                    is passed a VulkanFramework* rather than a void that it casts. KS.
//     10th Jun 2024. Added some safety checks to the code. Should no longer crash if the shaders
//                    can't be loaded. KS.
//      2nd Aug 2024. Now calls the Framework's SetFrameBufferSize() before creating the SwapChain,
//                    and on a change to the window size. KS.
//     23rd Aug 2024. Changed the names of the shader files used. KS.
//      7th Sep 2024. Moved shaders into default directory, to match the Metal version. KS.
//     14th Sep 2024. Modified following renaming of Framework routines and types. KS.

#include "RendererVulkan.h"

#include <string>
#include <math.h>

//  _debugOptions is the comma-separated list of all the diagnostic levels that the built-in debug
//  handler recognises. If a call to _debug.Log() or .Logf() is added with a new level name, this
//  new name must be added to this string.

const std::string Renderer::_debugOptions = "Setup,Timing";

Renderer::Renderer(KVVulkanFramework* FrameworkPtr)
{
    //  Constructor. For this Vulkan-based version, the setup argument must be the address
    //  of a Vulkan basic framework object that has already had its basic setup performed
    // (including interaction with the windowing system) and a logical device created.
    
    _nx = 0;
    _ny = 0;
    _viewWidth = 512.0;
    _viewHeight = 512.0;
    _frameworkPtr = FrameworkPtr;
    _frames = 0;
    _iterLimit = 1024;
    _imageCount = 0;
    _maxOverVerts = 0;
    _overVerts = 0;
    _pipeline = VK_NULL_HANDLE;
    _overlayPipeline = VK_NULL_HANDLE;
    _bufferHandles.clear();
    _overlayBufferHandles.clear();
    _positionsMemAddr = nullptr;
    _positionsBytes = 0;
    _coloursMemAddr = nullptr;
    _coloursBytes = 0;
    _frameTimer.Restart();
    _commandBuffers.clear();
    _positionsIndex = 0;
    _coloursIndex = 1;
    _currentImage = 0;
    _debug.SetSubSystem("Renderer");
    _debug.LevelsList(_debugOptions);
}

void Renderer::Initialise(const std::string& DebugLevels)
{
    //  We set up everything we possibly can that doesn't depend on the size of the images we're
    //  going to be asked to display. That is pretty much everything except for actually creating
    //  the Vulkan buffers and allocating the associated memory.
    
    bool StatusOK = true;
    _debug.SetLevels(DebugLevels);
    uint32_t PreferedImageCount = 2;
    _frameworkPtr->SetFrameBufferSize(_viewWidth,_viewHeight,StatusOK);
    _imageCount = _frameworkPtr->CreateSwapChain(PreferedImageCount,StatusOK);
    _frameworkPtr->CreateImageViews(StatusOK);
    _frameworkPtr->CreateRenderPass(StatusOK);
    _frameworkPtr->CreateFramebuffers(StatusOK);
    _frameworkPtr->CreateCommandPool(&_commandPool,StatusOK);
    _frameworkPtr->CreateCommandBuffers(_commandPool,_imageCount,_commandBuffers,StatusOK);
    std::vector<VkSemaphore> ImageSemaphores;
    std::vector<VkSemaphore> RenderSemaphores;
    std::vector<VkFence> Fences;
    _frameworkPtr->CreateSyncObjects(_imageCount,ImageSemaphores,RenderSemaphores,Fences,StatusOK);
    if (StatusOK) StatusOK = BuildShaders();
    
    //  Setup main image positions buffer
    
    KVVulkanFramework::KVBufferHandle PosnsHndl = _frameworkPtr->SetBufferDetails(
                                                      0,"VERTEX","SHARED",StatusOK);
    long Locations[1] = {0};
    const char* FormatStrings[1] = {"vec2"};
    long Offsets[1] = {0};
    long Stride = sizeof(PositionVec);
    _frameworkPtr->SetVertexBufferDetails(PosnsHndl,Stride,true,1,
                                         Locations,FormatStrings,Offsets,StatusOK);
    
    //  Now the main image colours buffer
    
    KVVulkanFramework::KVBufferHandle ColoursHndl = _frameworkPtr->SetBufferDetails(
                                                         1,"VERTEX","SHARED",StatusOK);
    Locations[0] = 1;
    FormatStrings[0] = "vec3";
    Offsets[0] = 0;
    Stride = sizeof(ColourVec);
    _frameworkPtr->SetVertexBufferDetails(ColoursHndl,Stride,true,1,
                                         Locations,FormatStrings,Offsets,StatusOK);
    
    _bufferHandles.resize(2);
    _bufferHandles[_positionsIndex] = PosnsHndl;
    _bufferHandles[_coloursIndex] = ColoursHndl;
    
    //  And the overlay positions buffer
    
    Locations[0] = 0;
    FormatStrings[0] = "vec2";
    Stride = sizeof(PositionVec);
    KVVulkanFramework::KVBufferHandle OverlayPosnsHndl = _frameworkPtr->SetBufferDetails(
                                                             0,"VERTEX","SHARED",StatusOK);
    _frameworkPtr->SetVertexBufferDetails(OverlayPosnsHndl,Stride,true,1,
                                                 Locations,FormatStrings,Offsets,StatusOK);
    //  And the overlay colours buffer
    
    Locations[0] = 1;
    FormatStrings[0] = "vec3";
    Stride = sizeof(ColourVec);
    KVVulkanFramework::KVBufferHandle OverlayColoursHndl = _frameworkPtr->SetBufferDetails(
                                                            1,"VERTEX","SHARED",StatusOK);
    _frameworkPtr->SetVertexBufferDetails(OverlayColoursHndl,Stride,true,1,
                                                Locations,FormatStrings,Offsets,StatusOK);

    _overlayBufferHandles.resize(2);
    _overlayBufferHandles[_positionsIndex] = OverlayPosnsHndl;
    _overlayBufferHandles[_coloursIndex] = OverlayColoursHndl;

    //  The size of the image can change - and does whenever the window is resized - but the
    //  size of the buffers needed for the overlay only depends on the maximum number of
    //  iterations, and this is fixed. (At least, it is as the program is currently structured).
    //  So we can create the overlay buffers now.
    
    int MaxOverVerts = (_iterLimit * 2 + 1) * 2;
    int MaxOverVertBytes = MaxOverVerts * sizeof(PositionVec);
    KVVulkanFramework::KVBufferHandle OverlayPosnsHandle = _overlayBufferHandles[_positionsIndex];
    _frameworkPtr->CreateBuffer(OverlayPosnsHandle,MaxOverVertBytes,StatusOK);
    _overlayPositionsMemAddr = (PositionVec*)_frameworkPtr->MapBuffer(
                                OverlayPosnsHandle,&_overlayPositionsBytes,StatusOK);
    int MaxOverColsBytes = MaxOverVerts * sizeof(ColourVec);
    KVVulkanFramework::KVBufferHandle OverlayColoursHandle = _overlayBufferHandles[_coloursIndex];
    _frameworkPtr->CreateBuffer(OverlayColoursHandle,MaxOverColsBytes,StatusOK);
    _overlayColoursMemAddr = (ColourVec*)_frameworkPtr->MapBuffer(
                                OverlayColoursHandle,&_overlayColoursBytes,StatusOK);
    _maxOverVerts = MaxOverVerts;

    //  The main image pipeline
    
    VkPipelineLayout PipelineLayout;
    _frameworkPtr->CreateGraphicsPipeline(_vertexShader,"main",_fragmentShader,"main",
                        "TRIANGLE_STRIP",_bufferHandles,&PipelineLayout,&_pipeline,StatusOK);
    
    //  And the overlay pipeline
    
    _frameworkPtr->CreateGraphicsPipeline(_vertexShader,"main",_fragmentShader,"main",
            "LINE_STRIP",_overlayBufferHandles,&PipelineLayout,&_overlayPipeline,StatusOK);
    
    if (StatusOK) _debug.Log("Setup","Basic Vulkan Setup complete");
    
    SetImageSize(1024,1024);
}


Renderer::~Renderer()
{
}

void Renderer::SetImageSize (int Nx, int Ny)
{
    if (_nx != Nx || _ny != Ny) {
        _nx = Nx;
        _ny = Ny;
        BuildBuffers();
    }
}

void Renderer::SetDrawableSize (float width, float height)
{
    if (_viewHeight != height || _viewWidth != width) {
        _viewWidth = width;
        _viewHeight = height;
        bool StatusOK = true;
        _frameworkPtr->SetFrameBufferSize(int(_viewWidth),int(_viewHeight),StatusOK);
    }
}

void Renderer::SetMaxIter (int MaxIter)
{
    _iterLimit = MaxIter;
}

//  GetDebugOptions() returns the comma-separated list of the various diagnostic levels supported
//  by the renderer. Note that this is a static routine; it can be convenient for a program to
//  have this list available before the renderer is constructed, and it is in any case a fixed
//  list.

std::string Renderer::GetDebugOptions(void)
{
    return _debugOptions;
}

void Renderer::SetColourData (float* imageData, int Nx, int Ny)
{
    //  This was my first attempt at setting a suitable set of colours for the images,
    //  based on scaling over a given percentile range of data, but I decided this didn't
    //  make proper use of the limited number of available colour levels (256 in the
    //  old colour table I'm using). So I moved to the histogram equalisation scheme
    //  implemented in SetColourDataHistEq(). It might be that even that would gain by
    //  scaling within a percentile range, and I might look at that someday.
    
    MsecTimer theTimer;
    
    _nx = Nx;
    _ny = Ny;
    
    float Percentile = 95.0;
    float rangeMin,rangeMax;
    PercentileRange(imageData,Nx,Ny,Percentile,&rangeMin,&rangeMax);
    
    int cptr = 0;
    int iptr = 0;
    

    ColourVec* colours = _coloursMemAddr;
    
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            float data = imageData[iptr++];
            int index = int(((data - rangeMin) * 255.0 / (rangeMax - rangeMin)) + 0.5);
            float R,G,B;
            GetRGB (index,&R,&G,&B);
            ColourVec RGB = {R,G,B};
            for (int I = 0; I < 6; I++) { colours[cptr++] = RGB; }
        }
    }
    
    //printf ("Setting colours took %.2f msec\n",theTimer.ElapsedMsec());
}

void Renderer::SetColourDataHistEq (float* imageData, int Nx, int Ny)
{
    MsecTimer theTimer;
    
    //  Safety check - in case something went wrong in setup.
    
    if (_coloursMemAddr == nullptr || imageData == nullptr) return;
    
    _nx = Nx;
    _ny = Ny;
            
    //  This is made easier by the fact that we know all the values in imageData will
    //  be positive integers. Also, note that we expect a very large number of the
    //  pixels will contain zero, because those are the saturated ones.
    
    //  Allocate two arrays. The first is the colour index array. The Mandelbrot algorithm
    //  is such that no pixel normally gets a value of zero. Values are from 1 to 1 below the
    //  iteration limit (_iterLimit, usually 1024). However, any pixel that reaches or exceeds
    //  that iteration limit is regarded as being inside the Mandelbrot set, and the code for
    //  the algorithm sets these pixels to zero. So the potential range of data values
    //  in the array to be displayed goes from 0 up to _iterLimit - 1. We have a limited set
    //  of colours available - the look up table we use has 256 entries - and we want to
    //  distribute these as usefully as possible between the data values actually in the
    //  image we have been passed. The colour index array will contain the index into the
    //  colour table to be used for each data value. Data value zero will have colour index
    //  zero, because this is black in this colour table, and this usually looks best for
    //  displaying the pixels actually in the Mandelbrot set. We then want to distribute
    //  the rest of the available 255 colour index values as evenly as possible amongst
    //  only those data values that actually occur in the current data.
    
    int* ColourIndex = (int*) malloc(_iterLimit * sizeof(int));
    
    //  Now the second array. Hist will be the histogram of the data values. Hist[0] will
    //  be the count of pixels within the Mandelbrot set, and Hist[i] is the count of pixels
    //  with value i. We will use the distribution of actual values in the data to allocate
    //  the colour table indices amongst the data values.
    
    int* Hist = (int*) malloc(_iterLimit * sizeof(int));
    
    for (int I = 0; I < _iterLimit; I++) Hist[I] = 0;
    for (int iptr = 0; iptr < (Nx * Ny); iptr++) {
        int iData = int(imageData[iptr]);
        if (iData >= 0 && iData < _iterLimit) Hist[iData]++;
    }
      
    //  Determine the range of actual data values, ignoring the zero values pixels
    //  inside the Mandelbrot set. We can do this just by starting at each end of
    //  the data histogram and seeing when we first hit a non-zero count.
    
    int maxV = 0;
    int minV = _iterLimit;
    for (int iptr = 1; iptr < _iterLimit; iptr++) {
        int iData = int(Hist[iptr]);
        if (iData > 0) {
            minV = iptr;
            break;
        }
    }
    for (int iptr = _iterLimit - 1; iptr > 0; iptr--) {
        int iData = int(Hist[iptr]);
        if (iData > 0) {
            maxV = iptr;
            break;
        }
    }
    int nonZeroCount = Hist[0];
    if (maxV < minV) maxV = minV = 1;
    
    //  This loop aims to allocate the 256 levels of colour available in the colour
    //  table between the various data levels so as to distribute the colours as
    //  evenly as possible. It looks at the number of non-zero pixels in the data,
    //  and tries to allocate the data levels equally between them. (It's a process
    //  of histogram equalisation.) This turns the ColourIndex array into a lookup
    //  table that gives the index into the Colour table to be used for each data value.
    
    const int LevelsAvailable = 256;
    int Levels = LevelsAvailable;
    int PixPerLevel = nonZeroCount / Levels;
    int PixCount = 0;
    int Lev = 1;
    int Target = PixPerLevel;
    ColourIndex[0] = 0;
    for (int I = 1; I < minV; I++) ColourIndex[I] = Lev;
    for (int I = minV; I <= maxV; I++) {
        PixCount += Hist[I];
        ColourIndex[I] = Lev;
        if (PixCount > Target) {
            Lev++;
            Levels--;
            if (Lev >= LevelsAvailable) Lev = LevelsAvailable - 1;
            if (Levels < 1) Levels = 1;
            nonZeroCount -= PixCount;
            PixPerLevel = nonZeroCount / Levels - 1;
            Target += PixPerLevel;
        }
    }
    for (int I = maxV + 1; I < _iterLimit; I++) ColourIndex[I] = LevelsAvailable - 1;
    
    //  If there were fewer data values between minV and maxV than there were display levels
    //  available, we won't have used the whole colour range, although each data level will
    //  have its own colour level. This loop scales the levels up so they cover the whole
    //  colour range.
    
    if (Lev < (LevelsAvailable - 1)) {
        float Scale = float(LevelsAvailable - 1) / (float)Lev;
        for (int I = minV; I <= maxV; I++) {
            ColourIndex[I] = int(float(ColourIndex[I]) * Scale);
            if (ColourIndex[I] >= LevelsAvailable) ColourIndex[I] = LevelsAvailable - 1;
        }
    }
    
    //  ColourIndex[i] now represents the colour level to be used for data of value i.
    
    int cptr = 0;
    int iptr = 0;
    
    ColourVec* colours = _coloursMemAddr;

    //  The colours need to match the vertices as set up by BuildBuffers(). See comments 
    //  there about the number of vertices needed for each image line.
        
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            int idata = int(imageData[iptr++]);
            int index = ColourIndex[idata];
            float R,G,B;
            GetRGB (index,&R,&G,&B);
            ColourVec RGB = {R,G,B};
            if (Ix == 0) {
                for (int I = 0; I < 5; I++) { colours[cptr++] = RGB; }
            } else {
                colours[cptr++] = RGB;
                colours[cptr++] = RGB;
            }
        }
        colours[cptr++] = { 0.0,0.0,0.0 };
    }
    
    //  If the colours buffer is shared, it needs explicit synchronisation when modified.
    //  See comments in BuildBuffers() - this is a null operation for buffers that don't need it.

    bool StatusOK = true;
    VkQueue queueHndl;
    KVVulkanFramework::KVBufferHandle ColoursHandle = _bufferHandles[_coloursIndex];
    _frameworkPtr->GetDeviceQueue(&queueHndl,StatusOK);
    _frameworkPtr->SyncBuffer(ColoursHandle,_commandPool,queueHndl,StatusOK);

    free (Hist);
    free (ColourIndex);
    //printf("Full colour handling took %.2f msec\n", theTimer.ElapsedMsec());
}

void Renderer::PercentileRange (float* imageData,int Nx,int Ny,float Percentile,
                                                 float* rangeMin,float* rangeMax)
{
    //  This is made easier by the fact that we know all the values in imageData will
    //  be positive integers. Also, note that we expect a very large number of the
    //  pixels will contain zero, because those are the saturated ones.
    
    *rangeMin = 0.0;
    *rangeMax = 0.0;

    int maxV = 0;
    int nonZeroCount = 0;
    for (int iptr = 0; iptr < (Nx * Ny); iptr++) {
        int iData = int(imageData[iptr++]);
        if (iData > maxV) maxV = iData;
        if (iData != 0) nonZeroCount++;
    }
    
    //  Build up a histogram of the data values. Note that hist[0] is the count
    //  for pixels whose data value is 1. Zero valued pixels are ignored. So the
    //  histogram has the pixel counts with values from 1 up to and including
    //  maxV, with hist[i] representing the data value i+1.
    
    int* hist = (int*) malloc(maxV * sizeof(int));
    if (hist) {
        for (int I = 0; I < maxV; I++) hist[I] = 0;
        for (int iptr = 0; iptr < (Nx * Ny); iptr++) {
            int iData = int(imageData[iptr++]);
            if (iData > 0) hist[iData - 1]++;
        }

        //  excessPix is the number of pixels we want to drop off from the range at
        //  each end, if what's left in the middle covers Percentile percent of the
        //  non-zero pixels.

        *rangeMin = 0.0;
        int excessPix = int(float(nonZeroCount) * 0.01 * (100.0 - Percentile) / 2.0);
        int count = 0;
        for (int I = 0; I < maxV; I++) {
            count += hist[I];
            if (count > excessPix) {
                *rangeMin = float(I + 1);
                break;
            }
        }
        *rangeMax = float(maxV);
        count = 0;
        for (int I = maxV - 1; I >= 0; I--) {
            count += hist[I];
            if (count > excessPix) {
                *rangeMax = float(I + 1);
                break;
            }
        }

        //printf ("%f percentile range from %f to %f\n",Percentile,*rangeMin,*rangeMax);

        free(hist);
    }
}

bool Renderer::BuildShaders()
{
    bool StatusOK = true;
    _frameworkPtr->CreateShaderModuleFromFile("MandelVert.spv",&_vertexShader,StatusOK);
    _frameworkPtr->CreateShaderModuleFromFile("MandelFrag.spv",&_fragmentShader,StatusOK);
    return StatusOK;
}

void Renderer::BuildBuffers()
{
    //  This is called when the initial size of the image to display is first known, and is then
    //  called subsequently if that size changes.
    
    bool StatusOK = true;
    
    //  First, create (or resize) the Vulkan buffers used for position and colour data. This
    //  also creates (or resizes) the associated actual memory arrays used by the GPU and maps
    //  them so the CPU can access them.
    
    //  Getting the number of vertices right is made tricky by the zero-area triangles at
    //  start and end of each image line. All image pixels in a line except for the verry first
    //  need two vertices, one for each of the two triangles used. That's the (nx-1)*2 part.
    //  Then the first pixel needs 5 vertices - 3 for the zero-area triangle that starts
    //  it off, then one more for the two triangles that actually get drawn to make up that
    //  pixel. And then there is one more vertex to provide the zero-area triangle at the
    //  end of the line. Those 5 + 1 make the 6 extra vertices for each line.
    
    int Nx = _nx;
    int Ny = _ny;

    if (Nx <= 0 || Ny <= 0) return;

    _debug.Logf("Setup","Rebuilding renderer buffers to %d by %d.",Nx,Ny);
    MsecTimer theTimer;
    
    long NumVertices = ((Nx - 1) * 2 + 6) * Ny;
     
    long PosnsSizeInBytes = sizeof(PositionVec) * NumVertices;
    KVVulkanFramework::KVBufferHandle PosnsHandle = _bufferHandles[_positionsIndex];
    if (!_frameworkPtr->IsBufferCreated(PosnsHandle,StatusOK)) {
        _frameworkPtr->CreateBuffer(PosnsHandle,PosnsSizeInBytes,StatusOK);
    } else {
        _frameworkPtr->ResizeBuffer(PosnsHandle,PosnsSizeInBytes,StatusOK);
    }
    _positionsMemAddr = (PositionVec*)_frameworkPtr->MapBuffer(
                                                    PosnsHandle,&_positionsBytes,StatusOK);
    long ColoursSizeInBytes = sizeof(ColourVec) * NumVertices;
    KVVulkanFramework::KVBufferHandle ColoursHandle = _bufferHandles[_coloursIndex];
    if (!_frameworkPtr->IsBufferCreated(ColoursHandle,StatusOK)) {
        _frameworkPtr->CreateBuffer(ColoursHandle,ColoursSizeInBytes,StatusOK);
    } else {
        _frameworkPtr->ResizeBuffer(ColoursHandle,ColoursSizeInBytes,StatusOK);
    }
    _coloursMemAddr = (ColourVec*)_frameworkPtr->MapBuffer(ColoursHandle,&_coloursBytes,StatusOK);

    _debug.Logf("Timing","Resized renderer buffers at %.2f msec",theTimer.ElapsedMsec());

    //  This sets up a pretty inefficient way of displaying an Nx by Ny image. Each pixel
    //  of the image is represented by two triangles (two vertex positions are common to
    //  both) and all six of those vertices are set to the same colour (based on the data
    //  value for that pixel. The original code used a list of triangles, with each triangle
    //  specified with all three vertices. Since the triangles are actually adjacent, this
    //  meant a lot of vertices were being specified redundantly, and this code now uses
    //  a single triangle strip. The trick here is how the jump from one line of the image
    //  to the next is handled, and involves starting and ending each line with a zero-area
    //  triangle. This isn't rendered, but provides a starting point for the new line.
    
    //  The vertex positions remain constant even if a new image is computed, so long as the
    //  size of the image has not changed, but new image data requires that the colour values
    //  for each pixel be recalculated.
        
    //  Create scratch arrays to fill with the positions and colours for the vertices.
    
    PositionVec *positions = new PositionVec[NumVertices];
    ColourVec *colours = new ColourVec[NumVertices];
 
    //  Set the fixed positions for each vertex, and then set the default colours for
    //  each vertex. The default colours will never actually be seen unless something
    //  goes wrong, but it helps to have something visually meaningful, if only for
    //  testing.
    
    SetVertexPositions(positions,Nx,Ny);
    SetVertexDefaultColours(colours,Nx,Ny);
    _debug.Logf("Timing","Recalculated vertices & colours at %.2f msec",theTimer.ElapsedMsec());

    //  Copy the positions and colours into the GPU buffers. (In principle, we could
    //  just have called SetVertexPositions() and SetVertexDefaultColours() with the
    //  mapped buffer addresses, but it seems neater this way - and it may be quite
    //  efficient to do the buffer copies in one memcpy() call. Or that may be extra
    //  overhead. In any case, it's only done once.
    
    memcpy(_positionsMemAddr,positions,_positionsBytes);
    memcpy(_coloursMemAddr,colours,_coloursBytes);

    //  This code is only needed if the positions and colours buffers have been set up as
    //  staged buffers (which need explicit synchronisation) rather than as shared buffers.
    //  However, the SyncBuffer() call is essentially a null operation with a shared buffer,
    //  so there's no need to check which is being used.
    
    VkQueue queueHndl;
    _frameworkPtr->GetDeviceQueue(&queueHndl,StatusOK);
    _frameworkPtr->SyncBuffer(PosnsHandle,_commandPool,queueHndl,StatusOK);
    _frameworkPtr->SyncBuffer(ColoursHandle,_commandPool,queueHndl,StatusOK);

    _debug.Logf("Timing","Copied data to renderer buffers at %.2f msec",theTimer.ElapsedMsec());
    
    if (positions) delete[] positions;
    if (colours) delete[] colours;
}

void Renderer::SetVertexPositions (PositionVec positions[],int Nx,int Ny)
{
    //  The positions array must have at least ((Nx - 1) * 2 + 6) * Ny elements.
    
    //  Set up the vertex positions for all the triangles. Note that the coordinate range
    //  for a View is from -1.0 to +1.0. For each pixel to be displayed, we calculate the
    //  range it covers in this coordinate system as X to Xp1, Y to Yp1, and then set
    //  two trianges to make up the rectangle that will be used to represent the pixel.
    //  Metal and Vulkan reverse the Y-coordinate between them. The Metal version of this
    //  code has Yp1 = Y + Yinc instead of Y - Yinc, and does not have the two lines that
    //  multiply the calculated Y and Yp1 coordinates by -1.0
    
    int Nv = 0;
    float Yinc = 2.0f / float(Ny);
    float Xinc = 2.0f / float(Nx);
    for (int Iy = 0; Iy < Ny; Iy++) {
        float Y = Iy * Yinc - 1.0f;
        float Yp1 = Y - Yinc;
        Yp1 *= -1.0;
        Y *= -1.0;
        float X = -1.0;
        float Xp1 = X + Xinc;
        
        //  This draws a zero area triangle - the first two vertices are the same - then
        //  draws the two triangles for the first pixel in the line.
        
        positions[Nv++] = { X,Y };
        positions[Nv++] = { X,Y };
        positions[Nv++] = { X,Yp1 };
        positions[Nv++] = { Xp1,Y };
        positions[Nv++] = { Xp1,Yp1 };
        
        //  After that, we only need to add two vertices for each image pixel, one
        //  for each of the two triangles.
        
        for (int Ix = 1; Ix < Nx; Ix++) {
            Xp1 = (Ix + 1) * Xinc - 1;
            positions[Nv++] = {Xp1,Y};
            positions[Nv++] = {Xp1,Yp1};
        }
        
        //  And we end off each line with another zero-area triangle, just by repeating
        //  the last vertex.
        
        positions[Nv++] = {Xp1,Yp1};
    }
}

void Renderer::SetVertexDefaultColours (ColourVec colours[],int Nx,int Ny)
{
    //  The colours array must have at least ((Nx - 1) * 2 + 6) * Ny elements.

    //  Set a default set of colours for each of the vertices. This sets a
    //  set of grey scale values decreasing with distance from the centre of
    //  the image. These will be overwritten by SetColourDataHistEq(). All the
    //  6 vertices that specify the two triangles used for each pixel have the
    //  same colour, giving a rectange of solid colour when interpreted by the
    //  shader (which strictly varies the colour linearly between the vertices,
    //  but if they're all the same we get a solid rectangle). Generally, this
    //  grey-scale 'dome' will never get a chance to be seen, but the buffers
    //  do need to be initialised to something, and this shows how to set the
    //  colour buffer values.
    
    int Nc = 0;
    float NxBy2 = float(Nx) * 0.5f;
    float NyBy2 = float(Ny) * 0.5f;
    float MaxDistSq = NxBy2 * NyBy2;
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            float Xdist = float(fabs(float(Ix) - NxBy2));
            float Ydist = float(fabs(float(Iy) - NyBy2));
            float DistSq = Xdist * Xdist + Ydist * Ydist;
            float Grey = float(1.0 - sqrt(DistSq / MaxDistSq));
            ColourVec RGB= {Grey,Grey,Grey};
            int Vertices = 2;
            
            //  To match the colours to the vertices in the triangle strip, the first
            //  pixel for each line has 5 vertices and each subsequent pixel has 2.
            
            if (Ix == 0) Vertices = 5;
            for (int I = 0; I < Vertices; I++) { colours[Nc++] = RGB; }
        }
        
        //  Then there is one extra colour needed for at the end of each line for the
        //  final vertex that produces the terminating zero-area triangle.

        colours[Nc++] = {0.0,0.0,0.0};
    }
}

void Renderer::SetOverlay(float* xPosns,float* yPosns,int nPosns)
{
    if (nPosns > 0) {
        if (nPosns > _maxOverVerts) nPosns = _maxOverVerts;
        float xScale = 2.0 / _viewWidth;
        float yScale = 2.0 / _viewHeight;
        for (int i = 0; i < nPosns; i++) {
            float x = xPosns[i] * xScale - 1.0;
            float y = 1.0 - yPosns[i] * yScale;
            _overlayPositionsMemAddr[i] = {x,y};
            _overlayColoursMemAddr[i] = {1.0,1.0,1.0};
        }
    }
    _overVerts = nPosns;
}

void Renderer::Draw(void* /*pView*/, float* imageData )
{
    MsecTimer theTimer;
    
    SetColourDataHistEq(imageData,_nx,_ny);

    int Nx = _nx;
    int Ny = _ny;
    int NumVertices = ((Nx - 1) * 2 + 6) * Ny;
    
    //MsecTimer theTimer;

    bool StatusOK = true;
    
    if (_frameworkPtr) {
        
        //  If we want to draw the lines for a Mandelbrot path as an overlay as well as the main
        //  Mandelbror image, we need to run two pipelines for each frame, one for the image
        //  (drawn as triangles) and one for the overlap (the lines). (A single pipeline can only
        //  handle one data type.) DrawGraphicsFrame() will run a number of pipelines with their
        //  associated buffer sets and vertex counts, but we need to set up the arrays that
        //  describe all this. Whether the second elements of the arrays are used depends on
        //  the number of Stages specified, set to 2 for image and overlay, 1 just for image.
        
        int VertexCounts[2] = {NumVertices,_overVerts};
        VkPipeline Pipelines[2] = {_pipeline,_overlayPipeline};
        std::vector<KVVulkanFramework::KVBufferHandle> BufferHandleSets[2] =
                                                       {_bufferHandles,_overlayBufferHandles};
        int Stages = 1;
        if (_overVerts > 0) Stages = 2;
        _frameworkPtr->DrawGraphicsFrame(_currentImage,_commandBuffers[_currentImage],Stages,
                                          VertexCounts,BufferHandleSets,Pipelines,StatusOK);
    }
    if (_currentImage == 1) _currentImage = 0;
    else _currentImage = 1;

    _frames++;
    if (_frames % 1000 == 0) {
        //float msec = _frameTimer.ElapsedMsec();
        //printf ("Frames %d in %.2f sec, frames/sec = %.2f\n",
        //                               _frames,msec*0.001,_frames/(msec * 0.001));
    }
}

void Renderer::GetRGB (int Index, float* R, float* G, float* B) {
    
    //  This is the Figaro default colour table, initially provided by John Tonry.
    
    static int GrjtColourData[3][256] = {
        {  0,    128,    123,    123,    119,    119,    114,    114,
         110,    110,    105,    105,    100,    100,     95,     95,
          90,     90,     85,     85,     80,     80,     75,     75,
          70,     70,     64,     64,     59,     59,     53,     53,
          48,     48,     42,     42,     36,     36,     31,     31,
          25,     25,     19,     19,     12,     12,      6,      6,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,     24,     24,     48,     48,     73,     73,
          98,     98,    123,    123,    148,    148,    174,    174,
         200,    200,    201,    201,    202,    202,    203,    203,
         204,    204,    205,    205,    206,    206,    207,    207,
         208,    208,    209,    209,    210,    210,    211,    211,
         212,    212,    213,    213,    214,    214,    215,    215,
         216,    216,    217,    217,    218,    218,    219,    219,
         220,    220,    221,    221,    222,    222,    223,    223,
         224,    224,    225,    225,    226,    226,    227,    227,
         228,    228,    229,    229,    230,    230,    231,    231,
         232,    232,    233,    233,    234,    234,    235,    235,
         236,    236,    237,    237,    238,    238,    239,    239,
         240,    240,    241,    241,    242,    242,    243,    243,
         244,    244,    245,    245,    246,    246,    247,    247,
         248,    248,    249,    249,    250,    250,    251,    251,
         252,    252,    253,    253,    254,    254,    255,    255 },
         { 0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      5,      5,     10,     10,     14,     14,
          19,     19,     24,     24,     30,     30,     35,     35,
          40,     40,     45,     45,     51,     51,     56,     56,
          61,     61,     67,     67,     72,     72,     78,     78,
          84,     84,     90,     90,     95,     95,    101,    101,
         107,    107,    113,    113,    119,    119,    126,    126,
         132,    132,    138,    138,    144,    144,    151,    151,
         157,    157,    164,    164,    170,    170,    177,    177,
         184,    184,    185,    185,    186,    186,    187,    187,
         188,    188,    189,    189,    190,    190,    191,    191,
         192,    192,    193,    193,    194,    194,    195,    195,
         196,    196,    197,    197,    198,    198,    199,    199,
         200,    200,    195,    195,    189,    189,    184,    184,
         178,    178,    173,    173,    167,    167,    162,    162,
         156,    156,    150,    150,    144,    144,    138,    138,
         132,    132,    126,    126,    120,    120,    114,    114,
         108,    108,    102,    102,     95,     95,     89,     89,
          82,     82,     76,     76,     69,     69,     63,     63,
          56,     56,     49,     49,     42,     42,     35,     35,
          28,     28,     21,     21,     14,     14,      7,      7,
           0,      0,     10,     10,     19,     19,     29,     29,
          39,     39,     49,     49,     59,     59,     70,     70,
          80,     80,     90,     90,    101,    101,    111,    111,
         122,    122,    133,    133,    143,    143,    154,    154,
         165,    165,    176,    176,    187,    187,    199,    199,
         210,    210,    221,    221,    233,    233,    244,    244 },
        {  0,    128,    129,    129,    130,    130,    131,    131,
         132,    132,    133,    133,    134,    134,    135,    135,
         136,    136,    137,    137,    138,    138,    139,    139,
         140,    140,    141,    141,    142,    142,    143,    143,
         144,    144,    145,    145,    146,    146,    147,    147,
         148,    148,    149,    149,    150,    150,    151,    151,
         152,    152,    153,    153,    154,    154,    155,    155,
         156,    156,    157,    157,    158,    158,    159,    159,
         160,    160,    161,    161,    162,    162,    163,    163,
         164,    164,    165,    165,    166,    166,    167,    167,
         168,    168,    169,    169,    170,    170,    171,    171,
         172,    172,    173,    173,    174,    174,    175,    175,
         176,    176,    177,    177,    178,    178,    179,    179,
         180,    180,    181,    181,    182,    182,    183,    183,
         184,    184,    162,    162,    139,    139,    117,    117,
          94,     94,     71,     71,     47,     47,     24,     24,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,      0,      0,      0,      0,      0,      0,
           0,      0,     10,     10,     19,     19,     29,     29,
          39,     39,     49,     49,     59,     59,     70,     70,
          80,     80,     90,     90,    101,    101,    111,    111,
         122,    122,    133,    133,    143,    143,    154,    154,
         165,    165,    176,    176,    187,    187,    199,    199,
         210,    210,    221,    221,    233,    233,    244,    244}};
        
    if (Index > 255) Index = 255;
    if (Index < 0) Index = 0;
    
    *R = float(GrjtColourData[0][Index]) / 255.0f;
    *G = float(GrjtColourData[1][Index]) / 255.0f;
    *B = float(GrjtColourData[2][Index]) / 255.0f;
}

/*
                                   P r o g r a m m i n g   N o t e s
 
    o   The way the Nx by Ny pixels of the image are rendered is inefficient. There are
        two triangles for each pixel, making up a square, originally meaning there were 
        Nx * Ny * 2 * 3 vertices that had to be specified. Using triangle strips cuts that
        down to ((Nx - 1) * 2 + 6) * Ny (see BuildBuffer() comments for just where that
        comes from). It may be that using just two triangles to form a rectangle covering
        the whole image and using a texture would be even faster, but in any case, the time
        taken for the rendering is generally small compared to that required for the computation.
 
    o   The code in SetColourDataHistEq() that sets the colour buffer values used to have:

            int Vertices = 2;
            if (Ix == 0) Vertices = 5;
            for (int I = 0; I < Vertices; I++) { colours[cptr++] = RGB; }

        This was changed (21/2/24) to:

            if (Ix == 0) {
                for (int I = 0; I < 5; I++) { colours[cptr++] = RGB; }
            } else {
                colours[cptr++] = RGB;
                colours[cptr++] = RGB;
            }

        The effect is the same, and the original code ran fine on MacOS and Linux, but when
        tested under Windows, it was found that it ran about 3 to 4 times more slowly than
        expected. The revised code, which doesn't have a loop with a variable limit, ran
        much faster. This was with the VisualStudio C++ compiler set to optimize for speed.
        It just goes to show.

    o   All the colours for the overlay vertices are white, and there's really no need for
        the overlay colours buffer. It's only there because having the overlay vertex buffers
        with the same structure as the image vertex buffers means the same shaders can be
        used for both. It would be more efficient for the overlay pipeline to use a different
        vertex shader that simply set all colours to white and didn't use a colour buffer
        at all.

 */
