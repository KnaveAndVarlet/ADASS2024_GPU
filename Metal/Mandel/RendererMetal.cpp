
//  ------------------------------------------------------------------------------------------------

//                                  R e n d e r e r  M e t a l . c p p
//
//  The Renderer is a specialised C++ class designed to display sections of the Mandelbrot set as
//  part of the Mandelbrot display program. For complete details, see the comments at the start
//  of the Renderer.h header file. This version uses Metal.
//
//  History:
//     22 Jul 2023. First version after repackaging of the original testbed code. KS.
//     29 Sep 2023. Now initialises the vertex buffer addresses  properly in the constructor. KS.
//      6 Feb 2024. Depending on setting of _useManagedBuffers will use either shared or managed
//                  buffers. Managed may be faster for discrete GPUs. Also changed the rendering
//                  scheme to use a triangle strip rather than separate triangles, using a
//                  modified version of the code originally used for the Vulkan version. KS.
//     26 Feb 2024. Rationalising slightly the Metal and Vulkan versions, this has been renamed
//                  from Renderer.cpp to RendererMetal.cpp. KS.
//     30 Aug 2024. Added use of a debug handler and support for GetDebugOptions(). Also added
//                  Initialise(). This brings this up to date with the latest changes to the
//                  Vulkan version. KS.
//      4 Sep 2024. Added SetOverlay() to support display of Mandelbrot paths. KS.

#include "RendererMetal.h"

#include <simd/simd.h>

//  _debugOptions is the comma-separated list of all the diagnostic levels that the built-in debug
//  handler recognises. If a call to _debug.Log() or .Logf() is added with a new level name, this
//  new name must be added to this string.

const std::string Renderer::_debugOptions = "Setup,Timing";

Renderer::Renderer(MTL::Device* pDevice)
: _pDevice(pDevice->retain())
{
    _nx = 0;
    _ny = 0;
    _viewWidth = 512.0;
    _viewHeight = 512.0;
    _frames = 0;
    _iterLimit = 1024;
    _pVertexPositionsBuffer = nullptr;
    _pVertexColorsBuffer = nullptr;
    _pCommandQueue = _pDevice->newCommandQueue();
    _useManagedBuffers = true;
    _debug.SetSubSystem("Renderer");
    _debug.LevelsList(_debugOptions);
    _pOverlayVertexBuffer = nullptr;
    _pOverlayColorsBuffer = nullptr;
    _maxOverVerts = 0;
    _overVerts = 0;
}

Renderer::~Renderer()
{
    if (_pVertexPositionsBuffer) _pVertexPositionsBuffer->release();
    if (_pVertexColorsBuffer) _pVertexColorsBuffer->release();
    if (_pPSO) _pPSO->release();
    if (_pCommandQueue) _pCommandQueue->release();
    if (_pDevice) _pDevice->release();
}

void Renderer::Initialise(const std::string& DebugLevels)
{
    //  We set up everything we possibly can that doesn't depend on the size of the images we're
    //  going to be asked to display. For this metal version that doesn't amount to much other
    //  than building the shaders. But having this as a separate routine allows this to be used
    //  to enable the various debug levels before doing anything worth debugging.
    
    _debug.SetLevels(DebugLevels);
    //printf ("Set debuglevels to '%s'\n",DebugLevels.c_str());
    BuildShaders();
    _debug.Log("Setup","Basic Metal Setup complete");
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
    simd::float3* colours = (simd::float3*)_pVertexColorsBuffer->contents();
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            float data = imageData[iptr++];
            int index = int(((data - rangeMin) * 255.0 / (rangeMax - rangeMin)) + 0.5);
            float R,G,B;
            GetRGB (index,&R,&G,&B);
            //R = G = B = float(index) / 256.0;  // KLUDGE for grey scale
            simd::float3 RGB= {R,G,B};
            for (int I = 0; I < 6; I++) { colours[cptr++] = RGB; }
        }
    }
    if (_useManagedBuffers) {
        _pVertexColorsBuffer->didModifyRange(NS::Range::Make(0,_pVertexColorsBuffer->length()));
    }
    
    //printf ("Setting colours took %.2f msec\n",theTimer.ElapsedMsec());
}

void Renderer::SetColourDataHistEq (float* imageData, int Nx, int Ny)
{
    MsecTimer theTimer;
    
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
    simd::float3* colours = (simd::float3*)_pVertexColorsBuffer->contents();
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            int idata = int(imageData[iptr++]);
            int index = ColourIndex[idata];
            float R,G,B;
            GetRGB (index,&R,&G,&B);
            simd::float3 RGB= {R,G,B};
            int Vertices = 2;
            if (Ix == 0) Vertices = 5;
            for (int I = 0; I < Vertices; I++) { colours[cptr++] = RGB; }
        }
        colours[cptr++] = {0.0,0.0,0.0};
    }

    if (_useManagedBuffers) {
        _pVertexColorsBuffer->didModifyRange(NS::Range::Make(0,_pVertexColorsBuffer->length()));
    }

    free (Hist);
    free (ColourIndex);
    //printf ("Setting colours took %.2f msec\n",theTimer.ElapsedMsec());
}

void Renderer::PercentileRange (float* imageData,int Nx,int Ny,float Percentile,
                                                 float* rangeMin,float* rangeMax)
{
    //  This is made easier by the fact that we know all the values in imageData will
    //  be positive integers. Also, note that we expect a very large number of the
    //  pixels will contain zero, because those are the saturated ones.
    
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
    for (int I = 0; I < maxV; I++) hist[I] = 0.0;
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
    
    free (hist);
    
}

bool Renderer::BuildShaders()
{
    bool ReturnOK = false;
    
    using NS::StringEncoding::UTF8StringEncoding;

    //  This is a very basic pair of shaders, taken from the code used for the examples supplied
    //  with metal-cpp. The vertex shader takes two buffers, one giving the x,y,z positions for the
    //  various vertices and the other giving the corresponding R,G,B colours. It will be invoked
    //  for each vertex and returns the position as x,y,z,w where w is a scaling factor by which
    //  x,y,z are divided to scale them to the -1 to +1 range of the normalised coordinates
    //  used by Metal, and returns the colour as the r,g,b values unchanged. So it does very
    //  little indeed. The fragment shader is invoked for every display pixel, and is passed
    //  the interpolated r,g,b colour for that display pixel. It simply returns this as an
    //  r,g,b,a value where a is the opacity value, in this case set to 1 for fully opaque.
    //  In other words, neither do very much, but they have to be supplied for the pipeline
    //  to run.
    
    const char* shaderSrc = R"(
        #include <metal_stdlib>
        using namespace metal;

        struct v2f
        {
            float4 position [[position]];
            half3 color;
        };

        v2f vertex vertexMain( uint vertexId [[vertex_id]],
                               device const float3* positions [[buffer(0)]],
                               device const float3* colors [[buffer(1)]] )
        {
            v2f o;
            o.position = float4( positions[ vertexId ], 1.0 );
            o.color = half3 ( colors[ vertexId ] );
            return o;
        }

        half4 fragment fragmentMain( v2f in [[stage_in]] )
        {
            return half4( in.color, 1.0 );
        }
    )";

    //  Compile the shaders on the fly from the text above. It might be more usual to have
    //  them already compiled into a Metal library (see the compute handler code for an
    //  example of this), but it can be simpler to have everything in one place like this.
    
    NS::Error* pError = nullptr;
    MTL::Function* pVertexFn = nullptr;
    MTL::Function* pFragFn = nullptr;
    MTL::RenderPipelineDescriptor* pDesc = nullptr;
    
    MTL::Library* pLibrary =
        _pDevice->newLibrary(NS::String::string(shaderSrc,UTF8StringEncoding), nullptr, &pError );
    if ( !pLibrary ) {
        printf( "%s", pError->localizedDescription()->utf8String() );
    } else {

        pVertexFn = pLibrary->newFunction( NS::String::string("vertexMain",UTF8StringEncoding) );
        pFragFn = pLibrary->newFunction( NS::String::string("fragmentMain",UTF8StringEncoding) );

        pDesc = MTL::RenderPipelineDescriptor::alloc()->init();
        pDesc->setVertexFunction( pVertexFn );
        pDesc->setFragmentFunction( pFragFn );
        pDesc->colorAttachments()->object(0)->setPixelFormat(
                                                MTL::PixelFormat::PixelFormatBGRA8Unorm_sRGB );

        _pPSO = _pDevice->newRenderPipelineState( pDesc, &pError );
        if ( !_pPSO ) {
           printf( "%s", pError->localizedDescription()->utf8String() );
        } else {
            ReturnOK = true;
        }
    }

    if (pVertexFn) pVertexFn->release();
    if (pFragFn) pFragFn->release();
    if (pDesc) pDesc->release();
    if (pLibrary) pLibrary->release();
    
    return ReturnOK;
}

void Renderer::BuildBuffers()
{
    //  This is called when the initial size of the image to display is first known, and is then
    //  called subsequently if that size changes.

    //  First, create (or resize) the Metal buffers used for position and colour data.

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
    
    const size_t NumVertices = ((Nx - 1) * 2 + 6) * Ny;;

    const size_t positionsDataSize = NumVertices * sizeof( simd::float3 );
    const size_t colorDataSize = NumVertices * sizeof( simd::float3 );
    
    //  Should we test to see if the image dimensions have actually changed?
    
    if (_pVertexPositionsBuffer) _pVertexPositionsBuffer->release();
    if (_pVertexColorsBuffer) _pVertexColorsBuffer->release();
    
    MTL::ResourceOptions storageMode = MTL::ResourceStorageModeShared;
    if (_useManagedBuffers) storageMode = MTL::ResourceStorageModeManaged;
    MTL::Buffer* pVertexPositionsBuffer = _pDevice->newBuffer( positionsDataSize, storageMode);
    MTL::Buffer* pVertexColorsBuffer = _pDevice->newBuffer(colorDataSize, storageMode);
    
    _pVertexPositionsBuffer = pVertexPositionsBuffer;
    _pVertexColorsBuffer = pVertexColorsBuffer;
    _debug.Logf("Timing","Resized renderer buffers at %.2f msec",theTimer.ElapsedMsec());

    //  This adds overlay buffers, just as a test...
    if (_pOverlayVertexBuffer == nullptr) {
        storageMode = MTL::ResourceStorageModeShared;
        int MaxOverVerts = (_iterLimit * 2 + 1) * 2;
        printf ("MaxOverVerts = %d\n",MaxOverVerts);
        int MaxOverVertBytes = MaxOverVerts * sizeof(simd::float3);
        MTL::Buffer* pOverlayVertexBuffer = _pDevice->newBuffer(MaxOverVertBytes,storageMode);
        MTL::Buffer* pOverlayColorsBuffer = _pDevice->newBuffer(MaxOverVertBytes,storageMode);
        _pOverlayVertexBuffer = pOverlayVertexBuffer;
        _pOverlayColorsBuffer = pOverlayColorsBuffer;
        _maxOverVerts = MaxOverVerts;
    }
        
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
    
    simd::float3 *positions = new simd::float3[NumVertices];
    simd::float3 *colors = new simd::float3[NumVertices];
    
    //  Set up the vertex positions for all the triangles. Note that the coordinate range
    //  for a View is from -1.0 to +1.0. For each pixel to be displayed, we calculate the
    //  range it covers in this coordinate system as X to Xp1, Y to Yp1, and then set
    //  two trianges to make up the rectangle that will be used to represent the pixel.
    
    int Nv = 0;
    float Yinc = 2.0 / float(Ny);
    float Xinc = 2.0 / float(Nx);
    for (int Iy = 0; Iy < Ny; Iy++) {
        float Y = Iy * Yinc - 1.0;
        float Yp1 = Y + Yinc;
        float X = -1.0;
        float Xp1 = X + Xinc;
        
        //  This draws a zero area triangle - the first two vertices are the same - then
        //  draws the two triangles for the first pixel in the line.
        
        positions[Nv++] = {X,Y,0.0};
        positions[Nv++] = {X,Y,0.0};
        positions[Nv++] = {X,Yp1,0.0};
        positions[Nv++] = {Xp1,Y,0.0};
        positions[Nv++] = {Xp1,Yp1,0.0};
        
        //  After that, we only need to add one two vertices for each image pixel, one
        //  for each of the two triangles.
        
        for (int Ix = 1; Ix < Nx; Ix++) {
            Xp1 = (Ix + 1) * Xinc - 1;
            positions[Nv++] = {Xp1,Y,0.0};
            positions[Nv++] = {Xp1,Yp1,0.0};
         }
         
         //  And we end off each line with another zero-area triangle, just by repeating
         //  the last vertex.
         
         positions[Nv++] = {Xp1,Yp1,0.0};
    }

    //  Now set a default set of colours for each of the vertices. This sets a
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
    float NxBy2 = float(Nx) * 0.5;
    float NyBy2 = float(Ny) * 0.5;
    float MaxDistSq = NxBy2 * NyBy2;
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            float Xdist = fabs(float(Ix) - NxBy2);
            float Ydist = fabs(float(Iy) - NyBy2);
            float DistSq = Xdist * Xdist + Ydist * Ydist;
            float Grey = 1.0 - sqrt(DistSq / MaxDistSq);
            simd::float3 RGB= {Grey,Grey,Grey};
            int Vertices = 2;
            
            //  To match the colours to the vertices in the triangle strip, the first
            //  pixel for each line has 5 vertices and each subsequent pixel has 2.
            
            if (Ix == 0) Vertices = 5;
            for (int I = 0; I < Vertices; I++) { colors[Nc++] = RGB; }
        }
        
        //  Then there is one extra colour needed for at the end of each line for the
        //  final vertex that produces the terminating zero-area triangle.

        colors[Nc++] = {0.0,0.0,0.0};
    }
    _debug.Logf("Timing","Recalculated vertices & colours at %.2f msec",theTimer.ElapsedMsec());

    memcpy( _pVertexPositionsBuffer->contents(), positions, positionsDataSize );
    memcpy( _pVertexColorsBuffer->contents(), colors, colorDataSize );

    if (_useManagedBuffers) {
        _pVertexPositionsBuffer->didModifyRange(
                                          NS::Range::Make(0,_pVertexPositionsBuffer->length()));
        _pVertexColorsBuffer->didModifyRange(NS::Range::Make(0,_pVertexColorsBuffer->length()));
    }
    _debug.Logf("Timing","Copied data to renderer buffers at %.2f msec",theTimer.ElapsedMsec());

    if (positions) delete[] positions;
    if (colors) delete[] colors;

}

void Renderer::SetOverlay(float* xPosns,float* yPosns,int nPosns)
{
    if (nPosns > 0) {
        if (nPosns > _maxOverVerts) nPosns = _maxOverVerts;
        float xScale = 2.0 / _viewWidth;
        float yScale = 2.0 / _viewHeight;
        simd::float3* vBuffer = (simd::float3*)_pOverlayVertexBuffer->contents();
        simd::float3* cBuffer = (simd::float3*)_pOverlayColorsBuffer->contents();
        for (int i = 0; i < nPosns; i++) {
            float x = xPosns[i] * xScale - 1.0;
            float y = yPosns[i] * yScale - 1.0;
            float z = 0.0;
            vBuffer[i] = {x,y,z};
            cBuffer[i] = {1.0,1.0,1.0};
        }
    }
    _overVerts = nPosns;
}

void Renderer::Draw(MTK::View* pView, float* imageData )
{
    SetColourDataHistEq(imageData,_nx,_ny);

    int Nx = _nx;
    int Ny = _ny;
    //const size_t NumVertices = Nx * Ny * 2 * 3;
    const size_t NumVertices = ((Nx - 1) * 2 + 6) * Ny;

    //printf ("Draw called\n");
    MsecTimer theTimer;
    
    NS::AutoreleasePool* pPool = NS::AutoreleasePool::alloc()->init();

    MTL::CommandBuffer* pCmd = _pCommandQueue->commandBuffer();
    MTL::RenderPassDescriptor* pRpd = pView->currentRenderPassDescriptor();
    MTL::RenderCommandEncoder* pEnc = pCmd->renderCommandEncoder( pRpd );

    pEnc->setRenderPipelineState( _pPSO );
    
    pEnc->setVertexBuffer( _pVertexPositionsBuffer, 0, 0 );
    pEnc->setVertexBuffer( _pVertexColorsBuffer, 0, 1 );
    pEnc->drawPrimitives( MTL::PrimitiveType::PrimitiveTypeTriangleStrip, NS::UInteger(0),
                            NS::UInteger(NumVertices) );
    
    if (_overVerts > 0) {
        pEnc->setVertexBuffer( _pOverlayVertexBuffer, 0, 0 );
        pEnc->setVertexBuffer( _pOverlayColorsBuffer, 0, 1 );
        pEnc->drawPrimitives( MTL::PrimitiveType::PrimitiveTypeLineStrip, NS::UInteger(0),
                             NS::UInteger(_overVerts) );

    }
    pEnc->endEncoding();
    pCmd->presentDrawable( pView->currentDrawable() );
    pCmd->commit();

    pPool->release();
    
    //printf ("Draw took %.2f msec\n",theTimer.ElapsedMsec());
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
    *R = float(GrjtColourData[0][Index]) / 255.0;
    *G = float(GrjtColourData[1][Index]) / 255.0;
    *B = float(GrjtColourData[2][Index]) / 255.0;
}

/*
                                   P r o g r a m m i n g   N o t e s
 
 o   The way the Nx by Ny pixels of the image are rendered is inefficient. There are
     two triangles for each pixel, making up a square, originally meaning there were
     Nx * Ny * 2 * 3 vertices that had to be specified. Using triangle strips cuts that
     down to ((Nx - 1) * 2 + 6) * Ny (see BuildBuffers() comments for just where that
     comes from. It may be that using just two triangles to form a rectangle covering
     the whole image and using a texture would be even faster, but in any case, the time
     taken for the rendering is generally small compared to that required for the computation.

    o   For Apple's basic documentation on the render pipeline, see:
        https://developer.appl
            e.com/documentation/metal/using_a_render_pipeline_to_render_primitives?language=objc

 */
