//
//                   m a n d e l . m e t a l
//
//  This is the Metal GPU code used by the Mandel GPU demonstration
//  program. It calculates the Mandelbrot value for a single pixel
//  in an image, getting its pixel coordinates from the position
//  in the grid corresponding to the current thread. It combines
//  the pixel coordinate with the arguments passed in the constant
//  structure to get the coordinates of that pixel in Mandelbrot
//  space. It then iterates the standard Mandelbrot calculation
//  to see if it diverges. The value it associates with the pixel
//  is the number of iterations it takes for the calculation to
//  diverge. If it does not diverge in less than the maximum
//  number of iterations specified in the arguments, it assumes
//  the pixel is in the Mandelbrot set and sets its value to zero.
//  The value is then set into the appropriate pixel of the
//  output array specified as buffer(1). The argument structure
//  is assumed to be buffer(2).

//  (This code was originally based on the example code in the
//  LearnMetalCPP example linked to on developer.apple.com/metal/
//  but has been modified to write into an array in memory rather
//  than a texture and to fit in with the needs of the Mandel
//  demonstration program.

#include <metal_stdlib>
using namespace metal;

struct MandelArgs {
    float xCent;      // X-coordinate of the array center
    float yCent;      // Y-coordinate of the array center
    float dX;         // Change in X-coordinate per pixel
    float dY;         // Change in Y-coordinate per pixel
    int iter;         // Maximum number of iterations
};

kernel void mandel(device float *out [[ buffer(1) ]],
                   constant MandelArgs *args [[buffer(2)]],
                   uint2 index2 [[thread_position_in_grid]],
                   uint2 gridSize [[threads_per_grid]]) {
    
    //  Work out the Mandelbrot coordinate of the center of
    //  the pixel in question.
    
    float gridXcent = gridSize.x * 0.5;
    float gridYcent = gridSize.y * 0.5;
    float x0 = args->xCent + (index2.x - gridXcent) * args->dX;
    float y0 = args->yCent + (index2.y - gridYcent) * args->dY;

    //  Perform the standard Mandelbrot calculation and see if
    //  and when it is evident that it diverges.
    
    float x = 0.0;
    float y = 0.0;
    uint iteration = 0;
    uint max_iteration = args->iter;
    float xtmp = 0.0;
    while(((x * x) + (y * y) <= 4.0) && (iteration < max_iteration))
    {
        xtmp = (x + y) * (x - y) + x0;
        y = (2.0 * x * y) + y0;
        x = xtmp;
        iteration += 1;
    }

    //  Set the element of the output array to the iteration count,
    //  unless it is in the Mandelbrot set (in which case the
    //  iteration count will have exceeded the maximum limit), in
    //  which case we set it to zero.
    
    float value = iteration;
    if (iteration >= max_iteration) value = 0.0;
    uint index = index2.y * gridSize.x + index2.x;
    out[index] = value;
}
