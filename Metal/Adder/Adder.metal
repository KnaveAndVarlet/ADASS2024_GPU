//
//                               A d d e r . m e t a l
//
//  The compute shader code for the Metal version of the Adder GPU example program. The GPU is
//  passed two 2D arrays, one input and one output. It sets each element of the output array
//  to the value of the corresponding element of the input array plus the sum of its row and
//  column index values.

#include <metal_stdlib>
using namespace metal;

kernel void adder(device float *in [[ buffer(1) ]],
                   device float *out[[ buffer(2) ]],
                   uint2 index2 [[thread_position_in_grid]],
                   uint2 gridSize [[threads_per_grid]]) {

    //  The calculation performed is very simple - we add ix+iy to each input element of
    //  the array and store it in the output array. Note we have to work out the index into
    //  the buffer - we can't access it as an element of a 2D array.
    
    uint nx = gridSize.x;
    uint ix = index2.x;
    uint iy = index2.y;
    out[iy * nx + ix] = in[iy * nx + ix] + iy + ix;
}
