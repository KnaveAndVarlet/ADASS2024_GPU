#include <metal_stdlib>
using namespace metal;

kernel void adder(device float *in [[ buffer(1) ]],
                   device float *out[[ buffer(2) ]],
                   uint2 index2 [[thread_position_in_grid]],
                   uint2 gridSize [[threads_per_grid]]) {
    
    uint index = index2.y * gridSize.x + index2.x;
    out[index] = in[index] + index2.y + index2.x;
}
