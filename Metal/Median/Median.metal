//
//                          M e d i a n . m e t a l
//
//  This Metal code passes a median filter through a 2D image.
//  A conceptual box is drawn around each element of the image,
//  and the corresponding element of the output image is set
//  to the median value in the box.
//
//  Note the magic invocation to build this:
//  xcrun -sdk macosx metal -c median.metal -o median.air
//  xcrun -sdk macosx metallib median.air -o compute.metallib
//
//  Modified:
//     27th Oct 2024. Comments expanded. KS.

#include <metal_stdlib>
using namespace metal;

//  A MedianArgs structure is used to pass the dimensions of the
//  filter box. This allows for a rectangular box, with different
//  X and Y dimensions, but in fact this code uses a square box
//  Npix by Npix and sets Npix to the specified xsize value.

struct MedianArgs {
    uint xsize;
    uint ysize;
};

// Allow for values of Npix up to 11.

#define NPIXSQ_MAX 121

//  This finds the median value in the array X, without recursion,
//  using a quickselect algorithm which partially sorts X.

float CalcMedian(float X[NPIXSQ_MAX], uint len)
{
   #define swap(a,b) {float t = X[a]; X[a] = X[b], X[b] = t;}
   uint left = 0, right = len - 1;
   uint cent = len/2;
   float pivot;
   uint pos, i;
   float median = 0.0;
  
   while (left < right) {
      pivot = X[cent];
      swap(cent, right);
      for (i = pos = left; i < right; i++) {
         if (X[i] < pivot) {
            swap(i, pos);
            pos++;
         }
      }
      swap(right, pos);
      if (pos == cent) break;
      if (pos < cent) left = pos + 1;
      else right = pos - 1;
   }
   median = X[cent];
   
   //  If the array has an odd number of elements, X[cent] will be the median
   //  value, with all values greater than it in the upper partition and all
   //  values less than it in the lower partition. (But no more sorted than that.)
   //  If the array has an odd number of elements, we want the average of the two
   //  central elements. X[cent] will be the higher of the two - remember that
   //  the first element is X[0] - so we look for the highest value in the
   //  lower part of the array and return the average of it and X[cent].
    
   if ((len % 2) == 0) {
      float temp = X[cent - 1];
      for (uint i = 0; i < cent - 1; i++) {
         if (X[i] > temp) temp = X[i];
      }
      median = (median+temp)*0.5;
   }

   return median;
}

kernel void Median(const device float *inputImage [[ buffer(0) ]],
                   device float  *outputImage [[ buffer(1) ]],
                   constant MedianArgs *args [[buffer(2)]],
                   uint2 index2 [[thread_position_in_grid]],
                   uint2 gridSize [[threads_per_grid]])
{
  uint ix = index2.x;
  uint iy = index2.y;
  uint ny = gridSize.y;
  uint nx = gridSize.x;
  
  //  In order to fit the work into workgroups, some unnecessary threads are launched.
  //  We terminate those threads here. (Metal doesn't need this, but Vulkan does.)
  
  // if(ix >= nx || iy >= ny) return;

  //  Fill a work array with the input array elements in a box npix wide around the
  //  target element. Allow for the edges of the image.
  
  uint npix = args->xsize;
  while (npix * npix > NPIXSQ_MAX) npix--;
  float work[NPIXSQ_MAX];
  int npixby2 = int(npix) / 2;
  int ixmin = int(ix) - npixby2;
  int ixmax = int(ix) + npixby2;
  int iymin = int(iy) - npixby2;
  int iymax = int(iy) + npixby2;
  if (ixmin < 0) ixmin = 0;
  if (ixmax >= int(nx)) ixmax = int(nx - 1);
  if (iymin < 0) iymin = 0;
  if (iymax >= int(ny)) iymax = int(ny - 1);
  uint ipix = 0;
  for (int yind = iymin; yind <= iymax; yind++) {
     for (int xind = ixmin; xind <= ixmax; xind++) {
        work[ipix++] = inputImage[yind * nx + xind];
     }
  }
    
  float median = CalcMedian(work,ipix);

  outputImage[nx * iy + ix] = median;
}

/*                           P r o g r a m m i n g   N o t e s
 
    o   Because you can't allocate dynamically sized arrays, this code has to use a fixed
        size work array. This takes up resources in the shader - each element is probably
        a register - and it also sets a quite small limit on the size of the median box.
        I have experimented with a version where the shader is passed a huge local buffer
        - one with (nx * ny * npix * npix) elements, enough for each thread to have its
        own npix*npix part of the buffer to work with. This works, but is noticeably slower
        than this code with a small local work array, presumably because buffer access is
        going to be much slower than register access.
        
*/
