//  ------------------------------------------------------------------------------------------------

//                              M t k  P r i v a t e . c p p
//
//  This is a rather odd-looking program, but it does generate obkect code, and it - or
//  something like it) is needed when metal-cpp is used. The trick is that metal-cpp is
//  a header-only implementation, being supplied just as a collection of .hpp files.
//  However, some actual metal-cpp routines need to be included in the link of any code
//  that uses metal-cpp. Rather than require a metal-cpp object library, metal-cpp uses
//  a scheme where if the .hpp files are included with the 'PRIVATE' variables below
//  defined for the pre-processor, then these routines will be included in the code and
//  compiled and can be linked from the resulting .o file. Obviously, if more than one
//  .cpp file does this, these routines will be included in more than one .o file and
//  the linker will complain. So the metal-cpp documentation requires that only one
//  code file that includes the .hpp files define these PRIVATE variables. The easiest
//  way to do this - it seems to me - is to have one file (this one) that does this and
//  nothing more and to forget about this metal-cpp quirk in all the other code files.
//  So here it is, and it needs to be compiled and included in the build. (If you miss
//  it out, you get a load of undefined symbol messages for symbols with unlikely names
//  like "NS::Private::Class::s_kNSString".


#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"
#include "MetalKit/MetalKit.hpp"
