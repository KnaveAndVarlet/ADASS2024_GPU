//  ------------------------------------------------------------------------------------------------
//
//                                 M a n d e l  A r g s . h    ( Metal )
//
//  This provides a way of dealing with the command line arguments for the Metal version of
//  the Mandelbrot demonstration program. The Vulkan version is all in C++, and it is easy
//  enough for the main C++ code to handle the command line and pass the parameter values on
//  to the controller. However, in the Metal version the lines of communication between the
//  main routine (which is passed the argc,argv command line values) and the controller is
//  rather more convoluted and in a mixture of Objective-C, Objective-C++ and C++. This provides
//  two C-callable routines: the first - ParseMandelArgs() - is called from main() and extracts
//  the argument values from the command line (interacting with the user as needed), while the
//  second - GetMandelArgs() - can be called from the Controller, and returns the argument
//  values. Storing the argument values privately, but shared betwen the two routines, removes
//  any need to tediously pass the results all the way down the chain from main() to Controller.

//  The routines need to be called from Objective-C (which can call C) and also from C++, so
//  need to be called with C naming conventions, not those of C++. And this .h file needs to be
//  able to be included by both Obj-C and C++ code files.

#ifdef __cplusplus
extern "C" {
#endif

//  Define the layout of the global structure used to hold the value of the command line
//  arguments.

typedef struct {
    int Nx;
    int Ny;
    int Iter;
    char Debug[256];
} MandelArgs;

//  Parse the command line arguments and save their values.
int ParseMandelArgs (int Argc,char* Argv[]);

//  Return a pointer to the structure where the argument values were saved.
MandelArgs* GetMandelArgs (void);

#ifdef __cplusplus
}
#endif


