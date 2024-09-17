//
//                     A d d e r . c p p     ( M e t a l  V e r s i o n )
//
//  This program is intended as a piece of example code showing a calculation involving two
//  2D floating point arrays performed on a GPU, programmed using in C++ using Metal. It
//  allows the user to perform the same operation using the CPU and compare the timings,
//  and allows some experimentation with the number of CPU threads to be used. The size of
//  the arrays is set using command line parameters. Usually the operation needs to be
//  repeated a number of times to get a reasonable execution time.
//
//  The problem chosen is a trivial one: given an 2D array, add to each element the sum of its
//  two indices (in X and Y) and return the result in a second, similarly sized array. See
//  example code below for details.
//
//  Invocation:
//     Adder <Nx> <Ny> <Nrpt> <Threads> <cpu> <gpu> <debug>
//
//  Where:
//
//     Nx      is the X dimension of the arrays in question.
//     Ny      is the Y dimension of the arrays in question.
//     Nrpt    is the number of times the operation is to be repeated.
//     Threads if the CPU is used, the number of CPU threads to be used.
//
//     Nx,Ny,Nrpt,Threads are positional parameters, and can be specified either by just
//     providing values for them on the command line in the order above, or explicitly
//     by name and value, with an optional '=' sign. If Threads is zero, the maximum number
//     of CPU threads available will be used.
//
//     Cpu     specifies that the operation is to be carried out using the CPU.
//     Gpu     specifies that the operation is to be carried out using the GPU.
//
//     Cpu and Gpu are boolean values that can be specified explicitly, eg Cpu = true,
//     or just by appearing in the command line, optionally negated, for example as 'cpu'
//     or 'nogpu'. They are not mutually exclusive. By default both are false, but if neither
//     are specified the program will use the GPU.
//
//     Debug    is a string that can be used to control debug output.
//
//     Debug must be specified explicitly by name, eg Debug = "timing". The '=' is optional.
//     There are a number of different options for Debug. If you are prompted for the value
//     of Debug and reply with '?' a list of the options will be provided.
//
//     The command line is processed by the flexible but possibly quirky command line handler
//     used for all these GPU examples. With luck you'll get used to it. It also supports the
//     command line flags 'list' (lists all the parameter values that are going to be used),
//     'prompt' (explicitly prompts for all unspecified parameters), 'reset' (resets all
//     parameter values to their default - this is useful, because it remembers the values
//     used the last time the program was run) and 'help' which outputs a summary and brief
//     description of the various parameters.
//
//  Example:
//     ./Adder 1024 1024 1
//     ./Adder nrpt 1 nx = 1024 ny 1024       has exactly the same effect
//
//  Note:
//      The effect is like that of the following C code, if Nx and Ny give the image dimensions:
//
//      float In[Ny][Nx],Out[Ny,Nx];
//      for (int Iy = 0; Iy < Ny; Iy++) {
//          for (int Ix = 0; Ix < Nx; Ix++) {
//              Out[Iy][Ix] = In[Iy][Ix] + Ix + Iy;
//          }
//      }
//
//      (But note that in practice such a program will only work with very small values of Nx and
//      Ny, as the arrays will be allocated on the stack, and also standard C++ requires that Nx
//      and Ny should be constants. A more practical example would allocate arrays on the heap -
//      and would set the elements of In to sensible initial values.)
//
//  Acknowledgements:
//      This code had its origins in a program in Swift that was itself based on some example
//      Swift code for computation using Metal at:
//      gist.github.com/mhamilt/a5c2bbb02684e5db362712c9be7a02ca
//      That Swift code was then mangled to become code for calculating a Mandlebrot image in a
//      2D array and then reworked for C++ using Apple's metal-cpp subsystem. This then became
//      further re-worked to become the original trivial Metal example. By now it bears almost
//      no resemblance to the original example, but I'm grateful for the start it gave me.
//
//  History:
//       3rd Jul 2024. First fully commented version. KS.
//      17th Jul 2024. CPU code wasn't initialising the input array properly. Fixed. KS.
//      15th Aug 2024. Brought up to date to match the latest Vulkan version. Added a few
//                     'metal' diagnostics. KS.
//       2nd Sep 2024. Re-ordered some of the GPU setup and diagnostics to better match the
//                     Vulkan version. KS.
//      10th Sep 2024. Further slight tweaks to disgnostics. KS.

//  ------------------------------------------------------------------------------------------------
//
//                                 I n c l u d e  F i l e s
//
//  Needed to allow multi-threading of the CPU code

#include <thread>

//  Some utility code. A Command Handler provides a flexible way of handling command line
//  parameters, and simplifies the coding required for this. An MsecTimer provides a very simple
//  way of timing blocks of code. A Debug Handler provides control over debug output, allowing
//  various debug levels to be enabled from the command line.

#include "CommandHandler.h"
#include "MsecTimer.h"
#include "DebugHandler.h"

//  This provides a global Debug Handler that all the routines here can use.

DebugHandler TheDebugHandler;

//  ------------------------------------------------------------------------------------------------
//
//                             F o r w a r d  D e f i n i t i o n s

//  Perform the basic opetation using the GPU
void ComputeUsingGPU(int Nx,int Ny,int Nrpt);
//  Perform the basic operation using the CPU
void ComputeUsingCPU(int Threads,int Nx,int Ny,int Nrpt);
//  Set initial values for the input array.
void SetInputArray(float** InputArray,int Nx,int Ny);
//  Check the results of the operation
bool CheckResults(float** InputArray,int Nx,int Ny,float** OutputArray);
//  Utility to set up an array of row addresses to allow use of Array[Iy][Ix] syntax for access.
float** CreateRowAddrs(float* Array,int Nx,int Ny);

//  ------------------------------------------------------------------------------------------------
//
//                             D e b u g  A r g  H e l p e r
//
//  This provides a way for the program to give some additional help to the command line handler
//  for the 'Debug' parameter, which has a complex syntax, being a comma-separated list of
//  hierarchical options with support for wildcards. It's all very well making these things
//  flexible, but then thay take much more explaining...

class DebugArgHelper : public CmdArgHelper {
public:
    bool CheckValidity(const std::string& Value,std::string* Reason);
    std::string HelpText(void);
};

//  ------------------------------------------------------------------------------------------------
//
//                                    M a i n
//
//  Most of the code in the main routine is taken up with getting the values for the various
//  command line parameters. It then invokes either the CPU or the GPU (or both) to perform the
//  relatively trivial 'adder' operation as specified by those parameters.

int main (int Argc, char* Argv[]) {

    //  Set up the various levels for the Debug handler

    TheDebugHandler.LevelsList("Timing,Setup,Metal");
   
    //  Get the values of the command line arguments. This uses a Command Handler class that
    //  provides a flexible way of dealing with a number of different arguments, for example
    //  by allowing specifications such as 'Nx = 1024 Ny = 512". Each argument is handled by
    //  an instance of an argument class (eg NxArg, or DebugArg), and the Command Handler
    //  coordinates how these handle the actual command line arguments (argc and argv).
    
    //  Set up the Handler and the argument instances for the various command line arguments.
    
    CmdHandler TheHandler("Adder");
    IntArg NxArg(TheHandler,"Nx",1,"",1024,2,1024*1024,"X-dimension of computed image");
    IntArg NyArg(TheHandler,"Ny",2,"",1024,2,1024*1024,"Y-dimension of computed image");
    IntArg NrptArg(TheHandler,"Nrpt",3,"",1,0,1000000,"Repeat count for operation");
    int DefaultThreads = 1;
    int MaxThreads = std::thread::hardware_concurrency();
    if (MaxThreads < 0) { MaxThreads = 0; DefaultThreads = 0; }
    IntArg ThreadsArg(TheHandler,"Threads",4,"",DefaultThreads,0,MaxThreads,"CPU threads to use");
    BoolArg CpuArg(TheHandler,"Cpu",0,"",false,"Perform computation using CPU");
    BoolArg GpuArg(TheHandler,"Gpu",0,"",false,"Perform computation using GPU");
    StringArg DebugArg(TheHandler,"Debug",0,"NoSave","","Debug levels");
    DebugArgHelper DebugHelper;
    DebugArg.SetHelper(&DebugHelper);

    //  Now get the values for each of the command line arguments. The ReadPrevious() and
    //  SaveCurrent() calls allow the handler to remember the argument values used the last
    //  time the program was run. If you don't care for that, comment them out.
    
    std::string Error = "";
    if (TheHandler.IsInteractive()) TheHandler.ReadPrevious();
    bool Ok = TheHandler.ParseArgs(Argc,Argv);
    int Nx = NxArg.GetValue(&Ok,&Error);
    int Ny = NyArg.GetValue(&Ok,&Error);
    int Nrpt = NrptArg.GetValue(&Ok,&Error);
    int Threads = ThreadsArg.GetValue(&Ok,&Error);
    bool UseCPU = CpuArg.GetValue(&Ok,&Error);
    bool UseGPU = GpuArg.GetValue(&Ok,&Error);
    std::string DebugLevels = DebugArg.GetValue(&Ok,&Error);
    
    //  Check the argument parsing went OK, and didn't end with an exit being requested.
    
    if (!Ok) {
        if (!TheHandler.ExitRequested()) {
            printf ("Error parsing command line: %s\n",TheHandler.GetError().c_str());
        }
    } else {
        if (TheHandler.IsInteractive()) TheHandler.SaveCurrent();

        //  Set the active levels for the Debug handler as specified on the command line.
        
        TheDebugHandler.SetLevels(DebugLevels);
        
        printf ("\nPerforming 'Adder' test, arrays of %d rows, %d columns. Repeat count %d.\n\n",
                Ny,Nx,Nrpt);

        //  If neither CPU not GPU were specified on the command line, use GPU.
        
        if (!UseGPU && !UseCPU) UseGPU = true;
        
        //  Perform the test using either CPU or GPU (or both). Both ComputeUsingGPU() and its CPU
        //  equivalent, ComputeUsingCPU() are expected to create arrays of the size specified, call
        //  SetInputArray() to initialise the input array, then perform the basic 'adder' operation
        //  as specified, and then call CheckResults() to verify that they got the right answer.
        
        if (UseGPU) ComputeUsingGPU(Nx,Ny,Nrpt);
        
        if (UseCPU) ComputeUsingCPU(Threads,Nx,Ny,Nrpt);
    }
    return 0;
}

//  ------------------------------------------------------------------------------------------------
//
//                                    G P U  c o d e
//
//  This routine performs the 'Adder' operation using the GPU, and is coded using Apple's
//  metal-cpp Metal interface. It's convenient to use this, particularly for anyone who
//  knows C++ and would prefer not to have to use Apple's preferred Objective-C or Swift.
//  However, metal-cpp doesn't expose all that Metal can do, and - possibly more awkwardly -
//  Apple's on-line documentation for Metal allows you to switch between Objective-C and
//  Swift descriptions of the routines, but metal-cpp is less well documented.
//
//  This routine is passed the dimensions, Nx and Ny of the arrays to use. It creates an array
//  of floats, Nx by Ny, and calls SetInputArray() to initialise it. It is also passed the repeat
//  count for the operation (Nrpt). It also has to create an output array of the same size and
//  perform the basic 'adder' operation to set the output array values based on those of the
//  input array. It has to perform that operation the number of times specified in Nrpt. Clearly,
//  repeating the operation should have no effect on the final contents of the output array, and
//  is simply in order to get a better idea of the timing.

//                             M e t a l  I n c l u d e  F i l e s
//
//  A quirk of metal-cpp is that one (and only one) piece of code that includes the Metal headers
//  must define the 'private implementation' symbols. See Programming notes at the end of this
//  file for the tedious details.

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

//  Needed for Metal

#include <Metal/Metal.hpp>
#include <AppKit/AppKit.hpp>
#include <MetalKit/MetalKit.hpp>

//  Metal-cpp sometimes lets its underlying implementation in objective-C show through. One is
//  its use of NS::String objects, and it helps to not have to keep writing "NS::StringEncoding::".

using NS::StringEncoding::UTF8StringEncoding;

void ComputeUsingGPU(int Nx,int Ny,int Nrpt)
{
    //  This is where we actually start to use Metal, specifically the metal-cpp layer provided
    //  by Apple for use with C++.
    
    MsecTimer SetupTimer;
    TheDebugHandler.Log("Setup","GPU setup starting");

    //  The metal-cpp routines need an autorelease pool they can use to keep track of
    //  allocated items. This gives them what they need. All we need to do is create it.
    
    NS::AutoreleasePool* MainAutoreleasePool = NS::AutoreleasePool::alloc()->init();
    
    //  We create an MTL::Device object to represent the GPU itself. (If a system has multiple
    //  GPUs this can get more complicated, but most Apple systems only have one GPU, and the
    //  default device is usually what we need.)
    
    MTL::Device* Device = MTLCreateSystemDefaultDevice();
    TheDebugHandler.Logf("Setup","GPU device created at %.3f msec",SetupTimer.ElapsedMsec());
    TheDebugHandler.Logf("Metal","Device is '%s'",Device->name()->cString(UTF8StringEncoding));

    //  We need to get the 'adder' function from the library into which it's been compiled from
    //  the code in Adder.metal. This is one area of this code that might go wrong, for example
    //  if the library is not present, or if it doesn't contain the function we want, so we
    //  test for both those possibilities.
    
    NS::Error* ErrorPtr = nullptr;
    MTL::Function* AdderFunction = nullptr;
    MTL::Library* Library = Device->newLibrary(NS::String::string("Compute.metallib",
                                                       UTF8StringEncoding),&ErrorPtr);
    if (Library == nullptr || ErrorPtr != nullptr) {
        printf ("Error opening library 'Compute.metallib'.\n");
        if (ErrorPtr) {
            printf ("Reason: %s\n",ErrorPtr->localizedDescription()->cString(UTF8StringEncoding));
        }
    } else {
        TheDebugHandler.Logf("Setup","GPU library created at %.3f msec",
                                                                   SetupTimer.ElapsedMsec());
        AdderFunction = Library->newFunction(NS::String::string("adder",UTF8StringEncoding));
        if (AdderFunction == nullptr) printf ("Unable to find 'adder' function in library\n");
    }
    
    //  If we've got the Adder function set up, things are probably going to work. Because this
    //  is intended as example code rather than real production code, I don't want to clutter
    //  it too much with error handling from here on - production code should be more careful!
    
    if (AdderFunction) {
        TheDebugHandler.Logf("Setup","GPU adder function created at %.3f msec",
                                                                   SetupTimer.ElapsedMsec());

        //  To make things easier for the CPU code, we will create an array of the addresses of
        //  the rows of the input buffer - this allows the use of an ArrayRows[Iy][Ix] syntax when
        //  accessing it. We will do the same for the output array. We set the addresses of these
        //  arrays null now so we can check if they need freeing at the end of the routine.
        
        float** InputArray = nullptr;
        float** OutputArray = nullptr;

        //  Create a device buffer to contain the input data array. The allocation size we request
        //  needs to be padded to a multiple of the page size. This buffer is not a transient item,
        //  so it can be set up once now and the GPU can use it later on multiple occasions - ie
        //  in the repeat loop. We create it using the variant of the MTLDevice::newBuffer() call
        //  that uses the Obj-C routine newBufferWithLength::options. The options specify that the
        //  buffer is to be 'shared', ie visible to both the GPU and the CPU (since we need to
        //  use the CPU to set its initial values.
        
        int Length = Nx * Ny * sizeof(float);
        unsigned int Alignment = sysconf(_SC_PAGE_SIZE);
        int AllocationSize = (Length + Alignment - 1) & (~(Alignment - 1));
        uint BufferOptions = MTL::StorageModeShared;
        MTL::Buffer* InputBuffer = Device->newBuffer(AllocationSize,BufferOptions);
        
        //  To set the contents of the buffer using the CPU, we need the address the CPU can use
        //  for this buffer, which we get using its contents() method. Then we can initialise
        
        InputArray = CreateRowAddrs((float*)InputBuffer->contents(),Nx,Ny);
        SetInputArray(InputArray,Nx,Ny);

        //  And now a device buffer for the output data array. This is essentially the same as for
        //  the input buffer. This will have to be accessed on the CPU side by CheckResults(),
        //  so we use CreateRowAddrs() to set up for this.
        
        MTL::Buffer* OutputBuffer = Device->newBuffer(AllocationSize,BufferOptions);
        OutputArray = CreateRowAddrs((float*)OutputBuffer->contents(),Nx,Ny);
        TheDebugHandler.Logf("Setup","GPU buffers created at %.3f msec",SetupTimer.ElapsedMsec());

        //  We need a command queue that will be able to supply a command buffer.
        
        MTL::CommandQueue* CommandQueue = Device->newCommandQueue();
        TheDebugHandler.Logf("Setup","GPU command queue created at %.3f msec",
                                                                   SetupTimer.ElapsedMsec());

        //  This lets us create a 'pipeline state' that will execute this function.
        
        MTL::ComputePipelineState* PipelineState =
        Device->newComputePipelineState(AdderFunction,&ErrorPtr);
        TheDebugHandler.Logf("Setup","GPU pipeline state created at %.3f msec",
                                                                  SetupTimer.ElapsedMsec());

        //  Now we need to work out the way threads in the GPU will be allocated to the 
        //  computation. We need to set up the grid the GPU will use - the kernel code can
        //  get the grid dimensions and will use that to get the size of the arrays - it needs
        //  to know this, so we set it simply to Nx by Ny - the GPU can handle a third dimension
        //  but we set that to 1. This code simply follows the general guidelines in the Apple
        //  documentation.

        int ThreadGroupSize = PipelineState->maxTotalThreadsPerThreadgroup();
        int ThreadWidth = PipelineState->threadExecutionWidth();
        TheDebugHandler.Logf("Metal","Max threads per threadgroup %d, Thread width %d",
                                                                ThreadGroupSize,ThreadWidth);

        if (ThreadGroupSize > (Nx * Ny)) ThreadGroupSize = Nx * Ny;
        MTL::Size GridSize(Nx,Ny,1);
        MTL::Size ThreadGroupDims(ThreadGroupSize / ThreadWidth,ThreadWidth,1);
        TheDebugHandler.Logf("Setup","Thread group dimensions %d, %d, %d",
                                           ThreadGroupSize / ThreadWidth,ThreadWidth,1);
        TheDebugHandler.Logf("Setup","Grid size %d, %d, %d",
                                                GridSize.width,GridSize.height,GridSize.depth);
        TheDebugHandler.Logf("Setup","GPU setup took %.3f msec",SetupTimer.ElapsedMsec());

        //  And that completes the basic setup for the GPU pipeline. From here, all Metal items
        //  are transient, and need to be created anew if the calculation is to be repeated,
        //  which it is going to be in the loop that follows.
                
        MsecTimer ComputeTimer;
        
        for (int Irpt = 0; Irpt < Nrpt; Irpt++) {
            
            MsecTimer LoopTimer;
            
            //  This sets up the pipeline for the GPU calculation and runs it. It's good
            //  practice to use a separate autorelease pool for separate sections like this.
            //  (Actually, in this case, the main pool could be left quite happily to do the job.)
            
            NS::AutoreleasePool* PipeAutoreleasePool = NS::AutoreleasePool::alloc()->init();
                        
            //  Create a command buffer, a command encoder, and set the compute shader kernel
            //  that will perform the calculation.
            
            MTL::CommandBuffer* CommandBuffer = CommandQueue->commandBuffer();
            MTL::ComputeCommandEncoder* Encoder = CommandBuffer->computeCommandEncoder();
            TheDebugHandler.Logf("Timing","Command buffer and encoder created at %.3f msec",
                                                                     LoopTimer.ElapsedMsec());
            Encoder->setComputePipelineState(PipelineState);

            //  Set the two data buffers, for the input array and the output array. These are
            //  associated with different bindings, specified by the index numbers 1 and 2, which
            //  must match the index values used in the metal kernel code in Adder.metal
            
            Encoder->setBuffer(InputBuffer,0,1);
            Encoder->setBuffer(OutputBuffer,0,2);
            TheDebugHandler.Logf("Timing","Data buffers set at %.3f msec",LoopTimer.ElapsedMsec());

            Encoder->dispatchThreads(GridSize,ThreadGroupDims);
            Encoder->endEncoding();
            TheDebugHandler.Logf("Timing","Encoding finished at %.3f msec",LoopTimer.ElapsedMsec());
            
            //  Finally, we run the kernel. We 'commit' the command buffer, ie submit it
            //  for execution, and wait for it to complete. (A cleverer program could be doing
            //  something else while the buffer executes, but we just wait.)
            
            CommandBuffer->commit();
            TheDebugHandler.Logf("Timing","Compute committed at %.3f msec",LoopTimer.ElapsedMsec());
            CommandBuffer->waitUntilCompleted();
            TheDebugHandler.Logf("Timing","Compute complete at %.3f msec",LoopTimer.ElapsedMsec());

            //  And at the end of the block, let the auto-release pool for the loop do its thing.
            
            PipeAutoreleasePool->release();
        }
        
        //  Check that we got it right, and if so report on the timing.
        
        float Msec = ComputeTimer.ElapsedMsec();
        if (CheckResults(InputArray,Nx,Ny,OutputArray)) {
            printf ("GPU completed OK.\n");
            printf ("GPU took %.3f msec\n",Msec);
            printf ("Average msec per iteration for GPU = %.3f\n\n",Msec / float(Nrpt));
        }
                        
        //  Release the arrays used to hold the row addresses for the two arrays.
        
        if (OutputArray) free(OutputArray);
        if (InputArray) free(InputArray);
    }
    
    //  And now finish up by releasing any resources known to the main auto-release pool.
    
    MainAutoreleasePool->release();
}

//  ------------------------------------------------------------------------------------------------
//
//                                    C P U  c o d e
//
//  This code performs the 'Adder' operation using the CPU, and is designed to do exactly what
//  ComputeUsingGPU() does, but using the CPU. Just like ComputeUsingGPU(), it is passed the
//  dimensions, Nx and Ny of the arrays to use. It creates an array of floats, Nx by Ny, and
//  calls SetInputArray() to initialise it. It also has to create an output array of the same
//  size and perform the basic 'adder' operation to set the output array values based on those
//  of the input array. It is also passed the repeat count for the operation (Nrpt) and has to
//  perform that operation the number of times specified in Nrpt.

//  The basic operation is a really trivial piece of CPU code, shown here in ComputeRangeUsingCPU()
//  and you could do the whole thing just with one call to ComputeRangeUsingCPU() with the Y-range
//  parameters Iyst and Iyen set to 0 and Ny respectively. That would handle the whole image using
//  one CPU thread. And for something this simple, that's probably the fastest thing to do anyway.
//  However, just to allow some experimentation with CPU multi-threading, the main routine actually
//  calls OnePassUsingCPU() for each iteration. This then handles the threading, being able to
//  split the job up so that multiple threads each handle a subset of the rows of the image (ie
//  splitting up in Y). It starts up a number of threads, each running ComputeRangeUsingCPU()
//  over a different set of image rows.
//
//  In practice, for most sensible image sizes, the overheads of starting up multiple threads
//  will more than soak up the gain from multi-threading the actual operation. But this at lets
//  you play with this. (Also see Programming notes at the end of this code.)

//  ComputeUsingCPU() is the routine called from the main line code, but the actual work is
//  done by ComputeRangeUsingCPU(), which processes a range of rows, from Iyst to Iyen (starting
//  at 0), all in a single thread.

void ComputeRangeUsingCPU(float** InputArray,int Nx,int Iyst,int Iyen,float** OutputArray)
{
    for (int Iy = Iyst; Iy < Iyen; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            OutputArray[Iy][Ix] = InputArray[Iy][Ix] + float(Ix + Iy);
        }
    }
}

//  OnePassUsingCPU() performs one pass through the whole of the input data, splitting up
//  the work across multiple threads. It will use as many CPU threads as are available, up
//  to the value of Threads. If Threads is set to zero, it uses all available CPU threads.

int OnePassUsingCPU(int Threads,float** InputArray,int Nx,int Ny,float** OutputArray)
{
    //  If we're only using one thread, do the calculation in the main thread, avoiding any
    //  threading overheads.
    
    if (Threads == 1) {
        ComputeRangeUsingCPU(InputArray,Nx,0,Ny,OutputArray);
    } else {
        
        //  If threading, create the specified number of threads, and divide the image rows
        //  between them.
        
        std::thread ThreadList[Threads];
        int Iy = 0;
        int Iyinc = Ny / Threads;
        for (int IThread = 0; IThread < Threads; IThread++) {
            ThreadList[IThread] = std::thread (ComputeRangeUsingCPU,
                                               InputArray,Nx,Iy,Iy+Iyinc,OutputArray);
            Iy += Iyinc;
        }
        
        //  Wait for all the threads to complete.
        
        for (int IThread = 0; IThread < Threads; IThread++) {
            ThreadList[IThread].join();
        }
        
        //  If there were rows left unhandled (because Ny was not a multiple of NThreads),
        //  finish them off in the main thread.
        
        if (Iy < Ny) {
            ComputeRangeUsingCPU(InputArray,Nx,Iy,Ny,OutputArray);
        }
    }
    return Threads;
}

void ComputeUsingCPU(int Threads,int Nx,int Ny,int Nrpt)
{
    //  Create the two arrays we need, one for the input data, one for the output. To make things
    //  easier for ourselves, setup two arrays that contain the addresses of the start of the
    //  rows for the two arrays. This lets us use an InputArray[Iy][Ix] syntax instead of
    //  having to calculate offsets as ArrayData[Iy * Nx + Iy]. It means that InputArray is not
    //  actually the address of the allocated memory for the array itself. It's the address of
    //  the array of row addresses - this is the trick used by the C version of Numerical Methods,
    //  and I've always found it very effective.
    
    float* InputArrayData = (float*)malloc(Nx * Ny * sizeof(float));
    float** InputArray = CreateRowAddrs(InputArrayData,Nx,Ny);
    float* OutputArrayData = (float*)malloc(Nx * Ny * sizeof(float));
    float** OutputArray = CreateRowAddrs(OutputArrayData,Nx,Ny);
    TheDebugHandler.Log("Setup","CPU arrays created");
    
    //  Initialise the input array;
    
    SetInputArray(InputArray,Nx,Ny);

    //  See how many threads the hardware supports. If Threads was passed as zero, use this
    //  maximum number of threads. Otherwise use the number passed in Threads, if that many
    //  are available.
    
    int MaxThreads = std::thread::hardware_concurrency();
    if (MaxThreads <= 0) MaxThreads = 1;
    if (Threads <= 0) Threads = MaxThreads;
    if (Threads > MaxThreads) Threads = MaxThreads;
    TheDebugHandler.Logf("Setup","CPU using %d threads out of maximum of %d\n",Threads,MaxThreads);
    
    MsecTimer ComputeTimer;
    
    //  Repeat a single pass through the whole image, as many times as specified by the repeat
    //  count. OnePassUsingCPU is passed the number of threads specified (with zero meaning
    //  use as many as are available) and returns the number actually used.
    
    for (int Irpt = 0; Irpt < Nrpt; Irpt++) {
        MsecTimer LoopTimer;
        Threads = OnePassUsingCPU(Threads,InputArray,Nx,Ny,OutputArray);
        TheDebugHandler.Logf("Timing","CPU Compute complete at %.3f msec",LoopTimer.ElapsedMsec());
    }
    
    //  Report on results, and on timing.
    
    float Msec = ComputeTimer.ElapsedMsec();
    if (Nrpt <= 0) {
        printf ("No values computed using CPU, as number of repeats set to zero.\n");
    } else {
        if (CheckResults(InputArray,Nx,Ny,OutputArray)) {
            printf ("CPU completed OK, all values computed as expected.\n");
            printf ("CPU took %.3f msec\n",Msec);
            printf ("Average msec per iteration for CPU = %.3f (%d thread(s))\n\n",
                    Msec / float(Nrpt),Threads);
        }
    }
    
    //  Release the arrays used to hold the row addresses for the two arrays, and the array
    //  data.
    
    if (OutputArray) free(OutputArray);
    if (InputArray) free(InputArray);
    if (OutputArrayData) free(OutputArrayData);
    if (InputArrayData) free(InputArrayData);
}


//  ------------------------------------------------------------------------------------------------
//
//                                    S e t  I n p u t  A r r a y
//
//  This sets initial values for the elements of the input array, and is intended to be called
//  by ComputeUsingCPU() and ComputeUsingGPU() once they have allocated memory for the input
//  array. Note that it expects to be passed not the actual start of the input array, but
//  the start of an array giving the address of the start of each row of the input array, which
//  makes the indexing syntax much easier.

void SetInputArray(float** InputArray,int Nx,int Ny)
{
    //  Initialise the input array. The actual values don't really matter.
    
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            InputArray[Iy][Ix] = float(Ny - Iy + Nx - Ix);
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                                  C r e a t e  R o w  A d d r s
//
//  A simple utility that allocates memory for an array that will hold the addresses of the start
//  of each row for a block of memory (passed as Array) that holds an Nx by Ny array of floats.
//  It returns the address of this array of row addresses, which will need to be released using
//  free() once it is no longer needed.

float** CreateRowAddrs(float* Array,int Nx,int Ny)
{
    float** RowAddrs = (float**)malloc(Ny * sizeof(float*));
    for (int Iy = 0; Iy < Ny; Iy++) {
        RowAddrs[Iy] = Array + (Iy * Nx);
    }
    return RowAddrs;
}

//  ------------------------------------------------------------------------------------------------
//
//                              C h e c k  R e s u l t s
//
//  This routine simply uses the CPU to check that the results in OutputArray match what was
//  expected. If it finds any mismatch, it outputs an error message and returns false.

bool CheckResults(float** InputArray,int Nx,int Ny,float** OutputArray)
{
    bool AllOK = true;
    for (int Iy = 0; Iy < Ny; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            if (OutputArray[Iy][Ix] != InputArray[Iy][Ix] + float(Ix + Iy)) {
                printf ("*** Error at [%d][%d]. Got %.1f expected %.1f\n",Iy,Ix,
                        OutputArray[Iy][Ix],InputArray[Iy][Ix] + float(Ix + Iy));
                AllOK = false;
                break;
            }
        }
        if (!AllOK) break;
    }
    return AllOK;
}

//  ------------------------------------------------------------------------------------------------
//
//              D e b u g  A r g  H e l p e r  ::  C h e c k  V a l i d i t y
//

//  The most complicated command line parameter is 'Debug' because it can take a complex
//  set of values, and we need to be able to check that the given values will be accepted.
//  This is why we have a DebugArg helper to check on the validity of the string supplied
//  as the value for the 'Debug' parameter. This will be called by the DebugArg code as
//  part of its validation for the supplied value.

bool DebugArgHelper::CheckValidity(const std::string& Value,std::string* Reason)
{
    //  We need to know if all of the levels specified will be accepted by the debug handler,
    //  and we can do so by calling its CheckLevels() method.
    
    bool Valid = true;
    std::string Unrecognised = TheDebugHandler.CheckLevels(Value);
    if (Unrecognised != "") {
        *Reason = "'" + Unrecognised + "' not recognised";
        Valid = false;
    }
    return Valid;
}

//  ------------------------------------------------------------------------------------------------
//
//                D e b u g  A r g  H e l p e r  ::  H e l p  T e x t
//
//  Provides additional details about the available options for the Debug argument. The command
//  line code uses this when the user responds to a prompt for the argument value with '?'

std::string DebugArgHelper::HelpText(void)
{
    std::string Text = "";
    Text += "Top level options: " + TheDebugHandler.ListLevels() + "\n" +
            "(Should be a comma-separated list of options. '*' acts as a wildcard).";
    return Text;
}


//  ------------------------------------------------------------------------------------------------

//                            P r o g r a m m i n g   N o t e s
/*
    o   I've been using arrays of row addresses to simplify the syntax for accessing multi-
        dimensional arrays for years. I got it from the C version of Numerical Methods, and
        have packaged it into a C++ ArrayManager class that handles up to 4D arrays. I've
        not used this class here because for 2D arrays as used here the crude code in
        CreateRowAddrs() does the job with less bother than introducing more external code.
        Note that Boost provides a very similar functionality (of course it does!) but because
        this uses template code, it is very dependent on compiler optimisation being turned up
        to max to get performance - although at maximum optimisation Boost's version is very
        good. But the scheme used here is just as efficient even at low levels of optimisation.
 
    o   However, you do want to turn up the optimisation for this code, or the CPU version will
        run much slower than you might expect. The latest versions of clang and g++ will produce
        very efficient code for this operation, making use of the CPU vector operations at high
        levels of optimisation (say -O2 and -O3), which I think is very impressive.
 
    o   I don't understand the difference between makeCommandQueue() and newCommandQueue().
        Apple doco includes both, most references seem to use make, but metal-cpp only
        supports newCommandQueue(). There seem to be a number of similar make/new naming
        mismatches in metal-cpp.
 
    o   The PRIVATE_IMPLEMENTATION symbols. The trick is that metal-cpp is a header-only
        implementation, being supplied just as a collection of .hpp files. However, some
        compiled metal-cpp routines need to be included in the link of any code that uses
        metal-cpp. Rather than require a metal-cpp object library, metal-cpp uses a scheme
        where if the .hpp files are included with the 'PRIVATE_IMPLEMENTATION' symbols
        defined for the pre-processor, then these routines will be included in the code and
        compiled and can be linked from the resulting .o file. Obviously, if more than one
        .cpp file does this, these routines will be included in more than one .o file and
        the linker will complain. So the metal-cpp documentation requires that only one
        code file that includes the .hpp files define these PRIVATE variables. Clear?
 
    o   The CPU code would make much more efficient use of threads if we were to split up the
        calculation by rows, create a number of threads each handling a number of rows, and
        then let each thread repeat its operation as many times as specified by the repeat
        count. But that seems to be cheating - what we're trying to do is get a feel for
        how fast the CPU can do just one pass through the basic operation, and re-ordering
        things to speed this up artificially isn't quite the thing to do.
 
    o   Should SetInputArray() be called within the repeat loop or just once before entering
        the loop? Obviously, this code is more of a benchmark than anything else, but in most
        actual applications, even if you set up a GPU pipeline to add numbers to the elements
        of an array, and ran it multiple times, you'd not be running it on the same input data
        each time. Or maybe you would - you might change what you added instead. It doesn't
        actually make a lot of difference, although putting SetInputArray() calls into both the
        CPU and GPU loops slows both down about the same amount, at least on my M2 Macbook (I'd
        wondered if it would slow down the GPU loop more, since it's writing into shared GPU/CPU
        memory, but apparently not) - but this slows down the faster CPU loop proportionally
        more than it does the slower GPU loop. Feel free to experiment...
 
    o   The use of a debug helper for the Debug parameter was introduced in the Vulkan version
        of this code, where the problem is complicated by the use of two separate debug handlers,
        one top-level and one burried in the Vulkan-specific framework the code uses. The code
        used here is rather simpler.
*/
