//
//                     A d d e r . c p p     ( V u l k a n  V e r s i o n )
//
//  This program is intended as a piece of example code showing a calculation involving two
//  2D floating point arrays performed on a GPU, programmed using in C++ using Vulkan. It
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
//     Adder <Nx> <Ny> <Nrpt> <Threads> <cpu> <gpu> <validate> <debug>
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
//     Validate specifies that Vulkan validation is to be enabled.
//
//     Validation is enabled by default. If enabled, Vulkan overheads are increased -
//     particularly for device initialisation, but the diagnostics provided can be very useful
//     when debugging the Vulkan part of the code.
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
//      further re-worked to become the current trivial Metal example, and then again to become
//      this Vulkan version. By now it bears almost no resemblance to the original example, but
//      I'm grateful for the start it gave me.
//
//  History:
//       3rd Jul 2024. First fully commented version. KS.
//      17th Jul 2024. CPU code wasn't initialising the input array properly. Fixed. KS.
//       2nd Aug 2024. Metal version modified to become the Vukan version. Only changes were
//                     to comments and to the GPU section of the code. KS.
//       9th Aug 2024. Minor changes to make use of new command handler features (exit and
//                     help requests) and to debug output. KS.
//      15th Aug 2024. Introduced the use of a debug argument helper. KS.
//      26th Aug 2024. Made Validate true by default. KS.
//      10th Sep 2024. Slight tweaks to disgnostics. KS.
//      14th Sep 2024. Modified following renaming of Framework routines and types. KS.

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
void ComputeUsingGPU(int Nx,int Ny,int Nrpt,bool Validate,const std::string& DebugLevels);
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
    
    TheDebugHandler.LevelsList("Timing,Setup");

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
    BoolArg ValidateArg(TheHandler,"Validate",0,"",true,"Enable Vulkan validation layers");
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
    bool Validate = ValidateArg.GetValue(&Ok,&Error);
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
        
        if (UseGPU) ComputeUsingGPU(Nx,Ny,Nrpt,Validate,DebugLevels);
        
        if (UseCPU) ComputeUsingCPU(Threads,Nx,Ny,Nrpt);
    }
    return 0;
}

//  ------------------------------------------------------------------------------------------------
//
//                                    G P U  c o d e
//
//  This routine performs the 'Adder' operation using the GPU, and is coded using Vulkan.
//
//  This routine is passed the dimensions, Nx and Ny of the arrays to use. It creates an array
//  of floats, Nx by Ny, and calls SetInputArray() to initialise it. It is also passed the repeat
//  count for the operation (Nrpt). It also has to create an output array of the same size and
//  perform the basic 'adder' operation to set the output array values based on those of the
//  input array. It has to perform that operation the number of times specified in Nrpt. Clearly,
//  repeating the operation should have no effect on the final contents of the output array, and
//  is simply in order to get a better idea of the timing.

//                               I n c l u d e  F i l e s
//
//  Needed for Vulkan

#include "KVVulkanFramework.h"

//                                  C o n s t a n t s
//
//  These have to match the values used by the GPU shader code in Adder.comp.

static const uint32_t C_WorkGroupSize = 32;
static const int C_UniformBufferBinding = 0;
static const int C_InputBufferBinding = 1;
static const int C_OutputBufferBinding = 2;

void ComputeUsingGPU(int Nx,int Ny,int Nrpt,bool Validate,const std::string& DebugLevels)
{
    bool StatusOK = true;
    
    MsecTimer SetupTimer;
    TheDebugHandler.Log("Setup","GPU setup starting");

    //  To make things easier for the CPU code, we will create an array of the addresses of
    //  the rows of the input buffer - this allows the use of an ArrayRows[Iy][Ix] syntax when
    //  accessing it. We will do the same for the output array. We set the addresses of these
    //  arrays null now so we can check if they need freeing at the end of the routine.
    
    float** InputArray = nullptr;
    float** OutputArray = nullptr;

    //  Vulkan GPU code gets very long and detailed, because Vulkan is defined using a very
    //  low-level C interface, with lots of options and you have to specify just about everything.
    //  The VulkanFramework used here is not a standard part of Vulkan - it is a set of C++
    //  routines, that package up a number of the standard Vulkan operations, and was written
    //  mainly to help with this example code. For details of just what these routines do, look
    //  at the BasicFramework.cpp/.h files - they're fairly well commented.
    
    //  This is a basic Vulkan initialisation sequence - it creates a Vulkan 'instance', used
    //  to interact with Vulkan, locates a suitable GPU device (with is often the only one
    //  available) and opens that, creating a 'logical device' that represents the selected GPU.
    
    KVVulkanFramework Framework;
    Framework.SetDebugSystemName("Vulkan");
    Framework.SetDebugLevels(DebugLevels);
    Framework.EnableValidation(Validate);
    Framework.CreateVulkanInstance(StatusOK);
    Framework.FindSuitableDevice(StatusOK);
    Framework.CreateLogicalDevice(StatusOK);
    TheDebugHandler.Logf("Setup","GPU device created at %.3f msec",SetupTimer.ElapsedMsec());
    
    //  Create a device buffer to contain the input data array. The options specify that the
    //  buffer is to be used for storage (as opposed to uniform values) and 'shared', ie visible
    //  to both the GPU and the CPU (since we need to use the CPU to set its initial values.)
    //  To set the contents of the buffer using the CPU, we need the address the CPU can use
    //  for this buffer, which we get by mapping it. Then we can initialise the buffer, using
    //  a call to SetInputArray().

    int Length = Nx * Ny * sizeof(float);
    KVVulkanFramework::KVBufferHandle InputBufferHndl;
    InputBufferHndl = Framework.SetBufferDetails(C_InputBufferBinding,"STORAGE","SHARED",StatusOK);
    Framework.CreateBuffer(InputBufferHndl,Length,StatusOK);
    
    long Bytes;
    float* InputBufferAddr = (float*)Framework.MapBuffer(InputBufferHndl,&Bytes,StatusOK);
    InputArray = CreateRowAddrs(InputBufferAddr,Nx,Ny);
    SetInputArray(InputArray,Nx,Ny);
    
    //  And now a device buffer for the output data array. This is essentially the same as for
    //  the input buffer. This will have to be accessed on the CPU side by CheckResults(),
    //  so we use CreateRowAddrs() to set up for this.
    
    KVVulkanFramework::KVBufferHandle OutputBufferHndl;
    OutputBufferHndl = Framework.SetBufferDetails(C_OutputBufferBinding,"STORAGE",
                                                                             "SHARED",StatusOK);
    Framework.CreateBuffer(OutputBufferHndl,Length,StatusOK);
    float* OutputBufferAddr = (float*)Framework.MapBuffer(OutputBufferHndl,&Bytes,StatusOK);
    OutputArray = CreateRowAddrs(OutputBufferAddr,Nx,Ny);

    //  Create a uniform buffer - visible to all the GPU threads - to hold parameters we want
    //  to pass to the GPU code. In this case, the values of Ny and Ny. The layout of this
    //  structure must match that defined in the Adder.comp GPU shader code.
    
    struct AdderArgs {
        int Nx;
        int Ny;
    } Parameters = {Nx,Ny};
    
    long SizeInBytes = sizeof(AdderArgs);
    KVVulkanFramework::KVBufferHandle UniformBufferHndl;
    UniformBufferHndl = Framework.SetBufferDetails(C_UniformBufferBinding,
                                                   "UNIFORM","SHARED",StatusOK);
    Framework.CreateBuffer(UniformBufferHndl,SizeInBytes,StatusOK);
    void* UniformBufferAddr = Framework.MapBuffer(UniformBufferHndl,&Bytes,StatusOK);
    if (StatusOK && UniformBufferAddr) memcpy(UniformBufferAddr,&Parameters,Bytes);

    TheDebugHandler.Logf("Setup","GPU buffers created at %.3f msec",SetupTimer.ElapsedMsec());

    //  Given the handles to those three buffer descriptions, we can specify the layout of the
    //  descriptor set that will be needed to describe them to the GPU shader.
    
    std::vector<KVVulkanFramework::KVBufferHandle> Handles;
    Handles.push_back(UniformBufferHndl);
    Handles.push_back(InputBufferHndl);
    Handles.push_back(OutputBufferHndl);
    VkDescriptorSetLayout SetLayout;
    Framework.CreateVulkanDescriptorSetLayout(Handles,&SetLayout,StatusOK);
    
    //  We also create a pool that can supply such descriptor sets.
    
    VkDescriptorPool DescriptorPool;
    Framework.CreateVulkanDescriptorPool(Handles,1,&DescriptorPool,StatusOK);

    //  And get that pool to supply the descriptor set that we'll use for the pipeline
    //  and set it up with the details of the two buffers.
    
    VkDescriptorSet DescriptorSet;
    Framework.AllocateVulkanDescriptorSet(SetLayout,DescriptorPool,&DescriptorSet,StatusOK);
    Framework.SetupVulkanDescriptorSet(Handles,DescriptorSet,StatusOK);
    TheDebugHandler.Logf("Setup","GPU descriptors created at %.3f msec",SetupTimer.ElapsedMsec());

    //  And, given the set layout, we can specify the layout of the compute pipeline that will
    //  run the shader, and we can create it.
    
    VkPipelineLayout ComputePipelineLayout;
    VkPipeline ComputePipeline;
    Framework.CreateComputePipeline("Adder.spv","main",
                            &SetLayout,&ComputePipelineLayout,&ComputePipeline,StatusOK);
    TheDebugHandler.Logf("Setup","GPU pipeline for adder created at %.3f msec",
                                                               SetupTimer.ElapsedMsec());

    //  We can also create the one compute queue we will need
    
    VkQueue ComputeQueue;
    Framework.GetDeviceQueue(&ComputeQueue,StatusOK);
    
    //  And the command pool and command buffer
    
    VkCommandPool CommandPool;
    VkCommandBuffer CommandBuffer;
    Framework.CreateCommandPool(&CommandPool,StatusOK);
    Framework.CreateComputeCommandBuffer(CommandPool,&CommandBuffer,StatusOK);
    
    //  Tweaking the arrangement of GPU threads and thread groups can be tricky, but if
    //  we assume the GPU shader code has set up a hard-coded local workgroup size of
    //  {C_WorkGroupSize,C_WorkGroupSize,1}, then the following values for _workGroupCounts
    //  cover the whole image (with some possible spillover at the edges that the shader
    //  code has to allow for).
    
    uint32_t WorkGroupCounts[3];
    WorkGroupCounts[0] = (uint32_t(Nx) + C_WorkGroupSize - 1)/C_WorkGroupSize;
    WorkGroupCounts[1] = (uint32_t(Ny) + C_WorkGroupSize - 1)/C_WorkGroupSize;
    WorkGroupCounts[2] = 1;
    TheDebugHandler.Logf("Setup","Work group size %d, %d, %d",C_WorkGroupSize,C_WorkGroupSize,1);
    TheDebugHandler.Logf("Setup","Work group counts %d, %d, %d",
                         WorkGroupCounts[0],WorkGroupCounts[1],WorkGroupCounts[2]);
    if (StatusOK) {
        TheDebugHandler.Logf("Setup","GPU setup took %.3f msec",SetupTimer.ElapsedMsec());
    } else {
        printf("GPU setup failed.\n");
        Nrpt = 0;
    }

    //  This sets up the pipeline for the GPU calculation and runs it.
    
    MsecTimer ComputeTimer;
    
    for (int Irpt = 0; Irpt < Nrpt; Irpt++) {
        
        MsecTimer LoopTimer;

        Framework.RecordComputeCommandBuffer(CommandBuffer,ComputePipeline,
                                    ComputePipelineLayout,&DescriptorSet,WorkGroupCounts,StatusOK);
        TheDebugHandler.Logf("Timing","Command buffer recorded at %.3f msec",
                             LoopTimer.ElapsedMsec());
        
        Framework.RunCommandBuffer(ComputeQueue,CommandBuffer,StatusOK);
        TheDebugHandler.Logf("Timing","Compute complete at %.3f msec",LoopTimer.ElapsedMsec());
        if (!StatusOK) break;
    }
        
    //  Check that we got it right, and if so report on the timing.
        
    if (StatusOK) {
        float Msec = ComputeTimer.ElapsedMsec();
        if (Nrpt <= 0) {
            printf ("No values computed using GPU, as number of repeats set to zero.\n");
        } else {
            if (CheckResults(InputArray,Nx,Ny,OutputArray)) {
                printf ("GPU completed OK, all values computed as expected.\n");
                printf ("GPU took %.3f msec\n",Msec);
                printf ("Average msec per iteration for GPU = %.3f\n\n",Msec / float(Nrpt));
            }
        }
    } else {
        if (Nrpt > 0) printf ("GPU execution failed.\n");
    }

    //  Release the arrays used to hold the row addresses for the two arrays.
        
    if (OutputArray) free(OutputArray);
    if (InputArray) free(InputArray);
    
    //  The Framework destructor will release all the various Vulkan resources.
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
    //  The problem here is that we need to know if all of the levels specified will be accepted
    //  by at least one debug handler. However, we have two in use. One is used directly by this
    //  code (the global TheDebugHandler), but the other is burried in the Vulkan Framework code,
    //  and that handler hasn't even been created at the time the parameters need to be validated.
    
    bool Valid = true;
    
    //  First, we see if TheDebugHandler recognises all the specified levels. It will recognise
    //  things like 'timing' and 'setup', but not the Vulkan Framework specific levels. So any
    //  levels it returns in Unrecognised may be rubbish, or they may be levels accepted by
    //  the debugger in the Vulkan Framework code.
    
    std::string Unrecognised = TheDebugHandler.CheckLevels(Value);
    if (Unrecognised != "") {
        
        //  What we can do is create a temporary debug handler and set it up as if it were the
        //  VulkanFramework debug handler. We can get the list of debug options from the Framework
        //  GetDebugOptions() call - which is a static routine, all Framework instances use the
        //  same levels - and we know that we will give the Framework the "Vulkan" subsystem name
        //  because that's done in ComputeUsingGPU(). So we can get this stand-in to check the
        //  levels.
        
        DebugHandler VulkanStandInHandler("Vulkan");
        VulkanStandInHandler.LevelsList(KVVulkanFramework::GetDebugOptions());
        Unrecognised = VulkanStandInHandler.CheckLevels(Unrecognised);
    }
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
            "Vulkan level options: " + KVVulkanFramework::GetDebugOptions() + "\n" +
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
*/
