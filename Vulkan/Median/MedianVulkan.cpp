//
//                     M e d i a n . c p p     ( V u l k a n   V e r s i o n )
//
//  This program is intended as a piece of example code showing a calculation involving two
//  2D floating point arrays performed on a GPU, programmed using in C++ using Vulkan. It
//  allows the user to perform the same operation using the CPU and compare the timings,
//  and allows some experimentation with the number of CPU threads to be used. The input
//  array is read from a FITS file whose name is given as a command line parameter. Sometimes
//  the operation needs to be repeated a number of times to get a reasonable execution time.
//
//  The problem chosen involves a bit more computation than the similar Adder example. It runs
//  a median filter through the input image, creating the output image by setting each pixel
//  to the median value of an Npix by Npix square box centered on the corresponding input image
//  pixel.
//
//  Invocation:
//     Median <File> <Npix> <Nrpt> <Threads> <cpu> <gpu> <Nx> <Ny> <debug>
//
//  Where:
//     File    is the name of an existing FITS file containing the image.
//     Npix    is the number of pixels across the square median box used.
//     Nrpt    is the number of times the operation is to be repeated.
//     Threads if the CPU is used, the number of CPU threads to be used.
//
//     File,Npix,Nrpt,Threads are positional parameters, and can be specified either by just
//     providing values for them on the command line in the order above, or explicitly
//     by name and value, with an optional '=' sign. If Threads is zero, the maximum number
//     of CPU threads available will be used. Pix should be an odd number.
//
//     Cpu     specifies that the operation is to be carried out using the CPU.
//     Gpu     specifies that the operation is to be carried out using the GPU.
//
//     Cpu and Gpu are boolean values that can be specified explicitly, eg Cpu = true,
//     or just by appearing in the command line, optionally negated, for example as 'cpu'
//     or 'nogpu'. They are not mutually exclusive. By default both are false, but if neither
//     are specified the program will use the GPU.
//
//     If File is specified as blank, then the program will not read data from a file,
//     but will generate dummy data for test purposes. In this case, it will need to
//     know how large an array to generate, and this is provided by the optional
//     Nx,Ny parameters. If File is non-blank, these are ignored.
//
//     Nx      is the X dimension of the arrays in question.
//     Ny      is the Y dimension of the arrays in question.
//
//     Debug   is a string that can be used to control debug output. It must be specified
//             explicitly by name, eg Debug = "timing". The '=' is optional, but the quotes
//             are needed in some cases. 'Debug = timing,fits' is OK, but 'Debug = "*"' will
//             need quotes. If you specify '?' for the value of Debug a list of the options
//             will be provided.
//
//     The command line is processed by the flexible but possibly quirky command line handler
//     used for all these GPU examples. With luck you'll get used to it. It also supports the
//     command line flags 'list' (lists all the parameter values that are going to be used),
//     'prompt' (explicitly prompts for all unspecified parameters) and 'reset' (resets all
//     parameter values to their default). The command handler remembers the values used the
//     last time the program was run, so often the most useful way to use it is to run with
//     'prompt' the first time, then on subsequent runs, just specify any parameter you want
//     to change by name.
//
//  Example:
//     ./Median name.fits 5 1
//     ./Median file name.fits 1 pix 5      has exactly the same effect
//
//  Acknowledgements:
//      This code had its origins in a program in Swift that was itself based on some example
//      Swift code for computation using Metal at:
//      gist.github.com/mhamilt/a5c2bbb02684e5db362712c9be7a02ca
//      That Swift code was then mangled to become code for calculating a Mandlebrot image in a
//      2D array and then reworked for C++ using Apple's metal-cpp subsystem. This then became
//      further re-worked to become the trivial 'Adder' Metal example, which was then expanded
//      to become this 'Median' example, and now it had been further modified to become the
//      Vulkan version of the 'Median' example. By now it bears no resemblance at all to the
//      original code, but I'm grateful for the start it gave me.
//
//  History:
//      14th Aug 2024. First fully commented version. KS.
//      20th Aug 2024. Brought up to match changes to Metal code, particularly the use of
//                     extended command handler options such as argument helpers. KS.
//      12th Sep 2024. Changed 'File' default to blank, and now doesn't try to write out
//                     a result file if one wasn't specified. KS.
//      14th Sep 2024. Modified following renaming of Framework routines and types. KS.
//      24th Sep 2024. Added include of string.h for systems that need it. Fixed minor issue
//                     with error reporting when closing a FITS file. KS.

//  ------------------------------------------------------------------------------------------------
//
//                              I n c l u d e  F i l e s
//
//  Needed to allow multi-threading of the CPU code

#include <thread>

//  The file handling code uses C++17's std::filesystem

#include <filesystem>

//  Some utility code. A Command Handler provides a flexible way of handling command line
//  parameters, and simplifies the coding required for this. An MsecTimer provides a very simple
//  way of timing blocks of code. A Debug Handler provides control over debug output, allowing
//  various debug levels to be enabled from the command line.

#include "CommandHandler.h"
#include "MsecTimer.h"
#include "DebugHandler.h"

//  FITS file access uses the cfitsio library.

#include "fitsio.h"

//  Required on some systems for strncpy().

#include <string.h>

//  This provides a global Debug Handler that all the routines here can use.

DebugHandler TheDebugHandler;

//  ------------------------------------------------------------------------------------------------
//
//               F o r w a r d  D e f i n i t i o n s  &  S t r u c t u r e s

//  Structure used to collect details passed around during the program, mostly about the
//  input FITS file and the various data arrays.

struct MedianDetails {
    fitsfile* Fptr = nullptr;           //  Cfitsio routines access the file through this.
    float* InputData = nullptr;         //  Address of array used for input data from file.
    float* GPUOutputData = nullptr;     //  Address of array used for GPU version of output data.
    float* CPUOutputData = nullptr;     //  Address of array used for CPU version of output data.
    std::string OutputFileName = "";    //  Name of the output FITS file.
};

//  Read the data from the FITS file.
bool ReadFitsFile(std::string& Filename,int* Nx,int* Ny,MedianDetails* Details);
//  Perform the basic opetation using the GPU
void ComputeUsingGPU(int Nx,int Ny,int Pix,int Nrpt,bool Validate,const std::string& DebugLevels,
                                                                         MedianDetails* Details);
//  Perform the basic operation using the CPU
void ComputeUsingCPU(int Threads,int Nx,int Ny,int Pix,int Nrpt,MedianDetails* Details);
//  Set initial values for the input array.
void SetInputArray(float** InputArray,int Nx,int Ny,MedianDetails* Details);
//  Check the results of the operation
bool NoteResults(float** OutputArray,bool FromGPU,int Nx,int Ny,MedianDetails* Details);
//  Utility to set up an array of row addresses to allow use of Array[Iy][Ix] syntax for access.
float** CreateRowAddrs(float* Array,int Nx,int Ny);
//  Write calculated output array to copy of input FITS file.
bool WriteFitsFile(int Nx,int Ny,MedianDetails* Details);
//  Shutdown program and release resources.
void Shutdown(MedianDetails* Details);

//  ------------------------------------------------------------------------------------------------
//
//                                O d d  I n t  A r g
//
//  The size of the median box should really be an odd number of pixels. The command handler
//  used doesn't support an argument whose type is 'odd-valued integer', but it's easy enough
//  to implement one by inheriting from the standard integer argument. This is really overkill,
//  but it does show it can be done quite easily just by overriding two methods. This defines
//  the class used. The implementation comes at the end of this file.

class OddIntArg : public IntArg {
public:
    OddIntArg (CmdHandler& Handler,const std::string& Name,int Posn = 0,
               const std::string& Flags = "",long Reset = 0,long Min = 0, long Max = 0,
               const std::string& Prompt = "",const std::string& Text = "");
protected:
    virtual bool AllowedValue (const std::string& Value);
    virtual std::string Requirement (void);
};

//  ------------------------------------------------------------------------------------------------
//
//                             D e b u g  A r g  H e l p e r
//
//  This provides a way for the program to give some additional help to the command line handler
//  for the 'Debug' parameter, which has a complex syntax, being a comma-separated list of
//  hierarchical options with support for wildcards. It's all very well making these things
//  flexible, but then they take much more explaining...

class DebugArgHelper : public CmdArgHelper {
public:
    bool CheckValidity(const std::string& Value,std::string* Reason);
    std::string HelpText(void);
};

//  ------------------------------------------------------------------------------------------------
//
//                                    M a i n

int main (int Argc, char* Argv[]) {

    //  Set up the various levels for the Debug handler
    
    TheDebugHandler.LevelsList("Timing,Setup,Checks,Fits");

    //  Get the values of the command line arguments. This uses a Command Handler class that
    //  provides a flexible way of dealing with a number of different arguments, for example
    //  by allowing specifications such as 'NPix = 5 Nrpt = 10". Each argument is handled by
    //  an instance of an argument class (eg NxArg, or DebugArg), and the Command Handler
    //  coordinates how these handle the actual command line arguments (argc and argv).
    
    //  Set up the Handler and the argument instances for the various command line arguments.
    //  File,Npix,Nrpt and Threads are the most likely to be needed, and have defined positions
    //  given by Posn. Nx and Ny are only needed if no FITS file is specified.
    
    int Posn = 1;
    CmdHandler TheHandler("Median");
    FileArg FilenameArg(TheHandler,"File",Posn++,"MustExist,NullOk","",
                                                            "FITS file containing image");
    OddIntArg NpixArg(TheHandler,"Npix",Posn++,"",5,1,11,
                               "Size of median box in pixels - should be an odd number");
    IntArg NrptArg(TheHandler,"Nrpt",Posn++,"",1,0,5000,"Repeat count for operation");
    int DefaultThreads = 1;
    int MaxThreads = std::thread::hardware_concurrency();
    if (MaxThreads < 0) { MaxThreads = 0; DefaultThreads = 0; }
    IntArg ThreadsArg(TheHandler,"Threads",Posn++,"",DefaultThreads,0,MaxThreads,
                                                                   "CPU threads to use");
    IntArg NxArg(TheHandler,"Nx",0,"",1024,2,1024*1024,"X-dimension of image");
    IntArg NyArg(TheHandler,"Ny",0,"",1024,2,1024*1024,"Y-dimension of image");
    BoolArg CpuArg(TheHandler,"Cpu",0,"",false,"Perform computation using CPU");
    BoolArg GpuArg(TheHandler,"Gpu",0,"",false,"Perform computation using GPU");
    BoolArg ValidateArg(TheHandler,"Validate",0,"",false,"Enable Vulkan validation layers");
    StringArg DebugArg(TheHandler,"Debug",0,"NoSave","","Debug levels");
    DebugArgHelper DebugHelper;
    DebugArg.SetHelper(&DebugHelper);

    //  Now get the values for each of the command line arguments. The ReadPrevious() and
    //  SaveCurrent() calls allow the handler to remember the argument values used the last
    //  time the program was run. If you don't care for that, comment them out.
    
    std::string Error = "";
    if (TheHandler.IsInteractive()) TheHandler.ReadPrevious();
    bool Ok = TheHandler.ParseArgs(Argc,Argv);
    std::string Filename = FilenameArg.GetValue(&Ok,&Error);
    int Nx = 0,Ny = 0;
    if (Filename == "") {
        Nx = NxArg.GetValue(&Ok,&Error);
        Ny = NyArg.GetValue(&Ok,&Error);
    }
    int Npix = NpixArg.GetValue(&Ok,&Error);
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
        
        //  If a file name was specified, make a copy of the file (which will become the output
        //  file), get the image dimensions, and read in the main data array.

        MedianDetails Details;
        if (Filename != "") {
            ReadFitsFile(Filename,&Nx,&Ny,&Details);
        }
        
        printf ("\nPerforming 'Median' test, arrays of %d rows, %d columns. Repeat count %d.\n",
                Ny,Nx,Nrpt);
        printf ("Median box is %d by %d.\n\n",Npix,Npix);
        
        //  If neither CPU not GPU was specified on the command line, use GPU.
        
        if (!UseGPU && !UseCPU) UseGPU = true;
        
        //  Perform the test using either CPU or GPU (or both). Both ComputeUsingGPU() and its CPU
        //  equivalent, ComputeUsingCPU() are expected to create arrays of the size specified, call
        //  SetInputArray() to initialise the input array, then perform the basic 'Median' operation
        //  as specified.
        
        if (UseGPU) ComputeUsingGPU(Nx,Ny,Npix,Nrpt,Validate,DebugLevels,&Details);
        
        if (UseCPU) ComputeUsingCPU(Threads,Nx,Ny,Npix,Nrpt,&Details);
        
        //  Write out the filtered array to the copy of the input FITS file and close program.
        
        if (Filename != "") WriteFitsFile (Nx,Ny,&Details);
        Shutdown(&Details);
    }
    return 0;
}

//  ------------------------------------------------------------------------------------------------
//
//                             R e a d  F i t s  F i l e

bool ReadFitsFile(std::string& Filename,int* Nx,int* Ny,MedianDetails* Details)
{
    fitsfile *Fptr;
    char Error[80];
    int Status = 0;
    
    //  First, we make a copy of the input file with "Median_" prepended to the filename,
    //  and make sure the resulting file has write access. (This allows the FITS routines
    //  to modify the new file even if the original was readonly.) If such a file exists
    //  (it often will) overwrite it. Since we have to use C++17 anyway, we may as well
    //  use the std::filesystem routines for this.
    
    
    std::string MedianFile = "Median_" + Filename;
    TheDebugHandler.Logf("Fits","Copying input file %s to new output file %s",Filename.c_str(),
                                                                            MedianFile.c_str());
    std::error_code ErrorCode;
    if (std::filesystem::exists(MedianFile,ErrorCode)) {
        TheDebugHandler.Logf("Fits","File %s already exists and will be overwritten",
                                                                   MedianFile.c_str());
    }
    std::string Text;
    if (!std::filesystem::copy_file(Filename,MedianFile,
                            std::filesystem::copy_options::overwrite_existing,ErrorCode)) {
        Text = "Unable to create new median file: " + ErrorCode.message();
        Status = 1;
    } else {
        std::filesystem::permissions(MedianFile,std::filesystem::perms::owner_write,
                                          std::filesystem::perm_options::add,ErrorCode);
        if (ErrorCode) {
            Text = "Unable to make new median file writeable: " + ErrorCode.message();
            Status = 1;
        }
    }
    if (Status) {
        strncpy(Error,Text.c_str(),sizeof(Error) - 1);
    } else {
        
        //  Now open the copy of the original file, allocate a large enough block of memory
        //  to hold the main image, and read it in. (This code is using cfitsio routines, which
        //  have their origins some time back, and so have a bit of a different feel to the
        //  C++17 code just used to do the file copy.)
         
        if (fits_open_file(&Fptr,MedianFile.c_str(),READWRITE,&Status)) {
            fits_get_errstatus (Status,Error);
        } else {
            long Naxes[2];
            int Nfound;
            Details->Fptr = Fptr;
            if (fits_read_keys_lng(Fptr,"NAXIS",1,2,Naxes,&Nfound,&Status)) {
                fits_get_errstatus (Status,Error);
            } else {
                if (Nfound != 2) {
                    strncpy(Error,"File main image is not 2-dimensional",sizeof(Error));
                    Status = 1;
                } else {
                    int NPixels = Naxes[0] * Naxes[1];
                    float* Data = (float*)malloc(NPixels * sizeof(float));
                    long Fpixel = 1;
                    float Nullval = 0.0;
                    int Anynull;
                    if (fits_read_img(Fptr,TFLOAT,Fpixel,NPixels,&Nullval,Data,&Anynull,&Status)) {
                        fits_get_errstatus (Status,Error);
                    } else {
                        Details->InputData = Data;
                        *Nx = Naxes[0];
                        *Ny = Naxes[1];
                        TheDebugHandler.Logf("Fits","File opened, 2D data array %d by %d",*Nx,*Ny);
                    }
                }
            }
        }
        if (Status == 0) Details->OutputFileName = MedianFile;
    }
    if (Status) {
        if (TheDebugHandler.Active("Fits")) {
            TheDebugHandler.Logf("Fits","Error reading FITS file: %s",Error);
        } else {
            printf ("Error reading FITS file: %s\n",Error);
        }
    }
    return (Status == 0);
}

//  ------------------------------------------------------------------------------------------------
//
//                                    G P U  c o d e
//
//  This routine performs the 'Median' operation using the GPU, and is coded using Vulkan.
//
//  This routine is passed the dimensions, Nx and Ny of the arrays to use. It creates an array
//  of floats, Nx by Ny, and calls SetInputArray() to initialise it. It has to be passed the
//  size of the median box in pixels (this should be an odd number) and the repeat count for the
//  operation (Nrpt). It creates an output array of the same size and performs the basic 'Median'
//  operation, using the GPU, to set the output array values based on those of the input array.
//  It has to perform that operation the number of times specified in Nrpt. Clearly, repeating
//  the operation should have no effect on the final contents of the output array, and is simply
//  in order to get a better idea of the timing, if necessary. After performing the operation
//  for the final time, it calls NoteResults() to allow the main code to compare the results with
//  those of the CPU and to save the result to any output FITS file. It needs to be passed the
//  program's MedianDetails structure so it can pass this on in turn to NoteResults().

//                               I n c l u d e  F i l e s
//
//  Needed for Vulkan

#include "KVVulkanFramework.h"

//                                  C o n s t a n t s
//
//  These have to match the values used by the GPU shader code in Median.comp.

static const uint32_t C_WorkGroupSize = 32;
static const int C_UniformBufferBinding = 0;
static const int C_InputBufferBinding = 1;
static const int C_OutputBufferBinding = 2;

void ComputeUsingGPU(int Nx,int Ny,int Npix,int Nrpt,bool Validate,const std::string& DebugLevels,
                                                                           MedianDetails* Details)
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
    //  The KVVulkanFramework used here is not a standard part of Vulkan - it is a set of C++
    //  routines, that package up a number of the standard Vulkan operations, and was written
    //  mainly to help with this example code. For details of just what these routines do, look
    //  at the KVVulkanFramework.cpp/.h files - they're fairly well commented.
    
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
    TheDebugHandler.Logf("Setup","GPU setup device created at %.3f msec",SetupTimer.ElapsedMsec());
    
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
    SetInputArray(InputArray,Nx,Ny,Details);
    
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
    //  to pass to the GPU code. In this case, the values of Nx, Ny and Npix. The layout of this
    //  structure must match that defined in the Median.comp GPU shader code.
    
    struct MedianArgs {
        int Nx;
        int Ny;
        int Npix;
    } Parameters = {Nx,Ny,Npix};
    
    long SizeInBytes = sizeof(MedianArgs);
    KVVulkanFramework::KVBufferHandle UniformBufferHndl;
    UniformBufferHndl = Framework.SetBufferDetails(C_UniformBufferBinding,
                                                   "UNIFORM","SHARED",StatusOK);
    Framework.CreateBuffer(UniformBufferHndl,SizeInBytes,StatusOK);
    void* UniformBufferAddr = Framework.MapBuffer(UniformBufferHndl,&Bytes,StatusOK);
    if (StatusOK && UniformBufferAddr) memcpy(UniformBufferAddr,&Parameters,Bytes);
    
    TheDebugHandler.Logf("Setup","GPU setup buffers created at %.3f msec",SetupTimer.ElapsedMsec());
    
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
    Framework.CreateComputePipeline("Median.spv","main",
                                    &SetLayout,&ComputePipelineLayout,&ComputePipeline,StatusOK);
    TheDebugHandler.Logf("Setup","GPU pipeline created at %.3f msec",SetupTimer.ElapsedMsec());

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
        bool FromGPU = true;
        if (Nrpt <= 0) {
            printf ("No values computed using GPU, as number of repeats set to zero.\n");
        } else {
            if (NoteResults(OutputArray,FromGPU,Nx,Ny,Details)) {
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
//  This code performs the 'Median' operation using the CPU, and is designed to do exactly what
//  ComputeUsingGPU() does, but using the CPU. The main routine called is ComputeUsingCPU(). Just
//  like ComputeUsingGPU(), it is passed the dimensions of the arrays to use. It allocates an array
//  of floats, Nx by Ny, and calls SetInputArray() to initialise it. It also creates an output
//  array of the same size and performs the basic 'Median' operation to set the output array values
//  based on those of the input array. It has to be passed the size of the median box in pixels
//  (Npix, which should be an odd number).It is also passed the repeat count for the operation
//  (Nrpt) and performs the operation the number of times specified in Nrpt. After performing the
//  operation for the final time, it calls NoteResults() to allow the main code to compare the
//  results with those of the GPU and to save the result to any output FITS file. It needs to
//  be passed the program's MedianDetails structure so it can pass this on in turn to NoteResults().
//  The CPU code can make use of multiple CPU threads. If Threads is passed as zero, it uses the
//  maximum available number of CPU threads. Otherwise, Threads is taken as the maximum number of
//  CPU threads to use.
//
//  The basic operation is performed by ComputeRangeUsingCPU() and you could do the whole thing
//  just with one call to ComputeRangeUsingCPU() with the Y-range parameters Iyst and Iyen set
//  to 0 and Ny respectively. That would handle the whole image using one CPU thread. However, as
//  this operation will probably gain by using CPU multi-threading, the main routine actually
//  calls OnePassUsingCPU() for each iteration. This then handles the threading, being able to
//  split the job up so that multiple threads each handle a subset of the rows of the image (ie
//  splitting up in Y). It starts up a number of threads, each running ComputeRangeUsingCPU()
//  over a different set of image rows.

void ComputeUsingCPU(int Threads,int Nx,int Ny,int Npix,int Nrpt,MedianDetails* Details)
{
    //  Forward declaration for the routine that does most of the work.
    
    int OnePassUsingCPU(int Threads,float** InputArray,int Nx,int Ny,int Npix,float** OutputArray);
    
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
    TheDebugHandler.Logf("Setup","CPU arrays created, %d by %d",Nx,Ny);
    
    //  Initialise the input array;
    
    SetInputArray(InputArray,Nx,Ny,Details);

    //  See how many threads the hardware supports. If Threads was passed as zero, use this
    //  maximum number of threads. Otherwise use the number passed in Threads, if that many
    //  are available.
    
    int MaxThreads = std::thread::hardware_concurrency();
    if (MaxThreads <= 0) MaxThreads = 1;
    if (Threads <= 0) Threads = MaxThreads;
    if (Threads > MaxThreads) Threads = MaxThreads;
    TheDebugHandler.Logf("Setup","CPU using %d threads out of maximum of %d\n",Threads,MaxThreads);
    
    MsecTimer LoopTimer;
    
    //  Repeat a single pass through the whole image, as many times as specified by the repeat
    //  count. OnePassUsingCPU is passed the number of threads specified (with zero meaning
    //  use as many as are available) and returns the number actually used.
    
    for (int Irpt = 0; Irpt < Nrpt; Irpt++) {
        Threads = OnePassUsingCPU(Threads,InputArray,Nx,Ny,Npix,OutputArray);
    }
    
    //  Report on results, and on timing.
    
    float Msec = LoopTimer.ElapsedMsec();
    bool FromGPU = false;
    if (Nrpt <= 0) {
        printf ("No values computed using CPU, as number of repeats set to zero.\n");
    } else {
        if (NoteResults(OutputArray,FromGPU,Nx,Ny,Details)) {
            printf ("CPU completed OK.\n");
            printf ("CPU took %.3f msec\n",Msec);
            printf ("Average msec per iteration for CPU = %.3f (threads = %d)\n\n",
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

//  ComputeRangeUsingCPU() performs the CPU median calculation over a range of lines of the
//  image - specified by Iyst and Iyen (starting from 0) - but only once and using a single thread.

void ComputeRangeUsingCPU(float** InputArray,int Nx,int Ny,int Iyst,
                                   int Iyen,int Npix,float** OutputArray)
{
    //  Forward definition of routine to calculate median for a specific array location, Ix,Iy.
    
    float MedianElement(float** InputArray,int Nx,int Ny,int Ix,int Iy,int Npix);

    for (int Iy = Iyst; Iy < Iyen; Iy++) {
        for (int Ix = 0; Ix < Nx; Ix++) {
            OutputArray[Iy][Ix] = MedianElement(InputArray,Nx,Ny,Ix,Iy,Npix);
        }
    }
}

//  OnePassUsingCPU() performs the CPU median calculation over the whole image, only once, using
//  up to a specified number (Threads) of CPU threads. If Threads is zero, it uses the maximum
//  available number of threads. It returns the number of threads actually used.

int OnePassUsingCPU(int Threads,float** InputArray,int Nx,int Ny,int Npix,float** OutputArray)
{
    //  If we're only using one thread, do the calculation in the main thread, avoiding any
    //  threading overheads.
    
    if (Threads == 1) {
        ComputeRangeUsingCPU(InputArray,Nx,Ny,0,Ny,Npix,OutputArray);
    } else {
        
        //  If threading, create the specified number of threads, and divide the image rows
        //  between them.
        
        std::thread ThreadList[Threads];
        int Iy = 0;
        int Iyinc = Ny / Threads;
        for (int IThread = 0; IThread < Threads; IThread++) {
            ThreadList[IThread] = std::thread (ComputeRangeUsingCPU,
                                        InputArray,Nx,Ny,Iy,Iy+Iyinc,Npix,OutputArray);
            Iy += Iyinc;
        }
        
        //  Wait for all the threads to complete.
        
        for (int IThread = 0; IThread < Threads; IThread++) {
            ThreadList[IThread].join();
        }
        
        //  If there were rows left unhandled (because Ny was not a multiple of NThreads),
        //  finish them off in the main thread.
        
        if (Iy < Ny) {
            ComputeRangeUsingCPU(InputArray,Nx,Ny,Iy,Ny,Npix,OutputArray);
        }
    }
    return Threads;
}

//  ------------------------------------------------------------------------------------------------
//
//                                    M e d i a n  c o d e
//
//  These routines are essentially the same as the code run by the GPU to calculate the median
//  values for the elements of the array. The CPU code above calls MedianElement() for each
//  element of the array. It collects the values in the box surrounding that element and passes
//  them to CalcMedian(), which works out the median of the values passed to it. Note that
//  on a normal CPU (as opposed to a GPU) this code could handle any size of median box, by
//  dynamically allocating a work array of the size needed. But the GPU can't do that, and so
//  has a fixed maximum size for the work array, and this code does the same.

// Allow for values of Npix up to 11.

#define NPIXSQ_MAX 121

//  This finds the median value in the array X, without recursion,
//  using a quickselect algorithm which partially sorts X.

float CalcMedian(float X[NPIXSQ_MAX], int len)
{
#define swap(a,b) {float t = X[a]; X[a] = X[b], X[b] = t;}
    int left = 0, right = len - 1;
    int cent = len/2;
    float pivot;
    int pos, i;
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
        for (int i = 0; i < cent - 1; i++) {
            if (X[i] > temp) temp = X[i];
        }
        median = (median+temp)*0.5;
    }
    
    return median;
}

float MedianElement(float** InputArray,int Nx,int Ny,int Ix,int Iy,int Npix)
{
    //  Fill a work array with the input array elements in a box Pix wide around the
    //  target element. Allow for the edges of the image.
    
    while (Npix * Npix > NPIXSQ_MAX) Npix--;
    float work[NPIXSQ_MAX];
    int npixby2 = Npix / 2;
    int ixmin = Ix - npixby2;
    int ixmax = Ix + npixby2;
    int iymin = Iy - npixby2;
    int iymax = Iy + npixby2;
    if (ixmin < 0) ixmin = 0;
    if (ixmax >= Nx) ixmax = Nx - 1;
    if (iymin < 0) iymin = 0;
    if (iymax >= Ny) iymax = Ny - 1;
    int ipix = 0;
    for (int yind = iymin; yind <= iymax; yind++) {
        for (int xind = ixmin; xind <= ixmax; xind++) {
            work[ipix++] = InputArray[yind][xind];
        }
    }
    return CalcMedian(work,ipix);
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

void SetInputArray(float** InputArray,int Nx,int Ny,MedianDetails* Details)
{
    //  If we have an opened FITS file and have read the data from it, copy that data into
    //  the input array. Note that InputArray[0] will hold the address of the start of the
    //  actual input array.
    
    if (Details->InputData) {
        memcpy(InputArray[0],Details->InputData,Nx * Ny * sizeof(float));
    } else {
    
        //  Otherwise, we just make up some suitable values. The actual values don't matter
        //  too much.
        
        for (int Iy = 0; Iy < Ny; Iy++) {
            for (int Ix = 0; Ix < Nx; Ix++) {
                InputArray[Iy][Ix] = float(Ny - Iy + Nx - Ix);
            }
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                             W r i t e  F i t s  F i l e
//
//  This routine writes the calculated image (saved by a call to NoteResults()) to the output
//  file created by ReadFitsFile() as its main image, replacing the original unfiltered image.

bool WriteFitsFile(int Nx,int Ny,MedianDetails* Details)
{
    int Status = 0;
    char Error[80];
    
    //  The cfitsio handle to the output file should be in Details->Fptr.
    
    fitsfile* Fptr = Details->Fptr;
    if (Fptr == nullptr) {
        strncpy(Error,"No output file open",sizeof(Error));
        Status = 1;
    } else {
        
        //  See if we do have a saved image, and write it out if we do.
        
        float* OutputImage = Details->GPUOutputData;
        if (OutputImage == nullptr) OutputImage = Details->CPUOutputData;
        if (OutputImage == nullptr) {
            strncpy (Error,"No output image calculated",sizeof(Error));
            Status = 1;
        } else {
            long Fpixel = 1;
            int NPixels = Nx * Ny;
            if (fits_write_img(Fptr,TFLOAT,Fpixel,NPixels,OutputImage,&Status)) {
                fits_get_errstatus (Status,Error);
            }
        }
    }
    
    //  Close the file (clearing Details->Fptr so Shutdown() won't try to close it twice),
    //  and summarise the final status. (If something has gone wrong, Status will be non-zero.)
    //  Note that fits_close_file() will close the file even if passed non-zero status, but we
    //  don't want to modify Error if it already has a description of an earlier error.
    
    int CloseStatus = 0;
    if (Fptr && fits_close_file(Fptr,&CloseStatus)) {
        if (Status == 0) fits_get_errstatus (CloseStatus,Error);
    }
    Details->Fptr = nullptr;
    if (TheDebugHandler.Active("Fits")) {
        if (Status) TheDebugHandler.Logf("Fits","Error writing to FITS file: %s",Error);
        else TheDebugHandler.Log("Fits","Output image written OK");
    } else {
        if (Status) printf ("Error writing to FITS file: %s\n",Error);
        else printf("Output image written OK to %s\n",Details->OutputFileName.c_str());
    }
    return (Status == 0);
}

//  ------------------------------------------------------------------------------------------------
//
//                              C r e a t e  R o w  A d d r s
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
//                                N o t e  R e s u l t s
//
//  This routine will be called by both the GPU and CPU versions of the code once they have
//  calculated their version of the output array. It can check them against each other if
//  called from both. It saves the first version of the data it is passed so that it can be
//  written to the output FITS file and used to compare against any second version. (This
//  allows the single routines that comprise the GPU and CPU code to release the arrays they
//  create, which keeps things simpler.)

bool NoteResults(float** OutputArray,bool FromGPU,int Nx,int Ny,MedianDetails* Details)
{
    bool AllOK = true;
    
    //  If we have debug switched on for this routine, we probably don't want to duplicate
    //  messages. This lets us check for this - it's probably overkill, to be honest.
    
    bool DebugChecks = TheDebugHandler.Active("Checks");

    //  Save the data we're being passed. See if we already have the data from the other device
    //  (ie the GPU if this data is from the CPU, or vice-versa). We only need to save the data
    //  from the first device that calls this routine. The data from both should be the same,
    //  after all, and we only need to save one of the two to be able to check this.
    
    const char* ThisDevice = FromGPU ? "GPU" : "CPU";
    const char* OtherDevice = FromGPU ? "CPU" : "GPU";
    float* OtherData = nullptr;
    if (FromGPU) OtherData = Details->CPUOutputData;
    else OtherData = Details->GPUOutputData;
    if (OtherData == nullptr) {
        TheDebugHandler.Logf("Checks","Saving %s data",ThisDevice);
        int DataBytes = Nx * Ny * sizeof(float);
        float* SavedData = (float*)malloc(DataBytes);
        memcpy(SavedData,OutputArray[0],DataBytes);
        if (FromGPU) Details->GPUOutputData = SavedData;
        else Details->CPUOutputData = SavedData;
    }
    //  If we have the data from the other calculation (either GPU or GPU), compare the two.
    //  The code is checking two floating point values for equality, which is usually frowned
    //  upon, but in this case the values in question aren't being calculated (in which case
    //  rounding error might be a problem) but simply copied, so should be exactly the same.
    
    if (OtherData) {
        TheDebugHandler.Logf ("Checks","Checking %s results against %s results",
                                                                  ThisDevice,OtherDevice);
        float** OtherArray = CreateRowAddrs(OtherData,Nx,Ny);
        for (int Iy = 0; Iy < Ny; Iy++) {
            for (int Ix = 0; Ix < Nx; Ix++) {
                if (OutputArray[Iy][Ix] != OtherArray[Iy][Ix]) {
                    if (DebugChecks) {
                        TheDebugHandler.Logf("Checks","Error at [%d][%d] %8.1f (%s) != %8.1f (%s)",
                              Iy,Ix,OutputArray[Iy][Ix],ThisDevice,OtherArray[Iy][Ix],OtherDevice);
                    } else {
                        printf ("Error at [%d][%d] %8.1f (%s) != %8.1f (%s)\n",
                            Iy,Ix,OutputArray[Iy][Ix],ThisDevice,OtherArray[Iy][Ix],OtherDevice);
                    }
                    AllOK = false;
                    break;
                }
            }
            if (!AllOK) break;
        }
        if (AllOK) {
            if (DebugChecks) TheDebugHandler.Log("Checks","Data from CPU and GPU match OK");
            else printf ("Data from CPU and GPU match OK\n");
        }
        free (OtherArray);
    }
    return AllOK;
}

//  ------------------------------------------------------------------------------------------------
//
//                                S h u t d o w n
//
//  This routine is called at the end of the program to release any resources still in use.

void Shutdown(MedianDetails* Details)
{
    int Status = 0;
    if (Details->Fptr) fits_close_file(Details->Fptr,&Status);
    if (Details->InputData) free(Details->InputData);
    if (Details->GPUOutputData) free(Details->GPUOutputData);
    if (Details->CPUOutputData) free(Details->CPUOutputData);
}

//  ------------------------------------------------------------------------------------------------
//
//                         C o m m a n d  H a n d l e r  C l a s s e s
//
//  The following OddIntArg and DebugArgHelper classes have nothing to do with median code as such
//  or with GPU calculations. They simply provide some help to the command handler to provide a
//  bit more feedback for the command line parameters.

//  ------------------------------------------------------------------------------------------------
//
//                                O d d  I n t  A r g
//
//  This class overrides the command handler's IntArg class used for integer arguments, to provide
//  an argument class that handles odd-valued integer arguments. All we have to do is replace the
//  AllowedValue() method with one that adds a test for an odd value, and to replace the
//  Requirement() method with one that includes the 'odd-valued' constraint in the description
//  it generates. (Obviously, writing a class like this requires a deal of knowledge of the
//  internals of the IntArg class - they do say 'Inheritance breaks encapsulation'.)

OddIntArg::OddIntArg (CmdHandler& Handler,const std::string& Name,int Posn,
                      const std::string& Flags,long Reset,long Min,long Max,
                      const std::string& Prompt,const std::string& Text) :
    IntArg (Handler,Name,Posn,Flags,Reset,Min,Max,Prompt,Text)
{
    //  The constructor does nothing but initialise the class just as a standard IntArg.
}

bool OddIntArg::AllowedValue (const std::string& Value)
{
    //  Use the standard checks for an allowed value, then add a check that the value is odd.
    //  (CheckValidValue() here is only called as a slightly awkward way of getting the integer
    //  value - we already know it will pass the validity test, as it passed AllowedValue().)
    
    bool Valid = IntArg::AllowedValue(Value);
    if (Valid) {
        long IntValue;
        (void) CheckValidValue(Value,&IntValue);
        if ((IntValue & 1) == 0) Valid = false;
    }
    return Valid;
}

std::string OddIntArg::Requirement (void)
{
    //  This does the same as the standard Requirement() method, but adds 'odd-valued'.
    
    long Min,Max;
    GetRange(&Min,&Max);
    std::string Text =
       "an odd-valued integer in the range " + FormatInt(Min) + " to " + FormatInt(Max);
    return Text;
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
   
    o   The CPU code would make much more efficient use of threads if we were to split up the
        calculation by rows, create a number of threads each handling a number of rows, and
        then let each thread repeat its operation as many times as specified by the repeat
        count. But that seems to be cheating - what we're trying to do is get a feel for
        how fast the CPU can do just one pass through the basic operation, and re-ordering
        things to speed this up artificially isn't quite the thing to do.
 
    o   Possibly confusingly, this code uses two separate ways to interact with the command
        handler code to provide additional features for some parameters. The Npix parameter,
        which has to be odd, is handled by providing a completely new form of Argument, the
        OddIntArg, which inherits from IntArg but overrides the AllowedValue() method to
        add a test for an odd value (and overrides the Requirement() method to explain this).
        The Debug parameter, which is a string that can have a complex set of possible values,
        makes use of the recently-added option of providing a CmdArgHelper instance for the
        parameter, which overrides the helper's CheckValidity() method to check against the
        possible string values (and overrides the HelpText() method to explain this). Both
        options have their uses, and either could have been used in each case. Creating a
        whole new argument class is the most flexible, but needs more internal knowledge.
        All of this is somewhat out of place in code that's supposed to demonstrate how to
        use a GPU. Sorry about that.
*/
