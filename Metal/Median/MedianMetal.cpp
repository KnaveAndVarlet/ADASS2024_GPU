//
//                     M e d i a n . c p p     ( M e t a l  V e r s i o n )
//
//  This program is intended as a piece of example code showing a calculation involving two
//  2D floating point arrays performed on a GPU, programmed using in C++ using Metal. It
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
//      to become this 'Median' example. By now it bears no resemblance at all to the original
//      example, but I'm grateful for the start it gave me.
//
//  History:
//      23rd Jul 2024. First fully commented version. KS.
//      19th Aug 2024. Brought up to match changes to Adder code, particularly the use of
//                     extended command handler options such as argument helpers. KS.
//      12th Sep 2024. Changed 'File' default to blank, and now doesn't try to write out
//                     a result file if one wasn't specified. KS.
//      27th Sep 2024. CPU and GPU times now reported even if results prove to be wrong. KS.

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
void ComputeUsingGPU(int Nx,int Ny,int Pix,int Nrpt,MedianDetails* Details);
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
    
    TheDebugHandler.LevelsList("Timing,Setup,Checks,Fits,Metal");

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
        
        if (UseGPU) ComputeUsingGPU(Nx,Ny,Npix,Nrpt,&Details);
        
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
//  This routine performs the 'Median' operation using the GPU, and is coded using Apple's
//  metal-cpp Metal interface. It's convenient to use this, particularly for anyone who
//  knows C++ and would prefer not to have to use Apple's preferred Objective-C or Swift.
//  However, metal-cpp doesn't expose all that Metal can do, and - possibly more awkwardly -
//  Apple's on-line documentation for Metal allows you to switch between Objective-C and
//  Swift descriptions of the routines, but metal-cpp is less well documented.
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

void ComputeUsingGPU(int Nx,int Ny,int Npix,int Nrpt,MedianDetails* Details)
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
    //  default device is usually what we need.
    
    MTL::Device* Device = MTLCreateSystemDefaultDevice();
    TheDebugHandler.Logf("Setup","GPU setup device created at %.3f msec",SetupTimer.ElapsedMsec());
    TheDebugHandler.Logf("Metal","Device is '%s'",Device->name()->cString(UTF8StringEncoding));

    //  We need to get the 'Median' function from the library into which it's been compiled from
    //  the code in Median.metal. This is one area of this code that might go wrong, for example
    //  if the library is not present, or if it doesn't contain the function we want, so we
    //  test for both those possibilities.
    
    NS::Error* ErrorPtr = nullptr;
    MTL::Function* MedianFunction = nullptr;
    MTL::Library* Library = Device->newLibrary(NS::String::string("Compute.metallib",
                                                       UTF8StringEncoding),&ErrorPtr);
    if (Library == nullptr || ErrorPtr != nullptr) {
        printf ("Error opening library 'Compute.metallib'.\n");
        if (ErrorPtr) {
            printf ("Reason: %s\n",ErrorPtr->localizedDescription()->cString(UTF8StringEncoding));
        }
    } else {
        TheDebugHandler.Logf("Setup","GPU setup library created at %.3f msec",
                                                                   SetupTimer.ElapsedMsec());
        MedianFunction = Library->newFunction(NS::String::string("Median",UTF8StringEncoding));
        if (MedianFunction == nullptr) printf ("Unable to find 'Median' function in library\n");
    }
    
    //  If we've got the Median function set up, things are probably going to work. Because this
    //  is intended as example code rather than real production code, I don't want to clutter
    //  it too much with error handling from here on - production code should be more careful!
    
    if (MedianFunction) {
        TheDebugHandler.Logf("Setup","GPU setup Median function created at %.3f msec",
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
        unsigned int BufferOptions = MTL::StorageModeShared;
        MTL::Buffer* InputBuffer = Device->newBuffer(AllocationSize,BufferOptions);
        
        //  To set the contents of the buffer using the CPU, we need the address the CPU can use
        //  for this buffer, which we get using its contents() method. Then we can initialise
        
        InputArray = CreateRowAddrs((float*)InputBuffer->contents(),Nx,Ny);
        SetInputArray(InputArray,Nx,Ny,Details);

        //  And now a device buffer for the output data array. This is essentially the same as for
        //  the input buffer. This will have to be accessed on the CPU side by NoteResults(),
        //  so we use CreateRowAddrs() to set up for this.
        
        MTL::Buffer* OutputBuffer = Device->newBuffer(AllocationSize,BufferOptions);
        OutputArray = CreateRowAddrs((float*)OutputBuffer->contents(),Nx,Ny);
        TheDebugHandler.Logf("Setup","GPU setup buffers created at %.3f msec",
                                                                   SetupTimer.ElapsedMsec());

        //  We don't need to explicitly set up a buffer to pass parameters to the GPU, but
        //  we do need to put them in a structure that matches what the GPU code in Median.metal
        //  expects. In this case, we don't really need a structure, but in more complex cases
        //  with more parameters we will, so we might as well use one here and initialise it
        //  with the size of the median box.
        
        struct MedianArgs {
            int Npix;
        };
        MedianArgs TheArgs = {Npix};
        
        //  We need a command queue that will be able to supply a command buffer.
        
        MTL::CommandQueue* CommandQueue = Device->newCommandQueue();
        TheDebugHandler.Logf("Setup","GPU setup command queue created at %.3f msec",
                                                                   SetupTimer.ElapsedMsec());

        //  This lets us create a 'pipeline state' that will execute this function.
        
        MTL::ComputePipelineState* PipelineState =
        Device->newComputePipelineState(MedianFunction,&ErrorPtr);
        TheDebugHandler.Logf("Setup","GPU setup pipeline state created at %.3f msec",
                                                                  SetupTimer.ElapsedMsec());

        //  Now we need to work out the way threads in the GPU will be allocated to the computation.
        //  This code simply follows the general guidelines in the Apple documentation.
        
        int ThreadGroupSize = PipelineState->maxTotalThreadsPerThreadgroup();
        int ThreadWidth = PipelineState->threadExecutionWidth();
        if (ThreadGroupSize > (Nx * Ny)) ThreadGroupSize = Nx * Ny;
        TheDebugHandler.Logf("Metal","Max threads per threadgroup %d, Thread width %d",
                             ThreadGroupSize,ThreadWidth);
        TheDebugHandler.Logf("Metal","Using thread group size %d",ThreadGroupSize);
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
            //  must match the index values used in the metal kernel code in Median.metal
            
            Encoder->setBuffer(InputBuffer,0,0);
            Encoder->setBuffer(OutputBuffer,0,1);
            TheDebugHandler.Logf("Timing","Data buffers set at %.3f msec",LoopTimer.ElapsedMsec());

            //  Metal lets us associate small amounts of data - in this case the parameter
            //  structure (TheArgs) that holds the X and Y size of the median box - with a
            //  binding index, which must also match that used in Median.metal.
            
            Encoder->setBytes(&TheArgs,sizeof(MedianArgs),2);

            //  We need to set up the grid the GPU will use - the kernel code can get the grid
            //  dimensions and will use that to get the size of the arrays - it needs to know
            //  this, so we set it simply to Nx by Ny - the GPU can handle a third dimension but
            //  we set that to 1.
            
            MTL::Size GridSize(Nx,Ny,1);
            MTL::Size ThreadGroupDims(ThreadGroupSize / ThreadWidth,ThreadWidth,1);
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
        
        //  Report on the timing, and check that we got it right.
        
        float Msec = ComputeTimer.ElapsedMsec();
        printf ("GPU took %.3f msec\n",Msec);
        printf ("Average msec per iteration for GPU = %.3f\n",Msec / float(Nrpt));
        bool FromGPU = true;
        if (Nrpt <= 0) {
            printf ("No values computed using GPU, as number of repeats set to zero.\n");
        } else {
            NoteResults(OutputArray,FromGPU,Nx,Ny,Details);
        }
        printf ("\n");

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
    
    //  Report on the timing, and check that we got it right.

    float Msec = LoopTimer.ElapsedMsec();
    printf ("CPU took %.3f msec\n",Msec);
    printf ("Average msec per iteration for CPU = %.3f (threads = %d)\n",
                                                        Msec / float(Nrpt),Threads);
    bool FromGPU = false;
    if (Nrpt <= 0) {
        printf ("No values computed using CPU, as number of repeats set to zero.\n");
    } else {
        NoteResults(OutputArray,FromGPU,Nx,Ny,Details);
    }
    printf ("\n");

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
    //  Note that fits_close_file() will close the file even if passed non-zero status.
    
    int CloseStatus = 0;
    if (Fptr && fits_close_file(Fptr,&CloseStatus)) fits_get_errstatus (CloseStatus,Error);
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
