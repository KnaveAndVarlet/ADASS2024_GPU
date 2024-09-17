//  ------------------------------------------------------------------------------------------------
//
//                                 M a n d e l  A r g s . c p p    ( Metal )
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

#include "MandelController.h"
#include "CommandHandler.h"
#include "DebugHandler.h"

#include "MandelArgs.h"

//  A single global structure holds the values of the command line arguments used by the Metal
//  version of the program.

static MandelArgs G_CommandLineArgs;

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
private:
};

//  ------------------------------------------------------------------------------------------------
//
//                               P a r s e   M a n d e l   A r g s
//
//  This is the C-callable routine that is passed the command line arguments (in Argc, Argv)
//  and uses a Command Handler to get the values of the command line parameters Nx,Ny,Iter and
//  Debug. It then saves these values in the global G_CommandLineArgs structure, which can then
//  be accessed by any routine that calls GetMandelArgs().

int ParseMandelArgs (int Argc,char* Argv[])
{
    CmdHandler TheHandler("MandelMetal");
    int Posn = 1;
    IntArg NxArg(TheHandler,"Nx",Posn++,"",1024,16,1024*1024,"X-dimension of computed image");
    IntArg NyArg(TheHandler,"Ny",Posn++,"",1024,16,1024*1024,"Y-dimension of computed image");
    IntArg IterArg(TheHandler,"Iter",Posn++,"",1024,16,1024*1024,"Iteration limit");
    StringArg DebugArg(TheHandler,"Debug",0,"NoSave","","Debug levels");
    DebugArgHelper DebugHelper;
    DebugArg.SetHelper(&DebugHelper);
    if (TheHandler.IsInteractive()) TheHandler.ReadPrevious();
    std::string Error = "";
    bool Ok = TheHandler.ParseArgs(Argc,Argv);
    int Nx = NxArg.GetValue(&Ok,&Error);
    int Ny = NyArg.GetValue(&Ok,&Error);
    int Iter = IterArg.GetValue(&Ok,&Error);
    std::string DebugLevels = DebugArg.GetValue(&Ok,&Error);
    
    int Result = 1;
    if (!Ok) {
        if (!TheHandler.ExitRequested()) {
            printf ("Error parsing command line: %s\n",TheHandler.GetError().c_str());
        }
        Result = 0;
    } else {
        if (TheHandler.IsInteractive()) TheHandler.SaveCurrent();
        G_CommandLineArgs.Nx = Nx;
        G_CommandLineArgs.Ny = Ny;
        G_CommandLineArgs.Iter = Iter;
        int Nchar = sizeof(G_CommandLineArgs.Debug);
        strncpy(G_CommandLineArgs.Debug,DebugLevels.c_str(),Nchar);
        G_CommandLineArgs.Debug[Nchar - 1] = '\0';
    }
    return Result;
}

//  ------------------------------------------------------------------------------------------------
//
//                                G e t  M a n d e l  A r g s
//
//  This returns the address of the global structure in which the values of the command line
//  arguments were saved by ParseMandelArgs().

MandelArgs* GetMandelArgs (void)
{
    return &G_CommandLineArgs;
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
    //  by at least one debug handler. However, the Renderer and Compute Handler each have their
    //  own, and these haven't even been created at the time the parameters need to be validated.
    
    bool Valid = true;
    
    //  What we can do is create a temporary debug handler and set it up in turn as if it were
    //  each of the debug handlers in use. The Renderer and Compute handler both have static
    //  GetDebugOptions() calls that supply the list of debug options that they recognise.
    //  We pass the supplied list of options to the stand-in debug handler's CheckLevels() routine
    //  and any unrecognised ones are returned. We do this with our local debug handler
    //  standing in for each of those actually used, passing what one handler did not recognise
    //  on to the next. If we are not left with a blank string, then some options were
    //  unrecognised by either handler.
    
    std::string Unrecognised = Value;
    DebugHandler StandInHandler;
    
    //  The Vulkan version has code here for the Vulkan Frameworks it uses, which have their
    //  own debug handlers. The Metal version doesn't need these.
    
    //  Then for the compute handler and the renderer. Note that this assumes we know the
    //  sub-system names these will use when setting up their debug handlers.
    
    StandInHandler.SetSubSystem("Compute");
    StandInHandler.SetLevelNames(MandelComputeHandler::GetDebugOptions());
    Unrecognised = StandInHandler.CheckLevels(Unrecognised);
    StandInHandler.SetSubSystem("Renderer");
    StandInHandler.SetLevelNames(Renderer::GetDebugOptions());
    Unrecognised = StandInHandler.CheckLevels(Unrecognised);
    
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
    Text += "Renderer level options: " + Renderer::GetDebugOptions() + "\n";
    Text += "Compute  level options: " + MandelComputeHandler::GetDebugOptions() + "\n";
    Text += "(Should be a comma-separated list of options. '*' acts as a wildcard).";
    return Text;
}

//  ------------------------------------------------------------------------------------------------

/*                              P r o g r a m m i n g   N o t e s
 
    o   This is specific to the Metal version of the code. A Vulkan version would include the
        additional 'Validate' argument, and the arg helper would have to allow for the two
        debug handlers embedded in the Renderer and Compute handler. However, a Vulkan version
        is not really needed, as the all-C++ structure of the Vulkan version removes the
        inter-language complexities that this code works around.
 */
