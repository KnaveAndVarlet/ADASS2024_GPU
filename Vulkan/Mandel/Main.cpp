//
//                           M a n d e l  -  m a i n . c p p    ( Vulkan )
//
//  This is the main routine for the Mandelbrot program designed to demonstrate use of
//  a GPU for computation and also for display of the results. This code should run
//  under MacOS, Linux and as part of a Windows console app. It sets up the basic structure
//  of the program: a window set up by the GLFW library, and a MandelController that
//  handles the overall running of the program. The MandelController works with a
//  compute handler that creates the Mandelbrot images and a renderer that displays them.
//  This is more or less a standard Model-View-Controller arrangement. Since both the
//  renderer and the compute handler make use of Vulkan for GPU interaction, the window
//  set up by GLFW has to support Vulkan. Getting this set up is a touch intricate, and
//  requires some interaction between GLFW and the Vulkan basic framework used by both
//  renderer and compute handler, which messes a bit with the neat structure of the 
//  program. This main code handles that setup, then arranges the necessary callbacks
//  between the GLFW window and the controller, and then lets the program run. After that,
//  user interaction with the window will trigger the creation of new images by the
//  compute handler and their display by the renderer.
//
//  Running:
//      ./Mandel <Nx> <Ny> <Iter> <Validate> <Debug>
//
//  where:
//      Nx        (integer) is the initial size of the calculated image in X. Default 1024.
//      Ny        (integer) is the initial size of the calculated image in Y. Default 1024.
//      Iter      (boolean) is the maximum number of iterations for the calculation. Default 1024.
//      Validate  (boolean) is true if the Vulcan validation layers are to be enabled. Default true.
//      Debug     (string) is a comma-separated list of hierarchical debugging options. Default "".
//
//  History:
//       8th Nov 2024. Initial version for MacOS.
//      23rd Feb 2024. Minor changes to get a clean compilation under Windows. KS.
//      26th Feb 2024. As part of a general tidying of the Metal and Vulkan versions, the
//                     MandelController constructor calling sequence has been changed. KS.
//      11th Mar 2024. Now the basic framework supports it, call to EnableValidation() added. KS.
//       7th Jun 2024. Renamed TheFramework to TheGraphicsFramework to make things clearer. Now
//                     supports the use of a DebugHandler by the Vulkan framework. KS.
//      27th Aug 2024. Significant reworking around changes to command handler code, supporting
//                     Nx,Ny,Iter and Validate parrameters properly, allowing for exit requests
//                     and adding use of a helper class for the Debug parameter. KS.
//       5th Sep 2024. Split MouseCallback() into the two routines MouseButtonCallback() and
//                     MouseMovedCallback(). KS.
//      14th Sep 2024. Modified following renaming of Framework routines and types. KS.

#include "WindowHandler.h"
#include "MandelController.h"
#include "CommandHandler.h"

#include <string>

//  ------------------------------------------------------------------------------------------------
//
//                                  A p p  C o n t a c t
//
//  The Controller that coordinates the Renderer (which handles the display) and the Compute
//  Handler (which calculates the images) defines a MandelAppContact class that it can use to
//  communicate with the main Application code (in practice, this means with the Window Handler).
//  At the moment, all we support is allowing the Controller to supply a string to be displayed
//  in the window's title bar.

class AppContact : public MandelAppContact
{
public:
    //  Constructor - initialises variables.
    AppContact() {
        _window = nullptr;
    }
    //  Destructor - has nothing to do.
    ~AppContact() {}
    //  Called by the Controller to set the window title string.
    void DisplayString (const char* Title) {
        if (_window) _window->SetTitle(Title);
    }
    //  Called as part of setup to supply the address of the Window Handler.
    void SetWindowHandler(WindowHandler* window)
    {
        _window = window;
    }
private:
    //  The address of the WindowHandler.
    WindowHandler* _window;
};

//  ------------------------------------------------------------------------------------------------
//
//                       W i n d o w  H a n d l e r  C a l l b a c k s
//
//  These routines are called by the Window Handler in response to events such as a key press
//  or a mouse click, or when the window contents need to be redisplayed. Their addresses are
//  passed to the WindowHandler as part of the setup, together with the address of the Controller,
//  which is passed to these routines as the value of the UserData parameter.

void DrawCallback(void* UserData)
{
    MandelController* ControllerPtr = (MandelController*)UserData;
    ControllerPtr->Draw();
}

void KeyCallback(int Key,int Scancode,int Action,int Mods,double Xpos,double Ypos,void* UserData)
{
    MandelController* ControllerPtr = (MandelController*)UserData;
    char KeyString[2];
    char KeyChar = static_cast<char>(Key);
    if (isupper(KeyChar)) KeyChar = tolower(KeyChar);
    KeyString[0] = KeyChar;
    KeyString[1] = 0;
    if (Action == GLFW_PRESS) ControllerPtr->KeyDown (KeyString,Mods,float(Xpos),float(Ypos));
    if (Action == GLFW_RELEASE) ControllerPtr->KeyUp (KeyString,Mods,float(Xpos),float(Ypos));
}

void ResizeCallback(double Width,double Height,void* UserData)
{
    MandelController* ControllerPtr = (MandelController*)UserData;
    ControllerPtr->SetViewSize (Width,Height);
}

void MouseButtonCallback(double XPos,double YPos,int Button,int Action,void* UserData)
{
    MandelController* ControllerPtr = (MandelController*)UserData;
    if (Button == GLFW_MOUSE_BUTTON_LEFT) {
        if (Action == GLFW_PRESS) {
            ControllerPtr->MouseDown(float(XPos),float(YPos));
        } else {
            ControllerPtr->MouseUp(float(XPos),float(YPos));
        }
    }
}

void MouseMovedCallback(double XPos,double YPos,void* UserData)
{
    MandelController* ControllerPtr = (MandelController*)UserData;
    ControllerPtr->MouseMoved(float(XPos),float(YPos));
}

void ScrollCallback(double XOffset,double YOffset,double XPos,double YPos,void* UserData)
{
    MandelController* ControllerPtr = (MandelController*)UserData;
    ControllerPtr->ScrollWheel (float(XOffset),float(YOffset),float(XPos),float(YPos));
}

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
    void SetSingleFramework(bool Single) { I_SingleFramework = Single; }
    bool CheckValidity(const std::string& Value,std::string* Reason);
    std::string HelpText(void);
private:
    bool I_SingleFramework = false;
};

//  ------------------------------------------------------------------------------------------------
//
//                                       M a i n
//
//  The main routine has to get the values of the command line parameters, setup the various
//  components of the program, including creating the display window, introduce them to one
//  another, make sure the window handler knows what to do with events such as cursor movements
//  and the pressing of keyboard heys, and then set the main event loop going for the window. In
//  particular, it has to coordinate the Vulkan system with the window handling code in order
//  to make sure Vulkan is able to access the window propery.

int main(int Argc,char* Argv[])
{
    //  See comments below about using one Vulkan Framework for both computation and graphics.

    bool UseSingleFramework = false;
    
    //  Set up the command line handler for the various command line parameters. Note that
    //  only Debug has a sufficiently complex syntax to need a helper to validate its value.

    CmdHandler TheHandler("MandelVulkan");
    int Posn = 1;
    IntArg NxArg(TheHandler,"Nx",Posn++,"",1024,16,1024*1024,"X-dimension of computed image");
    IntArg NyArg(TheHandler,"Ny",Posn++,"",1024,16,1024*1024,"Y-dimension of computed image");
    IntArg IterArg(TheHandler,"Iter",Posn++,"",1024,16,1024*1024,"Iteration limit");
    BoolArg ValidateArg(TheHandler,"Validate",0,"",true,"Enable Vulkan validation layers");
    StringArg DebugArg(TheHandler,"Debug",0,"NoSave","","Debug levels");
    DebugArgHelper DebugHelper;
    DebugArg.SetHelper(&DebugHelper);
    DebugHelper.SetSingleFramework(UseSingleFramework);
    if (TheHandler.IsInteractive()) TheHandler.ReadPrevious();
    std::string Error = "";
    
    //  Parse the command line and get the various parameter values.
    
    bool Ok = TheHandler.ParseArgs(Argc,Argv);
    int Nx = NxArg.GetValue(&Ok,&Error);
    int Ny = NyArg.GetValue(&Ok,&Error);
    int Iter = IterArg.GetValue(&Ok,&Error);
    bool Validate = ValidateArg.GetValue(&Ok,&Error);
    std::string DebugLevels = DebugArg.GetValue(&Ok,&Error);
    
    //  Check the argument parsing went OK, and didn't end with an exit being requested.
    
    if (!Ok) {
        if (!TheHandler.ExitRequested()) {
            printf ("Error parsing command line: %s\n",TheHandler.GetError().c_str());
        }
    } else {
        if (TheHandler.IsInteractive()) TheHandler.SaveCurrent();
        
        //  This program uses a Compute Handler to calculate the Mandelbrot images, and a Renderer
        //  to display them. Both of these use Vulkan, and make use of a Framework to simplify the
        //  otherwise rather long and laborious Vulkan code that's needed. Because of the intricate
        //  little dance between Vulkan and the window handler needed to set up the display window,
        //  we need to create the Framework used by the Renderer at this point - see next code
        //  section. We can either get the Compute Handler to use the same Framework (and Vulkan
        //  instance) or we can leave it to create its own Framework and Vulkan instance.
        //  There's no particular advantage either way, except that it can make diagnostics clearer
        //  if two frameworks are used (we call one "VulkanGraphics" and the other "VulkanCompute",
        //  as opposed to just "Vulkan"). The addresses of the Framework(s) to be used are passed
        //  to the Controller than coordinates the Compute Handler and Renderer. Both options are
        //  supported here, mostly to show that both options work. Using the same instance may have
        //  advantages if we wanted to couple the graphics and computation code more tightly, for
        //  example to keep more operations purely on the GPU.
        
        KVVulkanFramework TheGraphicsFramework;
        KVVulkanFramework* TheComputeFrameworkPtr = nullptr;
        KVVulkanFramework* TheGraphicsFrameworkPtr = &TheGraphicsFramework;
        if (UseSingleFramework) {
            TheGraphicsFramework.SetDebugSystemName("Vulkan");
            TheComputeFrameworkPtr = TheGraphicsFrameworkPtr;
        } else {
            TheGraphicsFramework.SetDebugSystemName("VulkanGraphics");
        }
        TheGraphicsFramework.SetDebugLevels(DebugLevels);
        
        //  Now we set up the window used for the display.
        
        bool StatusOK = true;
        WindowHandler TheWindowHandler;
        AppContact TheAppContact;
        
        //  The order here matters - the window must be created before GetWindowExtensions(), can
        //  be called, and AddInstanceExtensions() must be called before CreateVulkanInstance().
        //  Then the Vulkan instance is needed in order for the WindowHandler's CreateSurface()
        //  to be called.  And the window surface must be known in order to call the framework's
        //  EnableGraphics().
        
        TheWindowHandler.InitWindow(512,512,"Mandelbrot using Vulkan");
        TheGraphicsFramework.AddInstanceExtensions(TheWindowHandler.GetWindowExtensions(),StatusOK);
        TheGraphicsFramework.EnableValidation(Validate);
        TheGraphicsFramework.CreateVulkanInstance(StatusOK);
        TheWindowHandler.CreateSurface(TheGraphicsFramework.GetInstance());
        TheGraphicsFramework.EnableGraphics(TheWindowHandler.GetSurface(),StatusOK);
        TheGraphicsFramework.FindSuitableDevice(StatusOK);
        TheGraphicsFramework.CreateLogicalDevice(StatusOK);
        
        //  Create the Controller that creates and coordinates the Renderer and the Compute handler.
        
        MandelController* TheControllerPtr = new MandelController();
        TheControllerPtr->Initialise(TheComputeFrameworkPtr,TheGraphicsFrameworkPtr,
                                (MandelRendererView*)nullptr,Nx,Ny,Iter,Validate,DebugLevels);
        
        //  Set up the various callbacks for the Window handler. These all invoke the relevant
        //  action in the controller, passing it any details (mouse position, key details, etc)
        //  that it might need.
        
        TheWindowHandler.SetDrawCallback(DrawCallback,TheControllerPtr);
        TheWindowHandler.SetKeyCallback(KeyCallback,TheControllerPtr);
        TheWindowHandler.SetResizeCallback(ResizeCallback,TheControllerPtr);
        TheWindowHandler.SetMouseButtonCallback(MouseButtonCallback,TheControllerPtr);
        TheWindowHandler.SetMouseMovedCallback(MouseMovedCallback,TheControllerPtr);
        TheWindowHandler.SetScrollCallback(ScrollCallback,TheControllerPtr);
        
        TheAppContact.SetWindowHandler(&TheWindowHandler);
        TheControllerPtr->SetAppContact(&TheAppContact);
        
        if (StatusOK) {
            TheWindowHandler.MainLoop();
        }
        
        //  WindowHandler::Cleanup() makes use of the Vulkan instance, so needs to be called before
        //  the Vulkan framework is cleaned up. But the Vulkan graphics (particularly the swap 
        //  chain) needs to be cleaned up before WindowHandler::Cleanup() destroys the Vulkan
        //  surface it uses.
        
        TheGraphicsFramework.CleanupVulkanGraphics();
        TheWindowHandler.Cleanup();
        TheGraphicsFramework.CleanupVulkan();
        if (TheControllerPtr) delete TheControllerPtr;
    }
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
    //  by at least one debug handler. However, we potentially have four in use. The Renderer
    //  and Compute Handler each have their own, and the Vulkan Frameworks they use also have one,
    //  and these haven't even been created at the time the parameters need to be validated.
    //  Moreover, we may be using just one framework for both Renderer and Compute handler.
    
    bool Valid = true;
            
    //  What we can do is create a temporary debug handler and set it up in turn as if it were
    //  each of the three or four debug handlers in use. The Framework, Renderer and Compute
    //  handler all have static GetDebugOptions() calls that supply the list of debug options
    //  that they recognise. We pass the supplied list of options to the CheckLevels() routine
    //  and any unrecognised ones are returned. Once we have done this with our local debug handler
    //  standing in for each of those actually used, passing what one handler did not recognise
    //  on to the next, if we are not left with a blank string, then some options are
    //  unrecognised by any handler.
    
    std::string Unrecognised = Value;
    DebugHandler StandInHandler;
    
    //  First, the Vulkan frameworks, one or two depending on how many are in use. (We assume
    //  I_SingleFramework has been set through a call to SetSingleFramework() by the main code.)
    
    if (I_SingleFramework) {
        StandInHandler.SetSubSystem("Vulkan");
        StandInHandler.SetLevelNames(KVVulkanFramework::GetDebugOptions());
        Unrecognised = StandInHandler.CheckLevels(Unrecognised);
    } else {
        StandInHandler.SetSubSystem("VulkanCompute");
        StandInHandler.SetLevelNames(KVVulkanFramework::GetDebugOptions());
        Unrecognised = StandInHandler.CheckLevels(Unrecognised);
        StandInHandler.SetSubSystem("VulkanGraphics");
        StandInHandler.SetLevelNames(KVVulkanFramework::GetDebugOptions());
        Unrecognised = StandInHandler.CheckLevels(Unrecognised);
    }
    
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
    if (I_SingleFramework) {
        Text += "Vulkan  level options: " + KVVulkanFramework::GetDebugOptions() + "\n";
    } else {
        Text += "VulkanGraphics  level options: " + KVVulkanFramework::GetDebugOptions() + "\n";
        Text += "VulkanCompute   level options: " + KVVulkanFramework::GetDebugOptions() + "\n";
    }
    Text += "(Should be a comma-separated list of options. '*' acts as a wildcard).";
    return Text;
}

/*                              P r o g r a m m i n g   N o t e s
 
    o   GLFW reports all cursor clicks and movements while the window is active irrespective
        of whether the cursor is within the window or not. It isn't clear whether it would
        be better for MouseButtonCallback() and MouseMovedCallback() to test for this or
        not. Should clicks outside the window be filtered out at this level?
  
 */
