//
//                          W i n d o w  H a n d l e r . c p p
//
//  A WindowHandler provides some very basic facilities for a simple display program
//  using the GLFW window system. The main attraction of GLFW is that is provides a
//  portable window system that can run on MacOS, Linux and Windows, and that it
//  provides support for using Vulkan to display in a window. This code has been
//  developed as part of the Mandelbrot demonstration program, but might be useful
//  for similar applications. For more details, see the comments at the start of the
//  WindowHandler.h file.
//
//  History:
//      3rd Nov 2023. First working version, based somewhat on the example code in
//                    vulkan-tutorial. KS.
//     22nd Feb 2024. First version with proper header comments, following testing
//                    on the full trifecta of MacOS, Linux and Windows, and revision
//                    of the code that tries to keep to a specified frame rate. KS.
//     23rd Feb 2024. Minor tweak to get a clean compilation under Windows. KS.
//      5th Sep 2024. Split MouseCallback() into MouseButtonCallback() and
//                    MouseMovedCallback(). KS.

#include "WindowHandler.h"

#include <string>
#include <stdio.h>
#include <vector>

//  C_MaxFramesPerSec is the maximum frame rate the program will try to reach.
//  It probably won't actually run at this frame rate, no matter how fast the GPU/CPU
//  code, because of assorted overheads that we don't try to allow for. Keeping this
//  low (say 60) lowers the idling load on the system, at the expense of the frame
//  rate that can be reached when, say, zooming.

static const float C_MaxFramesPerSec = 120.0;

WindowHandler::WindowHandler()
{
    I_LastDrawMsec = 0.0;
    I_Width = 0;
    I_Height = 0;
    I_Window = nullptr;
    I_Instance = VK_NULL_HANDLE;
    I_Surface = VK_NULL_HANDLE;
    I_DrawCallbackPtr = nullptr;
    I_DrawCallbackData = nullptr;
    I_KeyCallbackPtr = nullptr;
    I_KeyCallbackData = nullptr;
    I_ResizeCallbackPtr = nullptr;
    I_ResizeCallbackData = nullptr;
    I_MouseButtonCallbackPtr = nullptr;
    I_MouseButtonCallbackData = nullptr;
    I_MouseMovedCallbackPtr = nullptr;
    I_MouseMovedCallbackData = nullptr;
    I_ScrollCallbackPtr = nullptr;
    I_ScrollCallbackData = nullptr;
}

WindowHandler::~WindowHandler()
{
    Cleanup();
}

void WindowHandler::Cleanup()
{
    if (I_Instance) {
        if (I_Surface != VK_NULL_HANDLE) vkDestroySurfaceKHR(I_Instance,I_Surface,nullptr);
    }
    I_Instance = VK_NULL_HANDLE;
    I_Surface = VK_NULL_HANDLE;
    if (I_Window) {
        glfwDestroyWindow(I_Window);
        glfwTerminate();
    }
    I_Window = nullptr;
}

void WindowHandler::InitWindow(int Width,int Height,const std::string& Name)
{
    glfwInit();

    //  GLFW was originally designed to work with OpenGL. This tells it that it's not being
    //  used that way this time.
    
    glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);

    //  Create the window, with the specified dimensions and window label.
    
    I_Window = glfwCreateWindow(Width,Height,Name.c_str(),nullptr,nullptr);
    
    //  Set the callback to be taken when the window is resized. This has to be a static
    //  routine, but we can set it up so it gets passed the address of this instance and it
    //  can use that to call a non-static routine which can do much more. We also set up
    //  callbacks for keyboard, mouse, and scrolling input.
    
    glfwSetInputMode(I_Window,GLFW_LOCK_KEY_MODS,GLFW_FALSE);
    glfwSetWindowUserPointer(I_Window,this);
    glfwSetFramebufferSizeCallback(I_Window,WindowResizedCallback);
    glfwSetKeyCallback(I_Window,KeypressCallback);
    glfwSetMouseButtonCallback(I_Window,MouseButtonCallback);
    glfwSetCursorPosCallback(I_Window,MouseMovedCallback);
    glfwSetScrollCallback(I_Window,ScrollCallback);
    
    I_Width = Width;
    I_Height = Height;
}

void WindowHandler::CreateSurface(VkInstance InstanceHndl)
{
    if (InstanceHndl != VK_NULL_HANDLE) {
        if (glfwCreateWindowSurface(InstanceHndl,I_Window,nullptr,&I_Surface) != VK_SUCCESS) {
            printf ("failed to create window surface\n");
        }
        I_Instance = InstanceHndl;
    }
}

VkSurfaceKHR WindowHandler::GetSurface()
{
    return I_Surface;
}


std::vector<const char*>& WindowHandler::GetWindowExtensions()
{
    static std::vector<const char*> ExtensionNames;
    ExtensionNames.clear();
    uint32_t ExtensionCount = 42;
    const char** GlfwExtensions = glfwGetRequiredInstanceExtensions(&ExtensionCount);
    for (uint32_t I = 0; I < ExtensionCount; I++) ExtensionNames.push_back(GlfwExtensions[I]);
    return ExtensionNames;
}

void WindowHandler::SetDrawCallback(
                    void (*RoutinePtr)(void*),void* UserData)
{
    I_DrawCallbackPtr = RoutinePtr;
    I_DrawCallbackData = UserData;
}

void WindowHandler::SetKeyCallback(void (*RoutinePtr)(
    int Key,int Scancode,int Action,int Mods,double Xpos,double Ypos,void* UserData),void* UserData)
{
    I_KeyCallbackPtr = RoutinePtr;
    I_KeyCallbackData = UserData;
}

void WindowHandler::SetResizeCallback(void (*RoutinePtr)(
                                double Width,double Height,void* UserData),void* UserData)
{
    I_ResizeCallbackPtr = RoutinePtr;
    I_ResizeCallbackData = UserData;
}

void WindowHandler::SetMouseButtonCallback(void (*RoutinePtr)(
                double Xpos,double Ypos,int Button,int Action,void* UserData),void* UserData)
{
    I_MouseButtonCallbackPtr = RoutinePtr;
    I_MouseButtonCallbackData = UserData;
}

void WindowHandler::SetMouseMovedCallback(void (*RoutinePtr)(
                                double Xpos,double Ypos,void* UserData),void* UserData)
{
    I_MouseMovedCallbackPtr = RoutinePtr;
    I_MouseMovedCallbackData = UserData;
}

void WindowHandler::SetScrollCallback(void (*RoutinePtr)(
            double DeltaX,double DeltaY,double AtX,double AtY,void* UserData),void* UserData)
{
    I_ScrollCallbackPtr = RoutinePtr;
    I_ScrollCallbackData = UserData;
}

void WindowHandler::MainLoop()
{
    float TickMsec = 1000.0f/C_MaxFramesPerSec;
    while (!glfwWindowShouldClose(I_Window)) {
        float NowMsec = I_Timer.ElapsedMsec();
        float MsecSinceLast = NowMsec - I_LastDrawMsec;
        float WaitMsec = TickMsec - MsecSinceLast;
        if (WaitMsec < 0.0) WaitMsec = 0.0;
        glfwWaitEventsTimeout(WaitMsec / 1000.0);
        I_LastDrawMsec = I_Timer.ElapsedMsec();
        DrawFrame();
    }
}

void WindowHandler::SetTitle(const char* Title)
{
    if (I_Window) glfwSetWindowTitle(I_Window,Title);
}


void WindowHandler::DrawFrame()
{
    if (I_DrawCallbackPtr)(*I_DrawCallbackPtr)(I_DrawCallbackData);
}

void WindowHandler::WindowResizedCallback(GLFWwindow* Window,int NewWidth,int NewHeight)
{
    //  This static routine is a thin layer that gets the address of the instance that set up
    //  the callback and uses that to call a non-static routine which can do the real work.
    
    WindowHandler* Instance = reinterpret_cast<WindowHandler*>(glfwGetWindowUserPointer(Window));
    if (Instance) Instance->WindowResized(NewWidth,NewHeight);
}

void WindowHandler::WindowResized(int NewWidth,int NewHeight)
{
    //  Confusingly, GLFW passes the framebuffer size for the window, not the height and width
    //  in pixel coordinates. These are not the same for a retina display, and we really want
    //  to work in pixel coordinates, which we can get using glfwGetWindowSize().
    
    int Width,Height;
    glfwGetWindowSize(I_Window,&Width,&Height);
    if (I_ResizeCallbackPtr) {
        (*I_ResizeCallbackPtr)(double(Width),double(Height),I_ResizeCallbackData);
    }
    I_Width = Width;
    I_Height = Height;
    DrawFrame();
}

void WindowHandler::KeypressCallback(GLFWwindow* Window,int Key,int Scancode,int Action,int Mods)
{
    WindowHandler* Instance = reinterpret_cast<WindowHandler*>(glfwGetWindowUserPointer(Window));
    double Xpos,Ypos;
    glfwGetCursorPos(Window,&Xpos,&Ypos);
    if (Instance) Instance->Keypress(Key,Scancode,Action,Mods,Xpos,Ypos);
}

void WindowHandler::Keypress(int Key,int Scancode,int Action,int Mods,double Xpos,double Ypos)
{
    if (I_KeyCallbackPtr) {
        Ypos = double(I_Height) - Ypos;
        (*I_KeyCallbackPtr)(Key,Scancode,Action,Mods,Xpos,Ypos,I_KeyCallbackData);
    }
}

void WindowHandler::MouseButtonCallback(GLFWwindow* Window,int Button,int Action,int Mods)
{
    WindowHandler* Instance = reinterpret_cast<WindowHandler*>(glfwGetWindowUserPointer(Window));
    double Xpos,Ypos;
    glfwGetCursorPos(Window,&Xpos,&Ypos);
    if (Instance) Instance->MouseButton(Button,Action,Mods,Xpos,Ypos);
}

void WindowHandler::MouseButton(int Button,int Action,int Mods,double Xpos,double Ypos)
{
    if (I_MouseButtonCallbackPtr) {
        Ypos = double(I_Height) - Ypos;
        (*I_MouseButtonCallbackPtr)(Xpos,Ypos,Button,Action,I_KeyCallbackData);
    }
}

void WindowHandler::MouseMovedCallback(GLFWwindow* Window,double Xpos,double Ypos)
{
    WindowHandler* Instance = reinterpret_cast<WindowHandler*>(glfwGetWindowUserPointer(Window));
    if (Instance) Instance->MouseMoved(Xpos,Ypos);
}

void WindowHandler::MouseMoved(double Xpos,double Ypos)
{
    if (I_MouseMovedCallbackPtr) {
        if (Xpos >= 0.0 && Xpos <= I_Width && Ypos >= 0.0 && Ypos <= I_Height) {
            Ypos = double(I_Height) - Ypos;
            (*I_MouseMovedCallbackPtr)(Xpos,Ypos,I_KeyCallbackData);
        }
    }
}

void WindowHandler::ScrollCallback(GLFWwindow* Window,double XOffset,double YOffset)
{
    WindowHandler* Instance = reinterpret_cast<WindowHandler*>(glfwGetWindowUserPointer(Window));
    double Xpos,Ypos;
    glfwGetCursorPos(Window,&Xpos,&Ypos);
    if (Instance) Instance->Scroll(XOffset,YOffset,Xpos,Ypos);
}

void WindowHandler::Scroll(double XOffset,double YOffset,double XPos,double YPos)
{
    if (I_ScrollCallbackPtr) {
        YPos = double(I_Height) - YPos;
        (*I_ScrollCallbackPtr)(XOffset,YOffset,XPos,YPos,I_ScrollCallbackData);
    }
}

/*
                                P r o g r a m m i n g  N o t e s
      
    o   I'm not completely happy with the frame rate code. Really, the main loop should
        only need to call DrawFrame() when the underlying display code is doing some sort
        of automatic re-display (eg a timed zoom in or out) and needs to be called on
        a regular basis, ideally at a specific frame rate. I feel one option might be
        to allow the lower-levels to make a call to the WindowHandler saying "I would
        like to start being invoked on a regular basis at such-and-such a frame rate",
        and, of course, to cancel this. Then the main code would only need to use
        glfwWaitEventsTimeout() when that was the case, and would set the timeout on
        the basis of the requested frame rate. If this were done, the lower-levels would
        be responsible for handling their own need to redraw once as a result of changes
        following, for exmaple, user input. The main advantage of this would be that the
        main loop could use glfwWaitEvents() most of the time and so avoid polling and
        the load this can put on the system.
 
 */
