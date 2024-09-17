//
//                          W i n d o w  H a n d l e r . h
//
//  A WindowHandler provides some very basic facilities for a simple display program
//  using the GLFW window system. The main attraction of GLFW is that is provides a
//  portable window system that can run on MacOS, Linux and Windows, and that it
//  provides support for using Vulkan to display in a window. This code has been
//  developed as part of the Mandelbrot demonstration program, but might be useful
//  for similar applications.
//
//  The sequence required to set up a GLFW window for use with Vulkan is slightly
//  intricate.
//  1) Create a WindowHandler, and call its InitWindow() method. This creates the
//     window.
//  2) Call the WindowHandler's GetWindowExtensions() to find out what Vulkan
//     extensions are required by a GLFW window.
//  3) Now that we know the required extensions, it will be possible to create a
//     Vukan instance that supports these extensions (or to discover that they aren't
//     supported, which would be a pity).
//  4) Get the Vulkan instance that has been created, and pass it to the WindowHandler's
//     CreateSurface(). This will create a surface that Vulkan can use for display.
//  5) Get the surface in question by calling the WindowHandler's GetSurface(), and
//     pass this to Vulkan, which will be able to use it to discover the modes and
//     formats the surface supports, and will be able to set up things like its
//     swap chain accordingly.
//
//  History:
//      3rd Nov 2023. First working version, based somewhat on the example code in
//                    vulkan-tutorial. KS.
//     22nd Feb 2024. First version with proper header comments, following testing
//                    on the full trifecta of MacOS, Linux and Windows, and revision
//                    of the code that tries to keep to a specified frame rate. KS.
//      5th Sep 2024. Split MouseCallback() into MouseButtonCallback() and
//                    MouseMovedCallback(). KS.

#ifndef __WindowHandler__
#define __WindowHandler__

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "MsecTimer.h"
#include <vector>
#include <string>

class WindowHandler {
public:
    WindowHandler();
    ~WindowHandler();
    void InitWindow(int Width,int Height,const std::string& Name);
    void CreateSurface(VkInstance InstanceHndl);
    void SetDrawCallback(void (*RoutinePtr)(void*),void* UserData);
    void SetResizeCallback(void (*RoutinePtr)(
                    double Width,double Height,void* UserData),void* UserData);
    void SetMouseButtonCallback(void (*RoutinePtr)(
                    double Xpos,double Ypos,int Button,int Action,void* UserData),void* UserData);
    void SetMouseMovedCallback(void (*RoutinePtr)(
                    double Xpos,double Ypos, void* UserData),void* UserData);
    void SetKeyCallback(void (*RoutinePtr)(
        int Key,int Scancode,int Action,int Mods,double Xpos,double Ypos,void* UserData),
        void* UserData);
    void SetScrollCallback(void (*RoutinePtr)(
        double DeltaX,double DeltaY,double AtX,double AtY,void* UserData),void* UserData);
    void MainLoop();
    void SetTitle(const char* Title);
    void Cleanup();
    std::vector<const char*>& GetWindowExtensions();
    VkSurfaceKHR GetSurface();
private:
    void DrawFrame();
    static void WindowResizedCallback(GLFWwindow* Window,int NewWidth,int NewHeight);
    void WindowResized(int NewWidth,int NewHeight);
    static void KeypressCallback(GLFWwindow* Window,int Key,int Scancode,int Action,int Mods);
    void Keypress(int Key,int Scancode,int Action,int Mods,double Xpos,double Ypos);
    static void MouseButtonCallback(GLFWwindow* Window,int Button,int Action,int Mods);
    void MouseButton(int Button,int Action,int Mods,double Xpos,double Ypos);
    static void MouseMovedCallback(GLFWwindow* Window,double Xpos,double Ypos);
    void MouseMoved(double Xpos,double Ypos);
    static void ScrollCallback(GLFWwindow* Window,double XOffset,double YOffset);
    void Scroll(double XOffset,double YOffset,double XPos,double YPos);
    GLFWwindow* I_Window;
    VkSurfaceKHR I_Surface;
    VkInstance I_Instance;
    MsecTimer I_Timer;
    float I_LastDrawMsec;
    int I_Width;
    int I_Height;
    void (*I_DrawCallbackPtr)(void*);
    void* I_DrawCallbackData;
    void (*I_ResizeCallbackPtr)(double Width,double Height,void* UserData);
    void* I_ResizeCallbackData;
    void (*I_KeyCallbackPtr)(
        int Key,int Scancode,int Action,int Mods,double Xpos,double Ypos,void* UserData);
    void* I_KeyCallbackData;
    void (*I_MouseButtonCallbackPtr)(double Xpos,double Ypos,int Button,int Action,void* UserData);
    void* I_MouseButtonCallbackData;
    void (*I_MouseMovedCallbackPtr)(double Xpos,double Ypos,void* UserData);
    void* I_MouseMovedCallbackData;
    void (*I_ScrollCallbackPtr)(
                    double DeltaX, double DeltaY, double AtX, double AtY,void* UserData);
    void* I_ScrollCallbackData;
};

#endif
