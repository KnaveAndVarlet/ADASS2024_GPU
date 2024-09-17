//
//                                     M a i n . m
//
//  This is the main routine for an example program that displays the Mandelbrot set and
//  allows the user to zoom in and to move around in the set. It is intended as an exmaple
//  of using the Metal API both for display and for GPU computation. Computing the Mandelbrot
//  set is a compute-intensive task, and using GPUs for this has significant benefits.
//
//  The structure of this code is a little unusual. The intention was to provide an example
//  program that could be built on any Macintosh that had the usual XCode tools - make and
//  clang, mainly - and did not require experience with XCode development. I also wanted
//  as much of the code to be in C++ rather than objective-c or Swift, as I thought my
//  particular target audience was more likely to be familiar with C++.
//
//  Apple provide a C++ interface to the Metal API, referred to as metal-cpp. This is
//  very useful, and provides interfaces to a basic set of the Cocoa API that it uses to
//  implement a quite impressive set of example programs. However, this seems to be lacking
//  a number of features I needed. For example, I could see no easy way to get access to
//  the mouse or scroll wheel.
//
//  So this code uses Objective-C to provide the basic structure needed to put up a simple
//  window, a very basic menu, and a view within that window that can be drawn into using
//  Metal. This view - which inherits from the MetalKit MTKView - can accept mouse and
//  keyboard input and can pass this on to the main C++ code. It does so through an adaptor
//  class written in Objective-C++, which can interface between the Objective-C and the
//  C++ parts of the code.
//
//  The main routine creates a new shared application, and provides it with an application
//  delegate that is notified of progress as the application loads. This application delegate
//  creates a very basic menu (only a Quit option, but this can be extended) and when the
//  application has finished launching, it creates the one window, which includes a view that
//  inherits from MTKView, allowing Metal to display into it. This view can override routines
//  such as mouseDown(), scrollWheel(), and keyDown(), allowing it to be notified of events
//  such a smouse movement, scroll wheel movement, and keyboard keys being pressed.
//
//  An MTKView works with a view delegate, which implements the MTKViewDelegate protocol, and
//  it is this that is invoked when the size of the view changes, or when there is a need
//  to draw into the view. In this code, it is the view delegate (whose code is in the files
//  MyMTKViewDelegate.mm/.h) that creates the adaptor object that will inteface between the
//  Objective-C part of the appplication and the C++ code that actually computes the image of
//  the Mandelbrot set and which displays it in the view. The view delegate does this the
//  first time its drawInMTKView() action is invoked, and it passes the address of the
//  newly created adaptor back to the view, so that the view can work directly with the
//  adaptor from then on - when it wants to pass on things such as scroll wheel changes or
//  key presses.
//
//  History:
//       1st Sep 2024. Astonishingly, the first addition of this history section to the
//                     header comments. Oops! This is now functionally complete, at least as
//                     needed for the ADASS demonstration. KS.
//       4th Sep 2024. Added support for continuous monitoring of mouse movements, to
//                     support display of Mandelbrot paths. KS.
//       7th Sep 2024. Now also monitors mouse movements when the mouse is used to drag
//                     the image in the window. Programming Notes tidied up. KS.
//
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <stdio.h>
#import "MandelArgs.h"

#import "ControllerAdaptor.h"

#import "MyMTKViewDelegate.h"

#import "MyMTKView.h"

// -------------------------------------------------------------------------------------------------
//
//                                   M y  M T K  V i e w
//
//  This inherits from MTKView, and implements a customised view that knows how to respond to
//  user input. It essentially passes all user input events on to the adaptor which in turn
//  passes them on to the C++ routines that do the bulk of the work. The main program creates both
//  this view and a view delegate. The view receives notification of events such as mouse or
//  scroll wheel events and key presses, while the view delegate is invoked when the size of
//  the view changes or when a draw request occurs. The view delegate creates the adaptor
//  the first time a draw request is made (this happens automatically at the start of the
//  program) and passes its address to this view code by calling setAdaptor(). After that,
//  this view code can pass on the various events it handles to the adaptor.
//
//  The @interface code for MyMTKView is in the separate file MyMTKView.h, as this needs to be
//  included by the view delegate code. It would probably make sense if this implementation
//  code were also in a separate .m file.

@implementation MyMTKView
{
    ControllerAdaptor* _pAdaptor;
    float _backingScale;
    id _monitor_id;
}

- (void) setAdaptor:(ControllerAdaptor*) pAdaptor
{
    //  This allows the view delegate to tell this view code where to find the adaptor.
    
    _pAdaptor = pAdaptor;
}

- (BOOL) acceptsFirstResponder
{
    //  This will be called as things are being set up, and the YES response tells the
    //  system that this view code is happy to accept mouse and other events.
    
    return YES;
}

- (void) enableMouseMoves
{
    //  We have to monitor cursor movements in two separate ways. Setting a tracking area will
    //  generate mouse moved events when the cursor is moved over the window, but not when
    //  the mouse is dragged in the window. We follow dragged movements using a local monitor
    //  which sets up a 'block' (a closure routine executed in response to the move) and this
    //  can call MouseMoved.
    
    NSEvent* (^mouseDragged)(NSEvent*) = ^NSEvent*(NSEvent* event) {
        [self mouseMoved:event];
        return event;
    };
    _monitor_id = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskLeftMouseDragged
                                                                          handler:mouseDragged];

    //  Moves that don't involve dragging can be followed using a tracking area. Here, we set
    //  the options for the whole visible rectange, and the specific rectangle used here for
    //  the frame is ignored. MouseMoved() will be called automatically when the cursor moves
    //  over the window.
    
    NSRect frame = NSMakeRect(0,0,0,0);
    NSTrackingArea* trackingArea = [[NSTrackingArea alloc] initWithRect:frame
        options: (NSTrackingMouseMoved | NSTrackingActiveAlways | NSTrackingInVisibleRect)
                                                                  owner:self userInfo:nil];
    [self addTrackingArea:trackingArea];
}

- (void) mouseMoved: (NSEvent *) event
{
    CGPoint point = event.locationInWindow;
    NSPoint localPoint = [self convertPoint:point fromView:nil];
    [_pAdaptor mouseMoved:localPoint.x atY:localPoint.y];
}

- (void) mouseDown: (NSEvent *) event
{
    //  There has been a mouse click in the view. All we do is pass this on to the
    //  C++ layers that do the real work by invoking mouseDown() in the adaptor.
    //
    //  We first convert from pixel coordinates in the window to coordinates in the
    //  view. (If the view is all there is in the window, this makes no difference, but
    //  we should allow for the window having more in it.) Note that the adaptor works in
    //  logical pixels (retina devices have two physical pixels to each logical pixel).
    //  The event structure also works in logical pixels. If we wanted physical device
    //  pixels we could multiply by the backing scale held in _backingScale.
    
    CGPoint point = event.locationInWindow;
    NSPoint localPoint = [self convertPoint:point fromView:nil];
    [_pAdaptor mouseDown:localPoint.x atY:localPoint.y];
}

- (void) mouseUp: (NSEvent *) event
{
    //  There has been a mouse release in the view. All we do is pass this on to the
    //  C++ layers that do the real work by invoking mouseUp() in the adaptor.
        
    CGPoint point = event.locationInWindow;
    NSPoint localPoint = [self convertPoint:point fromView:nil];
    [_pAdaptor mouseUp:localPoint.x atY:localPoint.y];
}

- (void) keyDown: (NSEvent *) event
{
    NSString* key = event.characters;
    const char *keyChars = [key cStringUsingEncoding:[NSString defaultCStringEncoding]];
    long flags = event.modifierFlags;
    //printf ("key down '%s' flags %lx\n",keyChars,flags);
    CGPoint point = event.locationInWindow;
    NSPoint localPoint = [self convertPoint:point fromView:nil];
    [_pAdaptor keyDown:keyChars flags:flags atX:localPoint.x atY:localPoint.y];
}

- (void) keyUp: (NSEvent *) event
{
    NSString* key = event.characters;
    const char *keyChars = [key cStringUsingEncoding:[NSString defaultCStringEncoding]];
    long flags = event.modifierFlags;
    //printf ("key up '%s' flags %lx\n",keyChars,flags);
    CGPoint point = event.locationInWindow;
    NSPoint localPoint = [self convertPoint:point fromView:nil];
    [_pAdaptor keyUp:keyChars flags:flags atX:localPoint.x atY:localPoint.y];
}

- (void) scrollWheel: (NSEvent *) event
{
    float deltaX = event.deltaX;
    float deltaY = event.deltaY;
    if (deltaX != 0.0 || deltaY != 0.0) {
        CGPoint point = event.locationInWindow;
        NSPoint localPoint = [self convertPoint:point fromView:nil];
        [_pAdaptor scrollWheel:deltaX deltaY:deltaY atX:localPoint.x atY:localPoint.y];
    }
}

- (void) viewDidChangeBackingProperties
{
    //  This lets us keep track of the backing scale (2.0 for retina, 1 normally) used for
    //  the screen the view is currently displayed on (and it can be dragged from one
    //  display to another). This could be used to convert between physical and logical
    //  pixel values - for example, the locations supplied in the event structures passed
    //  to mouseDown() etc are logical pixel values and we could choose to work with
    //  physical values instead.
      
    _backingScale = [[self window] backingScaleFactor];;
}

@end

// -------------------------------------------------------------------------------------------------

//                                  M y  A p p  D e l e g a t e
//
//  The application delegate monitors the progress of the application as it launches, and
//  creates the menus (minimal at present) and the one window for the program and the
//  Metal-compatible view that forms the only content for the window. It also creates the
//  view delegate along with the view, and locates the GPU device to be used. It tells the
//  view delegate where to find the window and the GPU device. The view delegate will then
//  pass this on to the adaptor and hence to the C++ code, which needs the graphics device
//  for display and computation, and may need to contact the window to provide additional
//  feedback to the user - such as putting information in the window title.

@interface MyAppDelegate : NSResponder <NSApplicationDelegate>
- (void) applicationWillFinishLaunching: (NSNotification *)notification;
- (void) applicationDidFinishLaunching: (NSNotification *)notification;
- (BOOL) applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)application;
@end

@implementation MyAppDelegate

- (void) menuAction: (id)sender {
 NSLog(@"%@", sender);
}

- (void) applicationWillFinishLaunching: (NSNotification *)notification
{
    NSMenu *menubar = [NSMenu new];
    [NSApp setMainMenu:menubar];
    
    //  The application menu only has the one Quit item.
    
    NSMenuItem *appMenuItem = [NSMenuItem new];
    NSMenu *appMenu = [NSMenu new];
    [appMenu addItemWithTitle: @"Quit" action:@selector(terminate:) keyEquivalent:@"q"];
    [appMenuItem setSubmenu:appMenu];
    [menubar addItem:appMenuItem];
   
    //  If I ever need to add more menus, this is a template for doing so..
    /*
    NSMenuItem *fileMenuItem = [NSMenuItem new];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenu addItemWithTitle:@"New" action:@selector(menuAction:) keyEquivalent:@""];
    [fileMenu addItemWithTitle:@"Open" action:@selector(menuAction:) keyEquivalent:@""];
    [fileMenu addItemWithTitle:@"Save" action:@selector(menuAction:) keyEquivalent:@""];
    [fileMenuItem setSubmenu: fileMenu];
    [menubar addItem: fileMenuItem];
    */
}
- (void) applicationDidFinishLaunching: (NSNotification *)notification
{
    
    int width = 512;
    int height = 512;
    int xorigin = 128;
    int yorigin = 128;
    NSRect frame = NSMakeRect(xorigin,yorigin,width,height);
    NSWindow* window  = [[NSWindow alloc] initWithContentRect:frame
        styleMask:(NSWindowStyleMaskClosable|NSWindowStyleMaskTitled|
                NSWindowStyleMaskMiniaturizable|NSWindowStyleMaskResizable) 
                                                backing:NSBackingStoreBuffered defer:NO];
    id<MTLDevice> _Nullable Device = MTLCreateSystemDefaultDevice();
    NSRect frameRect = NSMakeRect(0,0,width,height);
    MyMTKView* MetalView = [[MyMTKView alloc] initWithFrame:frameRect device:Device];
    MyMTKViewDelegate* ViewDel = [[MyMTKViewDelegate alloc] init];
    [ViewDel setDevice:Device];
    [ViewDel setWindow:window width:width height:height];
    [MetalView setDelegate:ViewDel];
    [window setContentView:MetalView];
    [window setBackgroundColor:[NSColor blueColor]];
    [window makeKeyAndOrderFront:NSApp];
    [MetalView enableMouseMoves];
    [NSApp activateIgnoringOtherApps:YES];
    
    
}

- (BOOL) applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)application
{
    return YES;
}

@end

// -------------------------------------------------------------------------------------------------

//                                        M a i n
//
//  All the main code has to do is create the application and the application delegate, and
//  set the app going. After that, the app delegate will create the menus and the window and the
//  Metal-compatible view and its associate view delegate, and the application structure takes
//  over from there. The command line parameters are handled by ParseMandelArgs(), which saves
//  the supplied values of the parameters in such a way that the Mandelbrot controller code
//  can pick them up once it starts up, by calling GetMandelArgs().

int main(int argc, char *argv[]) {
    if (ParseMandelArgs (argc,argv)) {
        [NSApplication sharedApplication];
        MyAppDelegate *appDelegate = [[MyAppDelegate alloc] init];
        [NSApp setDelegate:appDelegate];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp run];
    }
}

/*  ------------------------------------------------------------------------------------------------
 
                               P r o g r a m m i n g   N o t e s
 
    o   Getting rid of the autorelease in the window creation prevented a crash on shutting down
        using the window close red button.
 
    o   Changing
        @interface MyDelegate : NSObject
        to
        @interface MyDelegate : NSObject <NSApplicationDelegate>
        meant I no longer had to put a cast with nullable in when I set the delegate in the
        main routine.

    o   The preferred frame rate can be set explicitly, but the default is always the fastest
        the device supports, which is 60 on my macbook Air. If you set a larger figure, it won't
        actually call the drawing routine faster than the default rate. I put the following in
        applicationDidFinishLaunching:
        printf ("FPS = %d\n",MetalView.preferredFramesPerSecond);
        MetalView.preferredFramesPerSecond = 120;
        and it printed 60 and the actual frame rate achieved was just a touch below 60.

    o   I was surprised that mouseMoved gets called when the cursor is moved OVER the window
        but not during a drag, without the business of setting up a monitor.
 
    o   Using the whole 'behind the back of the program' way of handling the command line
        arguments through ParseMandelArgs() feels awkward. I first added the command line
        handler to the Vulkan version, where there was no problem having the C++ main()
        program interact with a C++ command handler class. But because this version of main
        is in Obj-C this doesn't work. I'm wondering if I should really just have written
        all of this in Obj-C++ instead.
*/
