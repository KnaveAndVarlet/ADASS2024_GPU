//
//                            C o n t r o l l e r  A d a p t o r . h
//
//  A ControllerAdaptor acts as an interface between Objective-C code that runs a application
//  and its GUI and controller code in C++ that has to react to GUI events. Writing the
//  adaptor code in Objective-C++ allows it to act as an adaptor between the code written
//  in the two different languages. This makes it easier to write the body of a program in
//  C++, while letting it work with the Apple application framework. This design assumes a
//  simple application with one window (an NSWindow) set up by the Objective-C code. When the
//  content of the window needs to be redrawn, the adaptor's draw() function should be called.
//  The intention is that the C++ code will use Metal calls (using Apple's metal-cpp interface)
//  to draw into the window, and the draw() call has to be passed both the Metal MTLDevice and
//  the MTKView to be used. The controller will need to be told the size of the window, both
//  before the first draw() call and whenever the window changes. This is done through the
//  Adaptor's frameChanged() call. 
//
//  When user-interface operations such as mouse clicks, key presses, or scroll wheeel movenents
//  happen, the adaptor's mouseDown(), keyDown(), keyUp() and scrollWheel() functions can be
//  called. If the C++ code needs to interact with the window, for example to change its title,
//  it can use the adaptor to do so - for this to work, the adaptor needs to have been passed
//  the NSWindow being used, and setWindow() allows this.
//
//  History:
//      4 Sep 2024. Added MouseMoved. This file doesn't seem to have had a history section
//                  up until now. KS.
//      7 Sep 2024. Added MouseUp. KS.

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface ControllerAdaptor : NSObject
-(void)draw:(MTKView*) view device: (id <MTLDevice>) device;
-(void)mouseDown:(float)atX atY:(float)atY;
-(void)mouseUp:(float)atX atY:(float)atY;
-(void)mouseMoved:(float)atX atY:(float)atY;
-(void)keyDown:(const char*)key flags:(long)flags atX:(float)atX atY:(float)atY;
-(void)keyUp:(const char*)key flags:(long)flags atX:(float)atX atY:(float)atY;
-(void)scrollWheel:(float)deltaX deltaY:(float)deltaY
                               atX:(float)atX atY:(float)atY;;
-(void)frameChanged:(float)width height:(float)height;
-(void)setWindow:(NSWindow *) window;
@end

/*
    Note that this .h file will need to be included in the main .m file, and so cannot
    itself include (or import) any .h file that includes C++ code such as a C++ class
    definition. The saving grace here is that an obj-c .h file like this that defines
    an obj-c class doesn't need to declare the instance variables in the interface.
    They can be defined in the @implementation section in the corresponding .mm file.
 
    So ControllerAdaptor.mm can #include the C++ include file Renderer.h, which it needs
    to define the instance variable that holds the address of the Renderer it works with.
    But this ControllerAdaptor.h file doesn't need to do that.
 */
