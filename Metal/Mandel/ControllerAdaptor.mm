//
//                         C o n t r o l l e r  A d a p t o r . m m
//
//  A ControllerAdaptor acts as a thin interface between Objective-C code that runs a application
//  and its GUI and controller code in C++ that has to react to GUI events. The interface defined
//  in ControllerAdaptor.h makes no assumptions about the nature of the C++ controller code, but
//  this implementation specifically works with a MandelController object and is designed for
//  the Madlelbrot demonstration code. Mostly, all it does is pass on the calls to that object,
//  using C++ syntax rather than Objective-C. However, it is also this code that creates the Mandel
//  controller, which it will do when the first draw() call is made.

//  History:
//     26 Feb 2024. First version after rationalisation of the original Metal & Vulkan versions. KS.
//     13 Jun 2024. Now matches changes to initialisation of MandelController introduced in Vulkan
//                  version to support use of a debug handler. KS.
//      4 Sep 2024. Added MouseMoved. KS.
//      7 Sep 2024. Added MouseUp. KS.

#import "ControllerAdaptor.h"
#import "MandelController.h"
#import "MandelArgs.h"

//  The MandelAppContact class is defined in MandelController.h and is really just an interface
//  that the MandelController can use to request interaction with the application and the GUI.
//  At the moment, this is only used to set the window title to indicate the magnification.
//  This AppContact class implements this. The adaptor has an AppContact as one of its instance
//  variables, and passes the address of this object to the MandelController so it can call it
//  directly.

class AppContact : public MandelAppContact
{
public:
    AppContact() {}
    ~AppContact() {}
    void DisplayString (const char* Title) {
        NSString* StringTitle = [NSString stringWithUTF8String: Title];
        [_window setTitle:StringTitle];
    }
    void SetWindow(NSWindow * window)
    {
        _window = window;
    }
private:
    NSWindow* _window;
};


@implementation ControllerAdaptor
{
    MandelController* _pController;
    AppContact _contact;
    float _height;
    float _width;
}

- (void)setWindow:(NSWindow *) window
{
    _contact.SetWindow(window);
}
-(void)mouseDown:(float)atX atY:(float)atY
{
    if (_pController) {
        _pController->MouseDown(atX,atY);
    }
}
-(void)mouseUp:(float)atX atY:(float)atY
{
    if (_pController) {
        _pController->MouseUp(atX,atY);
    }
}
-(void)mouseMoved:(float)atX atY:(float)atY
{
    if (_pController) {
        _pController->MouseMoved(atX,atY);
    }
}
-(void)keyDown:(const char*)key flags:(long)flags atX:(float)atX atY:(float)atY
{
    if (_pController) {
        _pController->KeyDown(key,flags,atX,atY);
    }
}
-(void)keyUp:(const char*)key flags:(long)flags atX:(float)atX atY:(float)atY
{
    if (_pController) {
        _pController->KeyUp(key,flags,atX,atY);
    }
}
-(void)scrollWheel:(float)deltaX deltaY:(float)deltaY
                                 atX:(float)atX atY:(float)atY;
{
    if (_pController) {
        _pController->ScrollWheel(deltaX,deltaY,atX,atY);
    }
}
-(void)frameChanged:(float)width height:(float)height
{
    _width = width;
    _height = height;
    if (_pController) _pController->SetViewSize(_width,_height);
}

-(void)draw:(MTKView*) view device: (id <MTLDevice>) device
{
    //  If the controller hasn't been created yet, create it now and set it up with the
    //  window size and let it know where to find the object it can use to make requests
    //  of the GUI.
    
    if (_pController == nullptr) {
        _pController = new MandelController();
        _pController->SetAppContact(&_contact);
        bool validate = false;
        MandelArgs* Args = GetMandelArgs();
        _pController->Initialise((__bridge MTL::Device *)device,
                        (__bridge MTL::Device *)device,(__bridge MTK::View *)view,
                                Args->Nx,Args->Ny,Args->Iter,validate,Args->Debug);
        _pController->SetViewSize(_width,_height);
    }
    if (_pController) _pController->Draw();
}
-(void)dealloc
{
    //  Note that dealloc() may not be called for items when the application closes.
    //  Obj-C usually assumes the operating system will do any necessary cleanup.
    
    [super dealloc];
    if (_pController) delete _pController;
}
@end

