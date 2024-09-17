//
//                       M y  M T K  V i e w  D e l e g a t e  .  m
//
//  A MyMTKViewDelegate provides a couple of services required by the MyMTKView used by the
//  program. It is the entity invoked by the application a) when the size of the view changes,
//  and b) when the contents of the view need to be redrawn. When the size of the view changes,
//  drawableSizeWillChange() is called, and when the view needs to be redrawn, drawInMTKView()
//  is called. The application also calls drawInMTKView() at regular intervals, determined
//  by the frame rate set (which usually defaults to 60fps).
//
//  This code implements both these routines, and a couple of other housekeeping routines that
//  allow it to be provided with information that it needs, such as details about the window that
//  contains the view and the GPU device used to draw into the view.

#import "MyMTKViewDelegate.h"
#import <Metal/Metal.h>
#import "ControllerAdaptor.h"
#import "MyMTKView.h"
#include <stdio.h>

@implementation MyMTKViewDelegate
{
    ControllerAdaptor* _pAdaptor;
    id <MTLDevice> _device;
    NSWindow* _window;
    float _height;
    float _width;
}
- (void)setDevice:(id <MTLDevice>) device
{
    _device = device;
}

- (void)setWindow:(nonnull NSWindow *) window
                width:(float)width height:(float)height
{
    //  It's possible this call isn't really needed. This information could be obtained from
    //  the view itself, using [view window] and [view.frame.size].
    
    _window = window;
    _width = width;
    _height = height;
}

- (void)drawInMTKView:(nonnull MyMTKView *)view
{
    //  This routine is called whenever the contents of the view need to be redrawn,
    //  for example because the size of the view has changed. It is, by default, also
    //  called at the frame rate set for the application, usually 60 fps.
        
    if (_pAdaptor == nil) {
        
        //  Until this is called for the first time, the rest of the infrastructue for the
        //  program concerned with what is drawn in the view - the model, renderer, and
        //  their overall controller (none of which is known about at this level anyway) -
        //  has not been needed. That first call (when _pAdaptor is still null) is the
        //  point at which the adaptor used to talk to all that code is created. It's then
        //  up to the adaptor to set up everything else. Note that it's because all this
        //  model, controller, renderer code is in C++ that the adaptor is needed.
        
       _pAdaptor = [ControllerAdaptor alloc];
        [_pAdaptor setWindow:_window];
        [_pAdaptor frameChanged:_width height:_height];
        
        //  Having created the adaptor, let the view itself know about it. Then the view
        //  - which receives events such as mouse, key press and scroll movements - can
        //  pass these events directly on to the adaptor and hence to the C++ levels.
        
        [view setAdaptor: _pAdaptor];
    }
    
    //  Normally, all we do is pass on the draw request to the adaptor, and leave it
    //  to it to pass it on to the drawing code.
    
    [_pAdaptor draw:view device:_device];
}
- (void)mtkView:(nonnull MyMTKView *)view drawableSizeWillChange:(CGSize)size
{
    //  This routine is called when the view is first created, and is also called
    //  whenever the view is resized. This routine works out the new frame size for
    //  the view, in logical pixels, and passes this on to the adaptor.
    
    //  The size passed here is possibly misleading, because on a retina screen
    //  the numbers are twice the same as the size of the view returned by frame.size,
    //  and are double the logical pixel values included in event structures passed to
    //  the view delegate. We correct for this, but since the code in drawInMTKView()
    //  checks the frame size anyway, we probably don't need this routine at all.
    
    //  This is not called when the window is moved from, for example, a non-retina
    //  screen to a retina screen. This seems a little odd to me, as the size it
    //  reports (in native, physical, pixels) will have doubled, so has changed.
    //  However, the backing scale factor will change to compensate, so the size in
    //  logical pixels, which this routine will pass on to the adaptor, does not
    //  change - which I suppose is why it doesn't get called, but it still seems
    //  odd.
    
    float Scale = [[view window] backingScaleFactor];
    _width = size.width / Scale;
    _height = size.height / Scale;
    [_pAdaptor frameChanged:_width height:_height];
}

@end

