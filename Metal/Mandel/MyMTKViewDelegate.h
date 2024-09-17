//
//                       M y  M T K  V i e w  D e l e g a t e  .  h
//
//  A MyMTKViewDelegate provides a couple of services required by the MyMTKView used by the
//  Metal version of the Mandel GPU demonstration program. It is the entity invoked by the
//  application a) when the size of the view changes, and b) when the contents of the view
//  need to be redrawn. This .h file defines its interface.

#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface MyMTKViewDelegate : NSObject <MTKViewDelegate>
- (void)drawInMTKView:(nonnull MTKView *)view;
- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size;
- (void)setDevice:(nonnull id <MTLDevice>) device;
- (void)setWindow:(nonnull NSWindow *) window
            width:(float)width height:(float)height;
@end

