// -------------------------------------------------------------------------------------------------
//
//                                 M y  M T K  V i e w . h
//
//  This is the interface file for MyMTKView, which inherits from MTKView, and implements a
//  customised view that knows how to respond to user input for the Metal version of the Mandel
//  GPU demonstration program. (The implementation code for MyMTKView is currently in main.m,
//  but really ought to be in a separate file.)
//
//  History:
//       1st Sep 2024. Astonishingly, the first addition of this history section to the
//                     header comments. Oops! This is now functionally complete, at least as
//                     needed for the ADASS demonstration. KS.
//       4th Sep 2024. Added enableMouseMoves. KS.

@interface MyMTKView : MTKView

//  MyMTKView overrides a number of the standard MTKView routines such as mouseDown(), but
//  it's important not to declare those here,  or it thinks we have our own routines with
//  those names; it doesn't think we're overriding the base routines. The only new routines
//  provided by MyMTKView, which does need to be declared here, are setAdaptor() and
//  enableMouseMoves().

- (void) setAdaptor:(ControllerAdaptor*) pAdaptor;
- (void) enableMouseMoves;

@end
