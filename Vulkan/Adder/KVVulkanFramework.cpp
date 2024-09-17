//
//                         K V  V u l k a n  F r a m e w o r k . c p p
//
//  This is an evolving, basic framework that is intended to provide a structure for very simple
//  Vulkan-based programs. It provides a number of fairly simple functions that provide a lot
//  of the basic bolilerplate code needed to set up Vulkan instances, buffers, pipelines etc.
//  Using this takes away a lot of the flexibility that Vulkan provides, but does provide a
//  simple way of running simple programs. It's intended as a learning exercise for me, as much
//  as anything else.
//
//  Note that, so far, this has been used only for some experimental code that combines compute
//  and graphics to implement a relatively simple set of demonstration programs. As I try to
//  use it for other things, I expect it will change, possibly significantly. At the moment, it
//  really is just a plaything for me.
//
//  Credit:
//     A lot of this code is based, sometimes very closely indeed, on the code provided
//     on the excellent www.vulkan-tutorials.com by Alexander Overvoorde and Sascha Willems. I
//     found that invaluable, and recommend it to anyone looking to get into Vulkan.
//
//   Keith Shortridge, keith@knaveandvarlet.com.au
//
//  History:
//      1st Sep 2023. First dated version. This provides enough functionality to run a simple
//                    compute function on a GPU. KS.
//      2nd Sep 2023. Added AddInstanceExtensions() and GetInstance() to allow use with the
//                    GLFW window code. KS.
//      8th Nov 2023. Finally provides the functionality to display a set of triangles in a
//                    window, which is enough to support a version of the Mandelbrot test
//                    with the same functionality as the original Metal version. KS.
//     19th Jan 2023. Added VertexType parameter to CreateGraphicsPipeline() to support
//                    options other than just a list of triangles. Also, to support building
//                    under Linux, handled the case where the portability values are not
//                    defined. Replaced a couple of int values with size_t to get a clean
//                    compilation under Linux. Set the 'cached' flag for shared buffers - this
//                    gave a huge increase in rendering speed on my Framework AMD laptop. KS.
//     30th Jan 2024. Added support for the use of 'staged' buffers, implemented using two
//                    buffers, one visible to the CPU and a local GPU buffer, explicitly
//                    synched using the new SyncBuffer() routine. KS.
//      4th Feb 2024. Added ListMemoryProperties(). KS.
//     24th Feb 2024. Various minor changes to get a clean compilation under Windows. KS.
//     26th Feb 2024. Couple of casts added to placate the gcc compiler under Linux. KS.
//      9th Mar 2024. Added support for DeviceSupportsDouble(). Support for double precision
//                    is now treated as a plus when selecting a device, and is enabled if
//                    present. KS.
//     11th Mar 2024. Added support for EnableValidation() and EnableValidationLevels(). KS.
//      3rd May 2024. Started to seriously flesh out the comments, particularly the headers
//                    for individual routines. The StageName parameter for CreateComputePipeline()
//                    is now passed as a reference. KS.
//     10th Jun 2024. Comprehensive re-commenting completed, use of the DebugHandler added to
//                    control debug logging, use of AllOK() added to provide some level of
//                    crash-proofing. KS.
//      2nd Jul 2024. Corrected a comment in SetBufferDetails() about staged buffers. KS.
//      2nd Aug 2024. A call to SetFrameBufferSize() will cause the swap chain to be recreated
//                    if the frame buffer has changed. KS.
//     13th Aug 2024. Added GetDebugOptions(). KS.
//      6th Sep 2024. Introduced DrawGraphicsFrame(), which is rather more flexible than the
//                    DrawFrame() routine, as it can handle multiple pipelines and buffer sets.
//                    DrawFrame() is now re-implemented using a call to DrawGraphicsFrame(). KS.
//     14th Sep 2024. Renamed to from VulkanFramework to KVVulkanFramework, which should make it
//                    clear this isn't a standard part of Vulkan. Added formal copyright text. KS.
//
//  Copyright (c) Knave and Varlet (K&V), (2024).
//  Significant portions of this code are based closely on code from the Vulkan-tutorial website,
//  www.vulkan-tutorials.com, by Alexander Overvoorde and Sascha Willems, which is covered by a
//  Creative Commons licence, CC-BY-SA-4.0.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//  and associated documentation files (the "Software"), to deal in the Software without
//  restriction, including without limitation the rights to use, copy, modify, merge, publish,
//  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all copies or
//  substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
//  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//  ------------------------------------------------------------------------------------------------
//
//                                 I n c l u d e  f i l e s

#include "KVVulkanFramework.h"

#include <limits>
#include <vector>
#include <string.h>
#include <assert.h>
#include <array>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <stdio.h>

#include <vulkan/vk_enum_string_helper.h>

//  This can be useful for diagnostics

#include "MsecTimer.h"

using std::cout;
using std::cerr;

//  ------------------------------------------------------------------------------------------------
//
//                                  C o n s t a n t s
//
//  I_DebugOptions is the comma-separated list of all the diagnostic levels that the debug handler
//  will recognise for the Framework. If a call to I_Debug.Log() or .Logf() is added with a new
//  level name, this new name must be added to this string.

const std::string KVVulkanFramework::I_DebugOptions =
                               "Progress,Instance,Device,Buffers,Swapchain,Properties";

//  ------------------------------------------------------------------------------------------------
//
//                          N e c e s s a r y  d e f i n i t i o n s

//  On a Mac, or any implementation which uses the portability extension scheme, the following
//  will be defined. Other systems may not even define these, so we do so here, giving them
//  dummy values. (This saves having #ifdefs in the main code, which I personally dislike.)

#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME ""
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0
#endif

//  ------------------------------------------------------------------------------------------------
//
//                                 C o n s t r u c t o r
//
//  This is the only constructor. All it does is initialise all the instance variables.

KVVulkanFramework::KVVulkanFramework(void)
{
    I_Instance = VK_NULL_HANDLE;
    I_Surface = VK_NULL_HANDLE;
    I_DebugMessenger = VK_NULL_HANDLE;
    I_SelectedDevice = VK_NULL_HANDLE;
    I_DeviceHasPortabilitySubset = false;
    I_DeviceSupportsDouble = false;
    I_EnableValidationErrors = false;
    I_EnableValidationWarnings = false;
    I_EnableValidationInformation = false;
    I_ValidationErrorFlagged = false;
    I_ErrorFlagged = false;
    I_GraphicsEnabled = false;
    I_FrameBufferWidth = 0;
    I_FrameBufferHeight = 0;
    I_SwapChainExtent = {0,0};
    I_ImageCount = 0;
    I_SwapChain = VK_NULL_HANDLE;
    I_SwapChainImageFormat = VK_FORMAT_UNDEFINED;
    I_RenderPass = VK_NULL_HANDLE;
    I_LogicalDevice = VK_NULL_HANDLE;
    I_DiagnosticsEnabled = false;
    I_QueueFamilyIndex = 0;
    
    //  This is to emphasise that we start with no Vulkan extensions that this code requires.
    //  If another part of the program - such as a windowing system like GLFW - requires specific
    //  extensions, it should call AddInstanceExtensions() before calling CreateVulkanInstance().
    //  The same applies to extensions that any logical device must provide.
    
    I_RequiredInstanceExtensions.clear();
    I_RequiredGraphicsExtensions.clear();
        
    //  Set the list of debug levels currently supported by the Debug handler. If new calls to
    //  I_Debug.Log() or I_Debug.Logf() are added, the levels they use need to be included in
    //  this comma-separated list. I_DebugOptions is a static string initialised in the constants
    //  section above.
    
    I_Debug.LevelsList(I_DebugOptions);
}

//  ------------------------------------------------------------------------------------------------
//
//                                 D e s t r u c t o r
//
//  The destructor simply releases any resources allocated by the framework.

KVVulkanFramework::~KVVulkanFramework()
{
    I_Debug.Log("Progress","Called KVVulkanFramework destructor.");
    CleanupVulkan();
}

//  ------------------------------------------------------------------------------------------------
//
//                            S e t  D e b u g  S y s t e m  N a m e
//
//  The Framework uses a DebugHandler to control diagnostic output. Different levels of diagnostic
//  can be enabled by name. Names have two levels of hierarchy; each sub-system (like this
//  Framework) has a system name and supports various named levels of diagnostic. This routine
//  allows the system name to be set - which allows more than one Framework to be operating in
//  a single program (this does work). The levels are the strings that enable or disable calls
//  to I_Debug.Log() or I_Debug.Logf(). Names like 'Setup' and 'Timing'. A program using this
//  Framework can call SetDebugLevels() to enable or disable levels, using strings like
//  "Vulkan.timing,Vulkan.setup" or "Vulkan.*". The full comma-separated list of levels used
//  is provided to the Debug handler using I_Debug.LevelsList() in the Framework constructor.

void KVVulkanFramework::SetDebugSystemName (const std::string& Name)
{
    I_Debug.SetSubSystem(Name);
}

//  ------------------------------------------------------------------------------------------------
//
//                               S e t  D e b u g  L e v e l s
//
//  As described above under SetDebugSystemName(), SetDebugLevels() is used to enable specific
//  diagnostic levels. The Levels string should be a comma-separated list of strings of the form
//  'system.level', with '*' acting as a match-all character.

void KVVulkanFramework::SetDebugLevels (const std::string& Levels)
{
    I_Debug.SetLevels(Levels);
}

//  ------------------------------------------------------------------------------------------------
//
//                               G e t  D e b u g  O p t i o n s
//
//  GetDebugOptions() returns the comma-separated list of the various diagnostic levels supported
//  by the Framework. Note that this is a static routine; it can be convenient for a program to
//  have this list available before any Framework is constructed, and it is in any case a fixed
//  list.

std::string KVVulkanFramework::GetDebugOptions(void)
{
    return I_DebugOptions;
}

//  ------------------------------------------------------------------------------------------------
//
//                              E n a b l e  V a l i d a t i o n
//
//  Normally, the first Framework routine called would be CreateVulkanInstance(), which creates
//  the 'instance' - the entity through which an application works with Vulkan. However, when
//  developing a new application, Vulkan 'validation' should first be enabled, though a call
//  to EnableValidation().
//
//  By default, Vulkan provides essentially no diagnostic output when it is in use. This means
//  parameter values, for example, are not checked by routines and if something goes wrong there
//  is very little in the way of error messages. This makes for a system with minimal overheads,
//  but little help for developers. Instead, Vulkan provides for 'validation layers' which
//  effectively insert themselves between Vulkan and the application, providing a really quite
//  comprehensive set of diagnostics. These have to be enabled before the Vulkan 'instance'
//  is created by the call to CreateVulkanInstance().
//
//  Calling this routine before CreateVulkanInstance() causes the Vulkan validation layer to
//  be activated, with error and warning messages enabled but with purely informational messages
//  suppressed. More control over which level messages are enabled and which suppressed can be
//  exercised by calling EnableValidationLevels().
//
//  Pre-requisites:
//     None. However, if this is to be called, it needs to be called before calling
//     CreateVulkanInstance() if it is to have any effect. If it is not called, by
//     default the framework will not enable any form of validation. In this context
//     'validation' means enabling use of the Vulkan validation layers.

void KVVulkanFramework::EnableValidation (bool Enable)
{
    if (Enable) {
        
        //  If any of these are set true, CreateVulkanInstance() will enable the standard
        //  VK_LAYER_KHRONOS_validation layer, directing all messages to DebugUtilsCallback().
        //  That routine will then output (or suppress) each message depending on the settings
        //  of these individual flags.
        
        I_EnableValidationErrors = true;
        I_EnableValidationWarnings = true;
        I_EnableValidationInformation = false;
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                       E n a b l e  V a l i d a t i o n  L e v e l s
//
//  Provides more specific control over the amount of validation output is seen. In most cases,
//  developers will want to see error and warning messages, but may prefer not to see the
//  potentially large number of information messages output by the Vulkan diagnostic levels.
//  This is how EnableValidation() sets things up. For more control, for example to enable
//  the information messages, or to change whether various messages classes are seen dynamically
//  as the program runs, this routine can be called.
//
//  Pre-requisites:
//     None. However, to have any effect once the Vulkan instance has been created - through
//     CreateVulkanInstance() - EnableValidation() should have been called before the call
//     to CreateVulkanInstance(). This routine cannot activate the Vulkan validation layer
//     it relies on after the instance has been created. It can, however, be called before
//     CreateVulkanInstance(), and if so calling it with any level enabled will cause
//     the instance to be created with the validation layers active.

void KVVulkanFramework::EnableValidationLevels (
                        bool EnableErrors,bool EnableWarnings,bool EnableInformation)
{
    I_EnableValidationErrors = EnableErrors;
    I_EnableValidationWarnings = EnableWarnings;
    I_EnableValidationInformation = EnableInformation;
}

//  ------------------------------------------------------------------------------------------------
//
//                                A l l  O K    (Internal routine)
//
//  This code makes use of an 'inherited status' convention, where most calls are passed a boolean
//  status variable, StatusOK. If this is false when passed, the routine is expected to return
//  immediately. If something goes wrong during execution of the routine, it should log an error
//  message and set StatusOK to false. This scheme means that if there are a set of routines being
//  called one after the other, it's possible to just test for an error at the end of the set,
//  since all subsequent routines will just do nothing. But it also allows for more detailed error
//  testing, following each routine, to give context in cases where this will help. It avoids
//  the use of exceptions - whether this is good or bad is a matter of taste, but personally I
//  don't care to have execution suddenly break out of code unexpectedly because of an exception.
//
//  This is complicated in this Framework code by the use of Vulcan validation, and by attempting
//  to provide safe (in the sense of not crashing) operation following an error. The validation
//  levels may log an error during a Vulcan routine, without this being reported to the caller.
//  Really, one feels that an error caught by the validation layers should result in bad status
//  being returned to the caller. In practice, this is a bit hit and miss. Some Vulkan routines
//  will report a validation error and then crash. There is an error code defined by Vulkan
//  called VK_ERROR_VALIDATION_FAILED_EXT, but this actually isn't there to allow a calling
//  routine to test for validation failure - it's for internal testing of the validation layers.
//  Still, we do try to catch a validation error that still lets the failed routine complete.
//
//  More to the point, this Framework isn't really intended to be a general-purpose framework
//  for Vulkan. It exists just to demonstrate some of what Vulkan can do and to let people like
//  me play with it. If one routine errors, there's probably a serious problem, and a call to
//  any subsequent routine is probably not going to work properly. So if any Framework routine
//  reports an error, an internal error flag (I_ErrorFlagged) is set, and subsequent calls will
//  simply drop through. It is possible to reset these internal error flags, but you do so at
//  your own risk. It's probably much better to close the Framework and restart from scratch.
//
//  This AllOK() routine gets all this to work. If the validation leyers report an error, then
//  LogValidationError() will be called to output the error message, and it also records (in
//  I_ValidationErrorFlagged()) that this has happened. If a non-validation errors occurs,
//  LogError() outputs an error message an sets I_ErrorFlagged. AllOK(StatusOK) tests both
//  StatusOK and the two internal error flags, and sets StatusOK false if either error is
//  flagged. So basically, "if (!AllOK(StatusOK))" is the same as "if (!StatusOK)" but with a
//  check of the internal error flags added. Since almost all Framework routines call AllOK()
//  at their start, an error effectively makes all subsequent Framework routines return
//  immediately.

bool KVVulkanFramework::AllOK (bool& StatusOK)
{
    if (I_ValidationErrorFlagged || I_ErrorFlagged) {
        StatusOK = false;
    }
    return StatusOK;
}

//  ------------------------------------------------------------------------------------------------
//
//                          C r e a t e  V u l k a n  I n s t a n c e
//
//  This routine creates the 'instance' used for communication with Vulkan. Normally, this will
//  be the first Framework routine called. It handles most of the initial setup required by Vulkan.
//  The Framework doesn't return any pointer or handle to the Vulkan instance it creates, as
//  normally all interaction with Vulkan should be through the Framework, and subverting that
//  by having the main program code interact with Vulkan behind the back of the Framework is
//  asking for trouble. However, GetInstance() can be used for cases, such as interaction with
//  the GLFW windowing system, where external code really needs to have access to the instance.
//
//  Pre-requisites:
//     None. However, if specific extensions are required to support, for example, a windowing
//     system like GLFW, then AddInstanceExtensions(). If validation is to be enabled,
//     EnableValidation() should be called before calling CreateVulkanInstance().
//
//  Parameters:
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.

void KVVulkanFramework::CreateVulkanInstance (bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;

    //  In this context, 'diagnostics' refers to using the Vulkan validation layers. If any of
    //  the levels are enabled, the validation layers will be activated when the instance is
    //  created.
    
    bool EnableDiagnostics =
       I_EnableValidationErrors || I_EnableValidationWarnings || I_EnableValidationInformation;
    
    //  A Vulkan instance is the basic way a program connects with Vulkan, and has to be the first
    //  think we create. We needf to call vkCreateInstance() and pass it a structure that has all
    //  the information Vulkan needs about how it will be used by the application. Many Vulkan
    //  structures, like this, have a .stype item that identifies their type explicitly, so we
    //  set that first.
    
    VkInstanceCreateInfo CreateInfo{};
    CreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

    //  Now we have to fill in the rest of the structure. Amongst other, perhaps more important,
    //  things, it has a sub-structure that describes the application itself. Most of this isn't
    //  particularly crucial, and we set up a straightforward structure here.

    VkApplicationInfo AppInfo{};
    AppInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    AppInfo.pApplicationName = "Vulkan application";
    AppInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    AppInfo.pEngineName = "None";
    AppInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    AppInfo.apiVersion = VK_API_VERSION_1_1;
    CreateInfo.pApplicationInfo = &AppInfo;

    //  When we create a Vulkan instance, we need to specify the extensions we want to enable, and
    //  any layers (levels that support things like validation, profiling, debugging etc) that
    //  we might want to use. We specify these as a set of extension names and a set of layer
    //  names. We start out with both sets empty, but will add ones we want (or need).
    
    std::vector<const char *> EnabledLayers;
    std::vector<const char *> EnabledExtensions;

    //  It helps to know what extensions and layers are available, so first of all, we find out.
    
    //  First. the layers. We need to find out how many there are, then allocate space to get
    //  their properties (as a set of VkLayerProperties structures, which includes their names)
    //  and then fill that set of structures. Calling vkEnumerateInstanceLayerProperties()
    //  with a null pointer for the data just returns the number of properties. We call this
    //  once to get that number, which we then use to create a vector large enough to hold
    //  a set of VkLayerProperties structures to receive the properties for each family,
    //  and then call it again to fill up that vector. This scheme - first getting the number
    //  of items, creating a vector of such items, and then calling the same routine again
    //  to fill the vector - is used a lot in Vulkan.
    
    uint32_t NumberLayers;
    vkEnumerateInstanceLayerProperties(&NumberLayers,nullptr);
    std::vector<VkLayerProperties> LayerProperties(NumberLayers);
    vkEnumerateInstanceLayerProperties(&NumberLayers,LayerProperties.data());
    if (I_Debug.Active("Instance")) {
        for (const VkLayerProperties& Property : LayerProperties) {
            I_Debug.Logf("Instance","Layer: %s",Property.layerName);
        }
    }
    
    //  Now the same sequence to get the details (including the names) of the extensions.

    uint32_t NumberExtensions;
    vkEnumerateInstanceExtensionProperties(nullptr,&NumberExtensions,nullptr);
    std::vector<VkExtensionProperties> ExtensionProperties(NumberExtensions);
    vkEnumerateInstanceExtensionProperties(nullptr,&NumberExtensions,ExtensionProperties.data());
    if (I_Debug.Active("Instance")) {
        for (const VkExtensionProperties& Property : ExtensionProperties) {
            I_Debug.Logf("Instance","Extension: %s",Property.extensionName);
        }
    }
    
    //  Now, what extensions do we need? If it's available, we want to enable the portability
    //  enumeration extension, particularly if we're running on a Mac, because this will enable
    //  us to determine if a device supports the portability subset. This becomes important when
    //  we create the device itself, see CreateDevice(). Conected with this, we also want to set a
    //  related bit in the flags passed when we create the instance.

    if (strcmp(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,"")) {
        for (const VkExtensionProperties& Property : ExtensionProperties) {
            if (!strcmp(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,Property.extensionName)) {
                EnabledExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
                CreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
                break;
            }
        }
    }

    //  There may be additional extensions required by the external code. This usually depends
    //  on the sub-system being used to handle any windows used by the program. We let a separate
    //  routine supply their names (if any) and add them to the list of extensions. Often, these
    //  will be extensions required by a window sub-system such as GLFW.
    
    I_Debug.Log ("Instance","Using Required Instance Extensions.");
    for (const char* NamePtr : I_RequiredInstanceExtensions) EnabledExtensions.push_back(NamePtr);

    //  That's the basic setup needed to set up the CreateInfo structure that gets passed
    //  to vkCreateInstance(). But this gets more complex if we want to enable some form
    //  of diagnostics, and that will need us to enable one more extension and one more
    //  layer, and there's a wrinkle about getting diagnostics out of the vkCreateInstance()
    //  call. In preparation, we clear the .pNext pointer in the CreateInfo structure. If we
    //  enable disgnostics we'll set it to point to a VkDebugUtilsMessengerCreateInfoEXT
    //  structure (DebugInfo) which we'll set up. That structure is declared here so it's
    //  still in scope if it needs to be used.
    
    CreateInfo.pNext = nullptr;
    VkDebugUtilsMessengerCreateInfoEXT DebugInfo{};
    
    if (EnableDiagnostics) {
        
        //  If we are enabling diagnostics, we will aim to use the VK_LAYER_KHRONOS_validation
        //  layer, if it is available. This is described as providing all the useful standard
        //  validation. If it isn't available should we fail with an error, or warn?**
    
        const std::vector<const char*>& LayerNames = GetDiagnosticLayers();
        for (const char* NamePtr : LayerNames) {
            for (const VkLayerProperties& Property : LayerProperties) {
                if (!strcmp(NamePtr,Property.layerName)) {
                    EnabledLayers.push_back(NamePtr);
                    break;
                }
            }
        }

        //  The extension VK_EXT_debug_utils allows you to register callbacks that will receive
        //  debugging information when things go wrong. Some earlier code used VK_EXT_debug_report,
        //  but VK_EXT_debug_utils is a more recent replacement. If this is available, and if we
        //  want the overhead of the diagnostics, we enable it. Same thing - if it's not available,
        //  what do we do?**
    
        for (const VkExtensionProperties& Property : ExtensionProperties) {
            if (!strcmp(VK_EXT_DEBUG_UTILS_EXTENSION_NAME,Property.extensionName)) {
                EnabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                break;
            }
        }

        //  Now for the slight wrinkle. Getting diagnostics out of the system using the
        //  VK_EXT_debug_utils extension normally makes use of a VkDebugUtilsMessengerEXT object,
        //  and we will set one of those up properly once we have created the Vulkan instance.
        //  But we can't do that right now, because we've not created that instance yet.
        //  Nonetheless, if we want diagnostics, we'd really like to also have diagnostics for
        //  anything that might go wrong in the vkCreateInstance() call itself. This looks like a
        //  quandary - to get diagnostics from the insteance creation step, we already need to have
        //  created an instance! There's a workaround. We can set the .pnext field of the
        //  CreateInto structure to the address of a VkDebugUtilsMessengerCreateInfoEXT structure
        //  (just the same as our eventual VkDebugUtilsMessengerEXT will use once we can create it),
        //  with the details of what to call and under what conditions).
    
        if (EnableDiagnostics) {
            SetupDebugMessengerInfo (DebugInfo);
            CreateInfo.pNext = &DebugInfo;
        }
    }
    
    //  And now, we know which extensions we need, so we can set them in the CreateInfo structure
    //  we will pass to vkCreateInstance(). We might chose at this point to check that these
    //  are all available - we know some are, because we've already looked, but we should check
    //  any added for the graphics, for example. **For the moment, we don't**
    
    CreateInfo.enabledExtensionCount = static_cast<uint32_t>(EnabledExtensions.size());
    CreateInfo.ppEnabledExtensionNames = EnabledExtensions.data();

    //  If we are enabling layers for debugging, validation, etc, we need to set these in the
    //  CreateInfo structure as well.
    
    int EnabledLayerCount = int(EnabledLayers.size());
    CreateInfo.enabledLayerCount = EnabledLayerCount;
    if (EnabledLayerCount > 0) CreateInfo.ppEnabledLayerNames = EnabledLayers.data();
    
    //  And now we can finally create the Vulkan instance.

    VkResult Result;
    I_Debug.Log("Instance","Creating Vulkan instance.");
    Result = vkCreateInstance(&CreateInfo,nullptr,&I_Instance);
    if (Result != VK_SUCCESS) {
        LogVulkanError("Failed to create instance.","vkCreateInstance",Result);
        StatusOK = false;
    } else {
        I_Debug.Log("Progress","Vulkan instance created OK.");
    }
    
    //  And now that we have an instance set up - assuming we do - we can do the final setup
    //  for diagnostics, because we can now set up the VkDebugUtilsMessengerEXT object that
    //  we need.
    
    if (StatusOK && EnableDiagnostics) {
        
        //  We can't just call the routine that creates the VkDebugUtilsMessengerEXT object
        //  because it's an extension. So we have to find the routine to call, and then call it.
        
        auto Function = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(I_Instance, "vkCreateDebugUtilsMessengerEXT");
        if (Function != nullptr) {
            
            //  If EnableDiagnostics is set, we already have the DebugInfo structure set up
            //  as needed, and we can create the required object and get its address in
            //  I_DebugMessenger.
            
            Result = Function(I_Instance,&DebugInfo,nullptr,&I_DebugMessenger);
            if (Result != VK_SUCCESS) {
                LogVulkanError("Failed to create DebugUtils Messenger.",
                                            "vkCreateDebugUtilsMessengerEXT" ,Result);
                StatusOK = false;
            }
            I_DiagnosticsEnabled = true;
        } else {
            LogError("Could not find the vkCreateDebugUtilsMessengerEXT function");
            StatusOK = false;
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                                 G e t  I n s t a n c e
//
//  Provides access to the Vulkan instance used by the Framework. Only use this if you know
//  what you're doing, or to pass to a system like GLFW that needs it to set up a surface on
//  which to display, and knows what it's doing. Working with the instance directly behind the
//  back of the framework risks subverting things, and may not end well. If the instance has
//  not been created, VK_NULL_HANDLE will be returned.
//
//  Returns:
//     (VkInstance) The opaque handle Vulkan provides to get access to the instance. In principle,
//                  only Vulkan knows what to make of this.

VkInstance KVVulkanFramework::GetInstance(void)
{
    return I_Instance;
}

//  ------------------------------------------------------------------------------------------------
//
//                                 E n a b l e  G r a p h i c s
//
//  If the Framework is going to be asked to do graphics work, rather more setup is going to
//  be required than if it is only going to be used for pure computation (which is much simpler).
//  Apart from anything else, it needs to be told the display surface that will be used for
//  the graphics. This will usually have to be supplied by a Vulkan-capable windowing system,
//  such as GLFW.
//
//  Parameters:
//     SurfaceHndl   (VkSurfaceKHR) The graphics surface to be used.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     It's a bit complicated.
//     If graphics is to be used, this must be called before FindSuitableDevice(). However, for
//     the windowing system to supply the surface to be used, it will generally have to have
//     been told what Vulkan instance is being used. The sequence is then usually something like:
//     KVVulkanFramework::CreateVulkanInstance()   to create the instance.
//     KVVulkanFramework::GetInstance() to get a handle to the instance.
//     Pass the handle to the windowing system and get it to create the surface to be used.
//     Get the surface from the windowing system.
//     KVVulkanFramework::EnableGraphics()      passing the surface and enabling graphics.
//     KVVulkanFramework::FindSuitableDevice()  to find a suitable physical device.
//     KVVulkanFramework::CreateLogicalDevice() to create a logical device for the framework to use.

void KVVulkanFramework::EnableGraphics(VkSurfaceKHR SurfaceHndl,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    if (SurfaceHndl == VK_NULL_HANDLE) {
        LogError("Graphics enabled with no surface specified.");
        StatusOK = false;
    } else {
        
        //  Record that graphics is to be used, and the surface to use. We will also need
        //  the VK_KHR_SWAPCHAIN swapchain extension to be supported.
        
        I_GraphicsEnabled = true;
        I_Debug.Log("Instance","Framework now setting window Surface");
        I_Surface = SurfaceHndl;
        //  Should we check this isn't there already?
        I_RequiredGraphicsExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                        S e t  F r a m e  B u f f e r  S i z e
//
//  The swap chain used for graphics display needs to know the size of the frame buffer being
//  used for the display, and this needs to be set before CreateSwapChain() is called. If the
//  display size changes, this routine will have to be called again, as the swap chain will need
//  to be recreated.
//
//  Parameters:
//     Width         (int) The width of the frame buffer, in pixels.
//     Height        (int) The height of the frame buffer, in pixels.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Note:
//     Vulkan allows the width to be set to a specific magic number in some circumstances. This
//     is a matter for Vulkan and the windowing system. I mention it here just in case anyone
//     sees a strange number being passed as the width. If Vulkan doesn't complain, don't worry
//     about it.

void KVVulkanFramework::SetFrameBufferSize(int Width,int Height,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    if (Width <= 0 || Height <= 0) {
        LogError("Invalid frame buffer size %d by %d specified.",Width,Height);
        StatusOK = false;
    } else {
        I_FrameBufferWidth = Width;
        I_FrameBufferHeight = Height;
        if (I_SwapChain) RecreateSwapChain(StatusOK);
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                           F i n d  S u i t a b l e  D e v i c e
//
//  This is part of the basic initialisation sequence needed for Vulkan, which is essentially
//  CreateVulkanInstance(), FindSuitableDevice() and CreateLogicalDevice(). Vulkan needs to
//  work with a GPU device, and if a system has multiple GPUs available, it needs to be told
//  which one to use. Vulkan's philosophy is to allow a program a lot of flexibility, and allows
//  a program to find out a lot about the characteristics and capabilities of each device and to
//  decide which one to use. This Framework is intended for much simpler programs that just want
//  all that done for them, and this routine just picks a device that seems suitable for
//  computation and (if EnableGraphics() has been called) graphics. Frankly, most systems will
//  only have one GPU, and this will pick it. If there are multiple GPUs available, it makes
//  what should be a reasonable choice - for example, it will pick a discrete GPU over an
//  integrated one, as that is usually the most powerful device.
//
//  Parameters:
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Pre-requisites:
//      The Vulkan instance must have been created through a call to CreateVulkanInstance().
//      If graphics are to be enabled, EnableGraphics() must have been called to let the
//      framework know what surface is going to be used for the display - this is required
//      so we can check whether the available devices provide swap chain support for that
//      surface. If, say, a windowing system has specific requirements for extensions that
//      the device has to support (as opposed to extensions the Vulkan instance has to support)
//      then AddGraphicsExtensions() should have been called.

void KVVulkanFramework::FindSuitableDevice (bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    I_Debug.Log ("Device","Searching for suitable GPU device.");
    
    if (I_Instance != VK_NULL_HANDLE) {
    
        //  We can use the Vulkan instance to get a list of all the physical devices available.
        //  There may just be one, of course, in which case we'd better hope it's suitable. However,
        //  there may be more than one, in which case we may have to make a choice.
        
        //  There may be requirements imposed by the program's graphics code. If all we want is
        //  compute capability, all devices provide this. There may be devices that don't actually
        //  support anything more than compute, and if we want to actually use our GPU for graphics
        //  we may need to enable extensions like VK_KHR_swapchain. The list of required graphics
        //  extensions (if any) is in I_RequiredGraphicsExtensions, and a program may have
        //  supplemented these if necessary by calling AddGraphicsExtensions().
                
        int HighestScore = 0;
        VkPhysicalDevice SelectedDevice = nullptr;
        bool HasPortabilitySubset = false;
        
        //  Find out how many devices there are
        
        uint32_t NumberDevices;
        vkEnumeratePhysicalDevices(I_Instance,&NumberDevices,nullptr);
        if (NumberDevices > 0) {
            
            //  Work through every device in turn.
            
            std::vector<VkPhysicalDevice> Devices(NumberDevices);
            vkEnumeratePhysicalDevices(I_Instance,&NumberDevices,Devices.data());
            for (VkPhysicalDevice Device : Devices) {
                
                if (I_Debug.Active("Device")) ShowDeviceDetails(Device);
                
                //  Checking for the required graphics extensions is easy, and we do that.
                
                uint32_t NumberExtensions;
                vkEnumerateDeviceExtensionProperties(Device,nullptr,&NumberExtensions,nullptr);
                std::vector<VkExtensionProperties> DeviceExtensions(NumberExtensions);
                vkEnumerateDeviceExtensionProperties(Device,nullptr,&NumberExtensions,
                                                                      DeviceExtensions.data());
                if (I_Debug.Active("Device")) {
                    for (const VkExtensionProperties& Property : DeviceExtensions) {
                        I_Debug.Logf("Device","Device extension: %s",Property.extensionName);
                    }
                }
                bool ExtensionSupportOK =
                            DeviceExtensionsOK(I_RequiredGraphicsExtensions,DeviceExtensions);
                
                //  If we're going to be using this for graphics, we also want to know if the
                //  device provides the necessary swap chain support.
                
                bool SwapChainSupportOK = SwapChainSupportAdequate(Device,StatusOK);
                if (!AllOK(StatusOK)) break;
                
                if (ExtensionSupportOK && SwapChainSupportOK) {

                    //  If the device passed those tests, then it satisfies the basic
                    //  requirements. Trying to gauge its suitablity for the task in hand is
                    //  going to depend on the details of what we want it to do, so we offload
                    //  that to a RateDevices() routine that in principle could be overriden
                    //  for a specific application. We keep track of which device is most
                    //  highly rated. Since at this point we have access to its list of extensions,
                    //  we also check for the portability subset - when we set up the device later
                    //  we will need to know if it supports this.
                    
                    int Score = RateDevice(Device);
                    if (Score > HighestScore) {
                        HighestScore = Score;
                        SelectedDevice = Device;
                        HasPortabilitySubset = DeviceHasPortabilitySubset(DeviceExtensions);
                    }
                }
                
            }
        }
        if (SelectedDevice) {
            
            //  Record the device we selected and record whether or not it supports double
            //  precision (that's useful for computation) and if it supports (and therefore
            //  requires) the portability subset (mainly the case for Macs).
            
            I_SelectedDevice = SelectedDevice;
            I_DeviceHasPortabilitySubset = HasPortabilitySubset;
            VkPhysicalDeviceFeatures Features;
            vkGetPhysicalDeviceFeatures(SelectedDevice,&Features);
            I_DeviceSupportsDouble = Features.shaderFloat64;
            if (I_Debug.Active("Device")) {
                VkPhysicalDeviceProperties Properties;
                vkGetPhysicalDeviceProperties(SelectedDevice,&Properties);
                I_Debug.Logf("Device","Selected Device: %s",Properties.deviceName);
            }
        } else {
            LogError("Unable to find a suitable GPU.");
            StatusOK = false;
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                           D e v i c e  S u p p o r t s  D o u b l e
//
//  Can be used to find out if the selected GPU supports double precision calculations. Many don't.
//  GPUs are typically aimed at fast single precision calculations, and often will even
//  work in half-precision, so double is not by no means a given (Apple M1 and M2 chips don't
//  support it).
//
//  Returns:
//      (bool)  True if the selected GPU supports double precision calculations.
//
//  Pre-requisites:
//      FindSuitableDevice() must have been called to select a physical GPU device.

bool KVVulkanFramework::DeviceSupportsDouble (void)
{
    //  Pre-requisites:
    //      FindSuitableDevice() must have been called to select a physical device.
    
    return I_DeviceSupportsDouble;
}

//  ------------------------------------------------------------------------------------------------
//
//                           C r e a t e  L o g i c a l  D e v i c e
//
//  This is part of the basic initialisation sequence needed for Vulkan, which is essentially
//  CreateVulkanInstance(), FindSuitableDevice() and CreateLogicalDevice(). Once a physical
//  device has been selected, a logical device needs to be created and this is what we use
//  to interact with the physical device through Vulkan. This involves enabling extensions,
//  and setting up queues for use with the device. Queues in Vulkan are complicated, and this
//  routine sets up the logical device with a very basic queue configuration - just one queue
//  that supports compute and (if enabled) graphics.
//
//  Parameters:
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Pre-requisites:
//      The physical device to use must have been selected through a call to FindSuitableDevice(),
//      which in turn means CreateVulkanInstance() must have been called, together with any other
//      optional routines such as EnableGraphics().

void KVVulkanFramework::CreateLogicalDevice (bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    I_Debug.Log ("Progress","Creating Logical Device.");

    //  Setting up the logical device - which is what we use to interact with the physical device -
    //  once we've identified the physical device to use - involves specifically enabling any
    //  extensions we need, and setting up any queues we are going to use. Eventually, we're
    //  going to create the logical device with a call to VkCreateDevice(), and we pass most of
    //  what's needed in a VkDeviceCreateInfo struture.
    
    VkDeviceCreateInfo DeviceCreateInfo{};
    DeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    //  We asssume we don't need any of the more complicated device features, and just pass a
    //  default structure that doesn't enable anything fancy. However, if the device supports
    //  double precision, we enable that.
    
    VkPhysicalDeviceFeatures EnabledDeviceFeatures{};
    if (I_DeviceSupportsDouble) EnabledDeviceFeatures.shaderFloat64 = VK_TRUE;
    DeviceCreateInfo.pEnabledFeatures = &EnabledDeviceFeatures;
    
    //  Queues are more complicated. Devices support multiple queues, bundled up into queue
    //  'families' in which each member has the same capabilities (ie can do graphics, can
    //  do compute, can do other things like memory transfers). Each queue family has its own
    //  index number. The CreateInfo structure we pass to vkCreateDevice() includes an array of
    //  VkDeviceQueueCreateInfo structures, one for each queue family we will use. Each of these
    //  specifies the index number for the queue family, and the number of queues from that
    //  family that we will use. A relatively simple program can use just one queue from a
    //  family that supports both compute and graphics, or a compute-only program can use just
    //  one queue that supports compute. A more complex program can submit multiple operations
    //  to multiple queues in multiple families. It depends on the program, so a general-purpose
    //  solution is tricky.
    
    //  For the moment, we look for just one queue family that supports both graphics (if enabled)
    //  and compute.
    
    bool UseGraphics = I_GraphicsEnabled;
    bool UseCompute = true;
    I_QueueFamilyIndex = GetIndexForQueueFamilyToUse(UseGraphics,UseCompute,StatusOK);
    
    //  Now we can fill in the single VkDeviceQueueCreateInfo structure for this family, and
    //  indicate that we'll only use one queue from that family. (Should test StatusOK ****)
    
    VkDeviceQueueCreateInfo QueueCreateInfo = {};
    QueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    QueueCreateInfo.queueFamilyIndex = I_QueueFamilyIndex;
    QueueCreateInfo.queueCount = 1;
    
    //  And we set that single structure into the overall CreateInfo structure. We need to specify
    //  relative priorities for the queues we're using, but since there's only one, that doesn't
    //  matter very much, and the priority array we submit is just one float value.
    
    DeviceCreateInfo.queueCreateInfoCount = 1;
    DeviceCreateInfo.pQueueCreateInfos = &QueueCreateInfo;
    float QueuePriorities = 1.0;
    QueueCreateInfo.pQueuePriorities = &QueuePriorities;

    //  There may be specific extensions we need to enable. If all we want is compute capability,
    //  that's there in all devices. There may be devices that don't actually support anything
    //  more than compute, and if we want to actually use our GPU for graphics we may need to
    //  enable extensions like VK_KHR_swapchain.

    std::vector<const char *> EnabledExtensions;
    I_Debug.Log ("Device","Using Required Graphics Extensions.");
    for (const char* Extension : I_RequiredGraphicsExtensions) {
        EnabledExtensions.push_back(Extension);
    }
    
    //  And if the device supports it, we have to enable the VK_KHR_portability_subset. See the
    //  comments in DeviceHasPortabilitySubset() for the gory details.
    
    if (I_DeviceHasPortabilitySubset) EnabledExtensions.push_back("VK_KHR_portability_subset");

    //  Now we can set the details of the required device entensions in the information structure.
    
    DeviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(EnabledExtensions.size());
    DeviceCreateInfo.ppEnabledExtensionNames = EnabledExtensions.data();
    
    //  If we are enabling diagnostics, we should probably set the details of the diagnostic layers
    //  in use in the information structure as well. In principle, later versions of Vulkan no
    //  longer distinguish between instance and device level diagnostics, so this should not be
    //  needed. But it doesn't hurt.
    
    if (I_DiagnosticsEnabled) {
        const std::vector<const char*>& LayerNames = GetDiagnosticLayers();
        DeviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(LayerNames.size());
        DeviceCreateInfo.ppEnabledLayerNames = LayerNames.data();
    } else {
        DeviceCreateInfo.enabledLayerCount = 0;
    }

    //  Finally, we can create the logical device.
    
    VkResult Result;
    I_Debug.Log("Device","Creating logical device");
    Result = vkCreateDevice(I_SelectedDevice,&DeviceCreateInfo,nullptr,&I_LogicalDevice);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError("Failed to create logical device.","vkCreateDevice",Result);
        StatusOK = false;
    } else {
        I_Debug.Log("Progress","Vulkan logical device created OK.");
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                             S e t  B u f f e r  D e t a i l s
//
//  Handling buffers is complicated, and is one place where the Framework provides a bit more
//  help than just acting as a wrapper for Vulkan calls. It keeps track of all the details
//  relating to a buffer internally, and provides its own opaque 'handle' for the calling program
//  to use when dealing with the buffer. SetBufferDetails() should be the first routine called
//  when a new buffer is to be created. It sets up the internal housekeeping to keep track of the
//  buffer and returns the new handle associated with that buffer. Vulkan pipelines need to know
//  many details of the buffers they will use, and these have to be set up in advance. A pipeline
//  needs to know in advance what sort of buffers it will be handling, but creation of the actual
//  buffers and allocation of the memory they will use can be deferred until later. And a pipeline
//  can be re-used with different actual buffers, so long as they match the way it was set up.
//  The code that actually runs on the GPU will use a 'binding' number to identify the buffer,
//  and the number supplied in this call must match that used by the GPU code (in the layout
//  specification in the GLSL file). The buffer details also have to describe the type of
//  buffer and the way it is accessed. Note that the size of the buffer does not have to specified
//  at this point, all that's needed are its type and the way it will be accessed. If the buffer
//  is to be used to hold vertices for graphical purposes, SetVertexBufferDetails() should be
//  called after SetBufferDetails() to add the needed vertex layout information.
//
//  Parameters:
//     Binding       (long) A number that must match the binding specified for the buffer in
//                   the 'layout' statement in the GLSL code that will use this buffer.
//     Type          (std::string&) A string describing the type of the buffer. One of:
//                   "UNIFORM" The buffer contains data common to all invocations of the
//                             GPU code. Often used to pass parameter values.
//                   "STORAGE" The buffer contains general-purpose data, often arrays.
//                   "VERTEX"  The buffer contains vertex data (used by graphics programs).
//     Access        (std::string&) A string describing the way the GPU accesses the buffer.
//                   "LOCAL"      The buffer data is local to the GPU. This is usually fast for the
//                                GPU to access, but the CPU cannot see it at all.
//                   "SHARED"     The buffer data can be accessed by both CPU and GPU. This may
//                                introduce overheads in accessing it.
//                   "STAGED_CPU" The buffer data is written into a CPU buffer and then transferred
//                                to a buffer on the GPU when it is needed.
//                   "STAGED_GPU" The buffer data is written into a GPU buffer and then transferred
//                                to a buffer on the CPU when it is needed.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Returns:
//     (KVBufferHandle) An opaque handle that can be passed to the Framework to identify the buffer.
//
//  Pre-requisites: 
//     None. This routine merely allocates space for the buffer details, records them and returns
//     a handle to them without making any Vulkan calls at all.
//
//  Notes:
//     o Although in principle you don't need to know this, the handle is merely an integer that
//     serves as an index into a vector of structures maintained by the Framework that describe
//     all the allocated buffers. The index starts from 1, allowing 0 to indicate an invalid
//     handle.
//     o The buffer handle will remain valid until the Framework is closed down or the buffer
//     is deleted through a call to DeleteBuffer(). But note that the handle for a deleted buffer
//     may be re-used for another buffer created later.

KVVulkanFramework::KVBufferHandle KVVulkanFramework::SetBufferDetails (
            long Binding,const std::string& Type,const std::string& Access,bool& StatusOK)
{
    //  Buffer details are held in I_BufferDetails[]. The 'handle' that the framework uses to
    //  refer to the buffer is in practice just the index into I_BufferDetails[] where its
    //  details are stored, plus 1 (so that a zero handle indicates an invalid or null handle).
    
    if (!AllOK(StatusOK)) return 0;

    I_Debug.Logf ("Buffers", "Setting new buffer details, binding %d, type %s, access %s",
                                                       Binding,Type.c_str(),Access.c_str());
    KVBufferHandle ReturnedHandle = 0;
    VkBufferUsageFlags UsageFlags = 0;
    VkMemoryPropertyFlags PropertyFlags = 0;
    VkBufferUsageFlags SecondaryUsageFlags = 0;
    VkMemoryPropertyFlags SecondaryPropertyFlags = 0;
    KVBufferType BufferType = TYPE_UNKNOWN;
    KVBufferAccess BufferAccess = ACCESS_UNKNOWN;
    if (Type == "UNIFORM") {
        UsageFlags |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        BufferType = TYPE_UNIFORM;
    } else if (Type == "STORAGE") {
        UsageFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        BufferType = TYPE_STORAGE;
    } else if (Type == "VERTEX") {
        UsageFlags |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        BufferType = TYPE_VERTEX;
    } else {
        LogError ("Invalid buffer type '%s' specified.",Type.c_str());
        StatusOK = false;
    }
    if (Access == "LOCAL") {
        
        //  Just one buffer, and it's a local one, visible to the GPU but not the CPU.
        
        PropertyFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        BufferAccess = ACCESS_LOCAL;
        
    } else if (Access == "SHARED") {
        
        //  Just one buffer, visible to both CPU and GPU, with access coordinated at a lower
        //  level, presumably by a combination of Vulkan, the driver and the hardware.
        
        PropertyFlags |=
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        BufferAccess = ACCESS_SHARED;
        
    } else if (Access == "STAGED_CPU") {
        
        //  Two buffers, the main one filled by the CPU and then its data transferred explicitly
        //  to the secondary one , which is a local buffer visible only to the GPU.
        
        UsageFlags |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        /* In testing so far, there doesn't seem to be any difference, coherent or not....
        PropertyFlags |=
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
         */
        PropertyFlags |=
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        SecondaryUsageFlags = UsageFlags | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        SecondaryPropertyFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        BufferAccess = ACCESS_STAGED_CPU;
        
    } else if (Access == "STAGED_GPU") {
        
        //  Two buffers, the main one is visible to the CPU, but the data is generated in the
        //  secondary one, which is a local buffer visibile only to the GPU. When it has been
        //  filled by the GPU, the data can be explicitly transferred to the main, CPU-visible,
        //  buffer.
        
        UsageFlags |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        PropertyFlags |=
                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        SecondaryUsageFlags = UsageFlags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        SecondaryPropertyFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        BufferAccess = ACCESS_STAGED_GPU;
        
    } else {
        LogError ("Invalid buffer access '%s' specified",Access.c_str());
        StatusOK = false;
    }
    if (AllOK(StatusOK)) {
        
        //  We keep track of all buffers in use in the I_BufferDetails vector. Buffers can be
        //  deleted as a program runs. When this happens, rather than remove the entry being
        //  used for that buffer from the vector (which would mess up the index values into the
        //  vector for all the remaining buffers, which would be sad as we need those index
        //  values to continue to be valid), we just flag it as not in use. First, see if there
        //  is an unused slot we can use for this new buffer. If not, we expand the buffer
        //  using resize(). Either way, I_BufferDetails ends up with a slot - whose index is
        //  given by Index - that we can use. And the handle the calling program can use to
        //  reference this buffer is just Index + 1 (so that zero can be used to indicate
        //  a non-existent entry).
        
        bool SlotFound = false;
        int Index = 0;
        for (int I = 0; I < int(I_BufferDetails.size()); I++) {
            if (!I_BufferDetails[I].InUse) {
                Index = I;
                SlotFound = true;
                break;
            }
        }
        if (!SlotFound) {
            Index = int(I_BufferDetails.size());
            I_BufferDetails.resize(Index + 1);
        }
        ReturnedHandle = Index + 1;
        
        //  Now set initial values for everything held for this new buffer. Most of these
        //  are just null values or sensible defaults.
        
        I_Debug.Logf ("Buffers","Recording buffer details at slot %d, handle %ld",
                                                                   Index,ReturnedHandle);
        T_BufferDetails BufferDetails;
        BufferDetails.InUse = true;
        BufferDetails.BufferType = BufferType;
        BufferDetails.BufferAccess = BufferAccess;
        BufferDetails.Handle = ReturnedHandle;
        BufferDetails.Binding = Binding;
        BufferDetails.SizeInBytes = 0;
        BufferDetails.MemorySizeInBytes = 0;
        BufferDetails.MappedAddress = nullptr;
        BufferDetails.MainBufferHndl = VK_NULL_HANDLE;
        BufferDetails.MainBufferMemoryHndl = VK_NULL_HANDLE;
        BufferDetails.MainUsageFlags = UsageFlags;
        BufferDetails.MainPropertyFlags = PropertyFlags;
        BufferDetails.SecondaryBufferHndl = VK_NULL_HANDLE;
        BufferDetails.SecondaryBufferMemoryHndl = VK_NULL_HANDLE;
        BufferDetails.SecondaryUsageFlags = SecondaryUsageFlags;
        BufferDetails.SecondaryPropertyFlags = SecondaryPropertyFlags;
        //  These are simply null values for the binding descriptor.
        BufferDetails.BindingDescr.binding = 0;
        BufferDetails.BindingDescr.stride = 0;
        BufferDetails.BindingDescr.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        I_BufferDetails[Index] = BufferDetails;
    }
    I_Debug.Logf ("Buffers","Buffer handle returned as %p",ReturnedHandle);
    return ReturnedHandle;
}

//  ------------------------------------------------------------------------------------------------
//
//                                C r e a t e  B u f f e r
//
//  This routine actually creates the buffer previously described in a call to SetBufferDetails().
//  In Vulkan, 'creating' a buffer involves creating what Vulkan refers to as a 'buffer', but
//  this does not actually include allocating the memory for the buffer. This routine does both,
//  doing the housekeeping for Vulkan to work with the buffer, and allocating memory for it. The
//  memory for the buffer will be allocated on the CPU or on the GPU (and in the case of 'shared'
//  access memory, two buffers are allocated, one local to the CPU, one local to the GPU). All
//  this is handled by this routine. Once created, a buffer can be deleted by a call to
//  DeleteBuffer(), or resized by a call to ResizeBuffer(). IsBufferCreated() can be used to see
//  if a buffer handle refers to a buffer that has actually been created yet or not. To access
//  the memory allocated to a buffer, call MapBuffer(), and to unmap a buffer call UnmapBuffer().
//  The routine SyncBuffer() needs to be called when it is necessary to make sure that the CPU
//  and GPU versions of a staged buffer are in sync. All buffers are automatically deleted and
//  their memory released when the Framework closes down.
//
//  Parameters:
//     BufferHandle  (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     SizeInBytes   (long) The size of the buffer in bytes.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     The basic Vulkan initialisation must have been completed - CreateVulkanInstance(),
//     FindSuitableDevice() and CreateLogicalDevice() - as Vulkan needs to know what actual
//     GPU will be using this buffer. SetBufferDetails() must have been called to describe
//     how the buffer will be used and to return the buffer handle that is passed to this
//     routine. The buffer should not have already been created - to change the size of an
//     existing buffer, use ResizeBuffer().

void KVVulkanFramework::CreateBuffer (KVBufferHandle BufferHandle,long SizeInBytes,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    int Index = BufferIndexFromHandle(BufferHandle,StatusOK);
    if (AllOK(StatusOK)) {
        if (SizeInBytes <= 0) {
            LogError ("Invalid buffer size (%d bytes) specified",SizeInBytes);
            StatusOK = false;
        }
    }
    if (AllOK(StatusOK)) {
        if (I_BufferDetails[Index].MainBufferHndl != VK_NULL_HANDLE) {
            LogError ("Attempt to create already existing buffer of %d bytes",SizeInBytes);
            StatusOK = false;
            //  We could simply extend the existing buffer - should we?
        } else {
            
            //  Create the main (and perhaps the only) actual vulkan buffer.
            
            VkBuffer Buffer;
            VkDeviceMemory BufferMemory;
            VkBufferUsageFlags UsageFlags = I_BufferDetails[Index].MainUsageFlags;
            VkMemoryPropertyFlags PropertyFlags = I_BufferDetails[Index].MainPropertyFlags;
            CreateVulkanBuffer(SizeInBytes,UsageFlags,PropertyFlags,&Buffer,
                                                      &BufferMemory,StatusOK);
            if (AllOK(StatusOK)) {
                I_Debug.Logf ("Buffers","VkBuffer %p created, size %ld bytes.",Buffer,SizeInBytes);
                I_BufferDetails[Index].SizeInBytes = SizeInBytes;
                I_BufferDetails[Index].MemorySizeInBytes = SizeInBytes;
                I_BufferDetails[Index].MainBufferHndl = Buffer;
                I_BufferDetails[Index].MainBufferMemoryHndl = BufferMemory;
                
                //  If the buffer is staged, we need to create the secondary buffer
                
                if (I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_CPU ||
                           I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_GPU) {
                    UsageFlags = I_BufferDetails[Index].SecondaryUsageFlags;
                    PropertyFlags = I_BufferDetails[Index].SecondaryPropertyFlags;
                    CreateVulkanBuffer(SizeInBytes,UsageFlags,PropertyFlags,&Buffer,
                                                            &BufferMemory,StatusOK);
                    if (AllOK(StatusOK)) {
                        I_Debug.Logf ("Buffers","Secondary VkBuffer %p created, size %ld bytes",
                                                                             Buffer,SizeInBytes);
                        I_BufferDetails[Index].SecondaryBufferHndl = Buffer;
                        I_BufferDetails[Index].SecondaryBufferMemoryHndl = BufferMemory;
                    }
                    
                }
            }
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                                D e l e t e  B u f f e r
//
//  This routine deletes a buffer previously described in a call to SetBufferDetails().
//  If the buffer has actually been created through a call to CreateBuffer() the Vulkan buffer(s)
//  and associated memory are also released. If the buffer has not yet been created, it simply
//  gets cleared from the internal tables maintained by the Framework.
//
//  Parameters:
//     BufferHndl    (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     SetBufferDetails() must have been called to describe how the buffer will be used and
//     to return the buffer handle that is passed to this routine.

void KVVulkanFramework::DeleteBuffer (KVBufferHandle BufferHndl,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    int Index = BufferIndexFromHandle(BufferHndl,StatusOK);
    if (AllOK(StatusOK)) {
        
        //  If the buffer is still mapped, unmap it before deleting it.
        
        if (I_BufferDetails[Index].MappedAddress) {
            vkUnmapMemory(I_LogicalDevice,I_BufferDetails[Index].MainBufferMemoryHndl);
            I_BufferDetails[Index].MappedAddress = nullptr;
        }
        
        //  Delete the Vulkan buffer and the actual memory associated with it.  The same for the
        //  secondary buffer, if the buffer is a staged buffer implemented using two Vulkan
        //  buffers.
        
        vkFreeMemory(I_LogicalDevice,I_BufferDetails[Index].MainBufferMemoryHndl,nullptr);
        vkDestroyBuffer(I_LogicalDevice,I_BufferDetails[Index].MainBufferHndl,nullptr);
        if (I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_CPU ||
                  I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_GPU ) {
            vkFreeMemory(I_LogicalDevice,I_BufferDetails[Index].SecondaryBufferMemoryHndl,nullptr);
            vkDestroyBuffer(I_LogicalDevice,I_BufferDetails[Index].SecondaryBufferHndl,nullptr);
        }
        I_BufferDetails[Index].InUse = false;
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                                I s  B u f f e r  C r e a t e d
//
//  This routine can be used to find out if a buffer previously described in a call to
//  SetBufferDetails() has actually been created through a call to CreateBuffer() or not.
//
//  Parameters:
//     BufferHndl    (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Returns:
//     (bool)        True if the buffer has been created, false if not.
//
//  Pre-requisites:
//     SetBufferDetails() must have been called to describe how the buffer will be used and
//     to return the buffer handle that is passed to this routine.

bool KVVulkanFramework::IsBufferCreated (KVBufferHandle BufferHndl,bool& StatusOK)
{
    bool IsCreated = false;
    if (!AllOK(StatusOK)) return IsCreated;
    int Index = BufferIndexFromHandle(BufferHndl,StatusOK);
    if (AllOK(StatusOK)) {
        if (I_BufferDetails[Index].MainBufferHndl != VK_NULL_HANDLE) IsCreated = true;
    }
    return IsCreated;
}

//  ------------------------------------------------------------------------------------------------
//
//                                R e s i z e  B u f f e r
//
//  This routine resizes a buffer previously described in a call to SetBufferDetails(), and
//  created using a call to CreateBuffer(). The size can be increased or decreased.
//
//  Parameters:
//     BufferHandle  (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     SizeInBytes   (long) The new size of the buffer in bytes.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     The basic Vulkan initialisation must have been completed - CreateVulkanInstance(),
//     FindSuitableDevice() and CreateLogicalDevice() - as Vulkan needs to know what actual
//     GPU will be using this buffer. SetBufferDetails() must have been called to describe
//     how the buffer will be used and to return the buffer handle that is passed to this
//     routine and the buffer should already have been created using CreateBuffer().
//
//  Post-requisites:
//     If the buffer had been mapped using MapBuffer(), then changing the size will invalidate
//     the mapping and it will need to be re-mapped. This routine will unmap an already mapped
//     buffer, so there is no need to call UnmapBuffer().
//
//  Note:
//     As currently implemented, reducing the size of a buffer will not release the memory
//     it uses. It simply leaves some unused, and if a later call increases the size up to
//     the original size, it will simply be reused. Only if the call increases the size of
//     the buffer beyond the original allocation will the memory be released and reallocated.

void KVVulkanFramework::ResizeBuffer (KVBufferHandle BufferHndl,long NewSizeInBytes,bool& StatusOK)
{
    //  Note that a buffer should always be remapped after being resized.
    
    if (!AllOK(StatusOK)) return;
    int Index = BufferIndexFromHandle(BufferHndl,StatusOK);
    if (AllOK(StatusOK)) {
        //  If the memory actually allocated for the buffer is large enough, then we can
        //  simply pretend to resize it. Any call to MapBuffer() will simply see that the
        //  buffer is already mapped and return efficiently.
        if (I_BufferDetails[Index].MemorySizeInBytes >= NewSizeInBytes) {
            I_BufferDetails[Index].SizeInBytes = NewSizeInBytes;
        } else {
            //  The validation layer warns about calling vkDestroyBuffer() for a buffer
            //  still in use by a command buffer that hasn't completed execution. The call
            //  here gets around this, but seems a mite heavy-handed. A vkQueueWaitIdle()
            //  might be better - but this routine doesn't know what queue is being used.
            vkDeviceWaitIdle(I_LogicalDevice);
            //  To extend a buffer we have to delete the existing buffer and recreate it.
            //  This is what DeleteBuffer() does, but this code does not clear the InUse flag in
            //  the I_BufferDetails[Index] structure. We allow for the possibility that the buffer
            //  does not actually exist (in which case, the caller should really have used
            //  CreateBuffer(), but we'll let them get away with it).
            if (I_BufferDetails[Index].MappedAddress) {
                vkUnmapMemory(I_LogicalDevice,I_BufferDetails[Index].MainBufferMemoryHndl);
                I_BufferDetails[Index].MappedAddress = nullptr;
            }
            if (I_BufferDetails[Index].MainBufferMemoryHndl != VK_NULL_HANDLE) {
                vkFreeMemory(I_LogicalDevice,I_BufferDetails[Index].MainBufferMemoryHndl,nullptr);
                I_BufferDetails[Index].MainBufferMemoryHndl = VK_NULL_HANDLE;
            }
            if (I_BufferDetails[Index].MainBufferHndl != VK_NULL_HANDLE) {
                vkDestroyBuffer(I_LogicalDevice,I_BufferDetails[Index].MainBufferHndl,nullptr);
                I_BufferDetails[Index].MainBufferHndl = VK_NULL_HANDLE;
            }
            //  These only apply to staged buffers, but need checking.
            if (I_BufferDetails[Index].SecondaryBufferMemoryHndl != VK_NULL_HANDLE) {
                vkFreeMemory(I_LogicalDevice,I_BufferDetails[Index].SecondaryBufferMemoryHndl,
                                                                                         nullptr);
                I_BufferDetails[Index].SecondaryBufferMemoryHndl = VK_NULL_HANDLE;
            }
            if (I_BufferDetails[Index].SecondaryBufferHndl != VK_NULL_HANDLE) {
                vkDestroyBuffer(I_LogicalDevice,I_BufferDetails[Index].SecondaryBufferHndl,nullptr);
                I_BufferDetails[Index].SecondaryBufferHndl = VK_NULL_HANDLE;
            }

            //  Now create a new buffer and the associated memory. Note that you can't simply
            //  change the binding of an existing buffer.
            //  stackoverflow.com/questions/54761909/is-it-possible-to-change-a-vkbuffers-size
            
            VkBuffer BufferHndl;
            VkDeviceMemory BufferMemoryHndl;
            VkBufferUsageFlags UsageFlags = I_BufferDetails[Index].MainUsageFlags;
            VkMemoryPropertyFlags PropertyFlags = I_BufferDetails[Index].MainPropertyFlags;
            I_Debug.Log ("Buffers","Creating new buffer.");
            CreateVulkanBuffer(NewSizeInBytes,UsageFlags,PropertyFlags,&BufferHndl,
                                                                   &BufferMemoryHndl,StatusOK);
            if (AllOK(StatusOK)) {
                I_BufferDetails[Index].MainBufferHndl = BufferHndl;
                I_BufferDetails[Index].MainBufferMemoryHndl = BufferMemoryHndl;
                I_BufferDetails[Index].MemorySizeInBytes = NewSizeInBytes;
                I_BufferDetails[Index].SizeInBytes = NewSizeInBytes;
            
                //  If we have a staged buffer, also need to recreate the secondary buffer.
            
                if (I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_CPU ||
                    I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_GPU ) {
                    I_Debug.Log ("Buffers","Creating new secondary buffer.");
                    UsageFlags = I_BufferDetails[Index].SecondaryUsageFlags;
                    PropertyFlags = I_BufferDetails[Index].SecondaryPropertyFlags;
                    CreateVulkanBuffer(NewSizeInBytes,UsageFlags,PropertyFlags,&BufferHndl,
                                       &BufferMemoryHndl,StatusOK);
                    if (AllOK(StatusOK)) {
                        I_BufferDetails[Index].SecondaryBufferHndl = BufferHndl;
                        I_BufferDetails[Index].SecondaryBufferMemoryHndl = BufferMemoryHndl;
                    }
                }
            }
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                   C r e a t e  V u l k a n  B u f f e r   (Internal routine)
//
//  This internal routine creates a single Vulkan buffer and allocates the memory for it. It
//  needs to know the buffer usage type (storage, uniform, vertex) and its properties (mostly
//  to do with sharing, ie shared, local, etc). These are set by SetBufferDetails() when the
//  initial details of the buffer are supplied. It also needs to know the buffer size in bytes.
//  It returns the Vulkan handles for the buffer and for its associated memory. Note that
//  what Vulkan calls a 'buffer' - something it references using a VkBuffer handle - and what
//  the Framework calls a 'buffer' are rather different; the Framework treats a buffer and its
//  memory as a signle item, and even treats a staged buffer (implemented through two Vulkan
//  buffers, one local to the CPU and one local to the GPU) as a single item. This routine
//  needs to be called twice for a staged buffer.
//
//  Parameters:
//     SizeInBytes   (long) The size of the buffer in bytes.
//     UsageFlags    (VkBufferUsageFlags) Describes the usage of the buffer - uniform, storage, etc.
//     PropertyFlags (VkMemoryPropertyFlags) Describes the memory properties for the buffer,
//                   GPU local, shared, etc. that the buffer needs to have.
//     BufferHndlPtr (VkBuffer*) Receives the buffer handle used by Vulkan to access the buffer.
//     BufferMemoryHandlPtr (VkDeviceMemory*) Receives the memory handle used by Vulkan to access
//                   the memory for the buffer.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to create the logical device and set its
//     handle in I_LogicalDevice. (I_SelectedDevice also needs to have been set to the handle
//     of the selected physical device - this is needed to determine the supported memory
//     types - but this is a necessary precursor to CreateLogicalDevice() anyway.)

void KVVulkanFramework::CreateVulkanBuffer(
        VkDeviceSize SizeInBytes,VkBufferUsageFlags UsageFlags,VkMemoryPropertyFlags PropertyFlags,
                        VkBuffer* BufferHndlPtr,VkDeviceMemory* BufferMemoryHndlPtr,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  The VkBuffer type is an opaque handle to a data buffer. You create one with a call to
    //  vkCreateBuffer(), which returns a 'handle' that identifies the buffer and which you can
    //  use in calls to other Vulkan routines. You shouldn't make any assumptions about what
    //  this handle is (it's probably an address, but it could be an index into an internal
    //  table maintained by Vulkan). But you can use a value of VK_NULL_HANDLE to indicate that
    //  it hasn't been set yet. Creating a buffer with vkCreateBuffer() doesn't actually create
    //  the memory for the data in the buffer. You have to do that separately, by calling
    //  vkAllocateMemory(), which returns another handle value identifying the memory, and then
    //  bind that memory to the buffer. The BufferHndlPtr parameter is the address of a buffer
    //  handle which will be set by this routine, hence the name.
        
    *BufferHndlPtr = VK_NULL_HANDLE;
    *BufferMemoryHndlPtr = VK_NULL_HANDLE;
    
    //  First though, we do have to create the buffer.
    
    //  It's assumed that the buffer's sharing mode will be VK_SHARING_MODE_EXCLUSIVE (the buffer
    //  will only be used by one queue family. The only alternative, VK_SHARING_MODE_CONCURRENT,
    //  is more complex to set up and needs a list of queue families to be provided in the
    //  BufferInfo structure (and so in the call to this routine).
    
    VkBufferCreateInfo BufferInfo{};
    BufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    BufferInfo.size = SizeInBytes;
    BufferInfo.usage = UsageFlags;
    BufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    I_Debug.Log ("Buffers","Creating Vulkan buffer.");
    VkResult Result;
    Result = vkCreateBuffer(I_LogicalDevice,&BufferInfo,nullptr,BufferHndlPtr);
    if (Result == VK_SUCCESS) {
                
        //  Before we allocate the memory needed for the buffer, we have to determine what type of
        //  memory we need. Vulkan devices provide different types of memory, and different buffer
        //  types have different memory requirements. Now that we have set up our VkBuffer, and
        //  have its address in BufferHndlPtr, we can find out what it actually requires.
        
        VkMemoryRequirements MemoryRequirements;
        vkGetBufferMemoryRequirements(I_LogicalDevice,*BufferHndlPtr,&MemoryRequirements);
        
        //  Then we can find out which of the various memory types will do for our buffer.
        
        uint32_t MemoryTypeIndex = GetMemoryTypeIndex(MemoryRequirements,PropertyFlags,StatusOK);
        if (AllOK(StatusOK)) {
            
            //  Finally, we can allocate some of the required memory, using vkAllocateMemory(),
            //  which will set up a VkDeviceMemory - this is another opaque handle, whose
            //  address we've set in BufferMemoryHndlPtr - so that we can refer to it.
            
            VkMemoryAllocateInfo AllocateInfo{};
            AllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            AllocateInfo.allocationSize = MemoryRequirements.size;
            AllocateInfo.memoryTypeIndex = MemoryTypeIndex;
            I_Debug.Logf ("Buffers","Memory type index = %d",AllocateInfo.memoryTypeIndex);

            Result = vkAllocateMemory(I_LogicalDevice,&AllocateInfo,nullptr,BufferMemoryHndlPtr);
            if (Result == VK_SUCCESS) {
                            
                //  And having done that, we can bind the memory to our buffer.
                
                vkBindBufferMemory(I_LogicalDevice,*BufferHndlPtr,*BufferMemoryHndlPtr,0);
            } else {
                LogVulkanError ("Failed to allocate buffer memory","vkAllocateMemory",Result);
                StatusOK = false;
            }
        }
        
    } else {
        LogVulkanError ("Failed to create buffer","vkCreateBuffer",Result);
        StatusOK = false;
    }
    
    //  If things went wrong, release anything that was allocated before the problem was spotted.
    
    if (!AllOK(StatusOK)) {
        if (*BufferMemoryHndlPtr != VK_NULL_HANDLE) {
            vkFreeMemory(I_LogicalDevice,*BufferMemoryHndlPtr,nullptr);
        }
        if (*BufferHndlPtr != VK_NULL_HANDLE) {
            vkDestroyBuffer(I_LogicalDevice,*BufferHndlPtr,nullptr);
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//               C r e a t e  V u l k a n  D e s c r i p t o r  S e t  L a y o u t
//
//  A computation is run as part of a pipeline, and a pipeline has to be set up in detail before
//  it can be run. It will access a set of one or more buffers, and it needs to know how these
//  are defined. To do this, it needs a 'descriptor set' which describes the bindings and the types
//  of the buffers. But before you can create a descriptor set, you have to specify its layout,
//  which reflects the details of the various buffers. This routine performs that initial step of
//  creating the necessary descriptor set. It is passed a vector containing the Framework buffer
//  handles for all the buffers involved, creates the descriptor set and returns a Vulkan handle
//  to the set.

//  Parameters:
//     BufferHandles (std::vector<KVVulkanFramework::KVBufferHandle>&) a vector containing the
//                   Framework handles for each buffer, as returned by SetBufferDetails().
//     SetLayoutHndlPtr (VkDescriptorSetLayout*) Receives the opaque Vulkan handle for the
//                   created descriptor set layout.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//     The details of each buffer in question must have been set using SetBufferDetails() and
//     the vector (BufferHandles) containing the handles for each buffer must have been created
//     and filled.
//
//  Note:
//     The list of buffers passed in BufferHandles is used to specify the number and type of each
//     buffer, so the buffer layout cannot be changed without creating a new descriptor set layout
//     (and new versions of everything based on that layout), but the sizes of the individual
//     buffers are only finally fixed when the set is fully sepecified using the routine
//     SetupVulkanDescriptorSet().

void KVVulkanFramework::CreateVulkanDescriptorSetLayout(
                    std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                                            VkDescriptorSetLayout* SetLayoutHndlPtr,bool& StatusOK)
{
    //  Pre-requisites:
    //     CreateLogicalDevice() must have been called to create the logical device and set its
    //     handle in I_LogicalDevice.
    
    if (!AllOK(StatusOK)) return;
    
    //  This sets up a descriptor set layout with the entries set up to match the specifications
    //  for the buffers as listed in BufferHandles. If any handles are invalid, we don't quit
    //  immediately - instead we continue on to check the rest of the handles. Might as well.
    
    std::vector<VkDescriptorSetLayoutBinding> LayoutBindings{};
    for (KVBufferHandle Handle : BufferHandles) {
        int Index = BufferIndexFromHandle(Handle,StatusOK);
        if (!AllOK(StatusOK)) continue;
        VkDescriptorSetLayoutBinding Binding;
        Binding.binding = I_BufferDetails[Index].Binding;
        Binding.descriptorCount = 1;
        if (I_BufferDetails[Index].BufferType == TYPE_UNIFORM) {
            I_Debug.Logf ("Buffers","Setting for uniform buffer, binding %d",Binding.binding);
            Binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            Binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        if (I_BufferDetails[Index].BufferType == TYPE_STORAGE) {
            I_Debug.Logf ("Buffers","Setting for storage buffer, binding %d",Binding.binding);
            Binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            Binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        Binding.pImmutableSamplers = nullptr;
        LayoutBindings.push_back(Binding);
    }

    if (AllOK(StatusOK)) {
        VkDescriptorSetLayoutCreateInfo LayoutInfo{};
        LayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        LayoutInfo.bindingCount = int(BufferHandles.size());
        LayoutInfo.pBindings = LayoutBindings.data();

        VkResult Result;
        Result = vkCreateDescriptorSetLayout(I_LogicalDevice,&LayoutInfo,nullptr,SetLayoutHndlPtr);
        if (Result != VK_SUCCESS) {
            LogVulkanError("Failed to create compute descriptor set layout",
                                                    "vkCreateDescriptorSetLayout",Result);
            StatusOK = false;
        } else {
            I_Debug.Logf("Progress","Compute descriptor set created with %d buffer bindings",
                                                                         BufferHandles.size());
        }
    }
        
    //  If the layout was created successfully, save the value of its handle so it can be
    //  destroyed during cleanup.
        
    if (AllOK(StatusOK)) I_DescriptorSetLayoutHndls.push_back(*SetLayoutHndlPtr);
}

//  ------------------------------------------------------------------------------------------------
//
//                   B u f f e r  I n d e x  F r o m  H a n d l e  (Internal routine)
//
//  This internal routine returns the index into the I_BufferDetails vector that corresponds to
//  the Framework handle for the buffer as returned by SetBufferDetails(). It also checks that
//  the handle is valid and returns bad status if it isn't.
//
//  Parameters:
//     Handle        (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     The buffer must have had its details set using SetBufferDetails(), which will have
//     returned the buffer handle.
//
//  Note:
//     This routine makes use of the fact that a KVBufferHandle is actually an int. It's an
//     index into the I_BufferDetails vector, but with 1 added to it so that 0 can be used
//     as an invalid index number. It isn't a foolproof scheme

int KVVulkanFramework::BufferIndexFromHandle (KVBufferHandle Handle,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return 0;
    
    int Index = Handle - 1;
    if (Index < 0 || Index >= int(I_BufferDetails.size())) {
        LogError ("Buffer handle value %d is out of range",Handle);
        Index = 0;
        StatusOK = false;
    } else {
        if (!I_BufferDetails[Index].InUse) {
            LogError ("Buffer handle value %d is no longer in use",Handle);
            StatusOK = false;
        }
    }
    return Index;
}

//  ------------------------------------------------------------------------------------------------
//
//                       C r e a t e  C o m p u t e  P i p e l i n e
//
//  This routine creates a relatively simple Vulkan pipeline that will run a specified GPU
//  program (written in GLSL), operating on a specified layout of Vulkan data buffers. At this
//  point, the sizes of the various buffers do not need to be known, but their type (uniform,
//  storage, etc) and the way they are accessed (local, shared, etc) must be known. The details
//  of the buffer layout is encapsulated in a descriptor set layout as created by the routine
//  CreateVulkanDescriptorSetLayout(). The GPU program is specified by giving this routine the
//  name of a file containing its SPIR-V semi-compiled code. This routine creates both a layout
//  for the pipeline and then the pipeline itself, and returns Vulkan handles for both.
//
//  Parameters:
//     ShaderFilename (const std::string&) The name of file containing the SPIR-V code for the
//                   GPU program to be run by the pipeline.
//     StageName     (const std::string&) The entry name of the routine in the CPU code to be
//                   run by the pipeline (often this will be 'main', but it does not have to be).
//     SetLayoutHndlPtr (VkDescriptorSetLayout*) The address of the opaque Vulkan handle for the
//                   descriptor set layout, as returned by CreateVulkanDescriptorSetLayout().
//                   Do note that this is passed as the address of the handle, not the handle
//                   itself - although this routine does not change the handle.
//     PipelineLayoutHndlPtr (VkPipelineLayout*) Receives the opaque Vulkan handle for the
//                   created pipeline layout.
//     PipelineHndlPtr (VkPipeline*) Receives the opaque Vulkan handle for the created pipeline.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//     The descriptor set layout for the buffers must have been created by a call to the routine
//     CreateVulkanDescriptorSetLayout(). The file name specified must be a string that can be
//     passed to fopen(), and the file should contain valid code.
//
//  Note:
//     o Usually in Vulkan the SPIR_V GPU code in the specified file will have been written in GLSL
//     and converted (semi-compiled) into SPIR-V using glslc. However, the SPIR-V code could have
//     been created in other ways - it could have been hand-written, although takes a braver soul
//     than I, or compiled from another language. All that matters here is that the file contains
//     valid SPIR-V code.
//     o Some valid SPIR-V code may be rejected if the GPU in use does not support it. For example,
//     an Apple GPU cannot run code that uses double precision GPU values, and this will be
//     spotted when this routine tries to process the code.
//     o The SetLayoutHndlPtr parameter is a pointer to the handle, as the name suggests, rather
//     than the handle itself. This is only because the pipeline can be created with a number of
//     different set layouts, passed to Vulkan as an array, and doing it this way makes it look
//     like an array with 1 element.

void KVVulkanFramework::CreateComputePipeline(
    const std::string& ShaderFilename,const std::string& StageName,
    VkDescriptorSetLayout* SetLayoutHndlPtr,VkPipelineLayout* PipelineLayoutHndlPtr,
                                        VkPipeline* PipelineHndlPtr,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    *PipelineLayoutHndlPtr = nullptr;
    *PipelineHndlPtr = nullptr;

    //  Read the SPIR_V code from the file, and create the required shader module.
    
    long LengthInBytes;
    uint32_t* ShaderCode = nullptr;
    ShaderCode = ReadSpirVFile(ShaderFilename,&LengthInBytes,StatusOK);

    VkShaderModule ShaderModule = CreateShaderModule(ShaderCode,LengthInBytes,StatusOK);
    
    if (AllOK(StatusOK)) {

        //  Set up the pipeline layout
        
        VkPipelineShaderStageCreateInfo ShaderStageInfo{};
        ShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ShaderStageInfo.module = ShaderModule;
        ShaderStageInfo.pName = StageName.c_str();

        VkPipelineLayoutCreateInfo PipelineLayoutInfo{};
        PipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        PipelineLayoutInfo.setLayoutCount = 1;
        PipelineLayoutInfo.pSetLayouts = SetLayoutHndlPtr;

        VkResult Result;
        Result = vkCreatePipelineLayout(I_LogicalDevice,&PipelineLayoutInfo,nullptr,
                                                                  PipelineLayoutHndlPtr);
        if (Result != VK_SUCCESS) {
            LogVulkanError ("Failed to create compute pipeline layout","vkCreatePipelineLayout",
                                                                                         Result);
            StatusOK = false;
        } else {
        
            //  And now create the actual pipeline.
            
            VkComputePipelineCreateInfo PipelineInfo{};
            PipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            PipelineInfo.layout = *PipelineLayoutHndlPtr;
            PipelineInfo.stage = ShaderStageInfo;

            Result = vkCreateComputePipelines(I_LogicalDevice,VK_NULL_HANDLE,1,
                                              &PipelineInfo,nullptr,PipelineHndlPtr);
            if (Result != VK_SUCCESS) {
                LogVulkanError ("Failed to create compute pipeline","vkCreateComputePipelines",
                                                                                       Result);
                StatusOK = false;
            }
        }
    }

    //  The ShaderCode buffer and the shader module are no longer needed, even if all went well.
    //  Release them if they have been created.
    
    if (ShaderCode) delete[] ShaderCode;

    if (ShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(I_LogicalDevice,ShaderModule,nullptr);
    }
    
    //  If the pipeline and its layout were created and bound successfully, record the details
    //  so they can be shut down properly at the end of the program. If things went wrong,
    //  release these if they were allocated before the problem was spotted.
    
    if (AllOK(StatusOK)) {
        T_PipelineDetails PipelineDetails;
        PipelineDetails.PipelineHndl = *PipelineHndlPtr;
        PipelineDetails.PipelineLayoutHndl = *PipelineLayoutHndlPtr;
        I_PipelineDetails.push_back(PipelineDetails);
        I_Debug.Log("Progress","Compute pipeline created.");
    } else {
        if (*PipelineLayoutHndlPtr != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(I_LogicalDevice,*PipelineLayoutHndlPtr,nullptr);
        }
        if (*PipelineHndlPtr != VK_NULL_HANDLE) {
            vkDestroyPipeline(I_LogicalDevice,*PipelineHndlPtr,nullptr);
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                   C r e a t e  V u l k a n  D e s c r i p t o r  P o o l
//
//  A Vulkan descriptor set needs to be created for the set of buffers to be used for a calculation.
//  Vulkan uses a descriptor pool to provide descriptor sets as required, and this routine creates
//  such a pool. Each pool provides descriptor sets with a particular layout, and this is provided
//  to this routine by passing it a vector containing all the Framework handles for all the buffers
//  in question, as returned by SetBufferDetails().
//
//  Parameters:
//     BufferHandles (std::vector<KVVulkanFramework::KVBufferHandle>&) a vector containing the
//                   Framework handles for each buffer, as returned by SetBufferDetails().
//     MaxSets       (int) The maximum number of descriptor sets that the pool is to provide.
//     PoolHndlPtr   (VkDescriptorPool*) Receives the opaque Vulkan handle for the created pool.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//     The details of each buffer in question must have been set using SetBufferDetails() and
//     the vector (BufferHandles) containing the handles for each buffer must have been created
//     and filled.

void KVVulkanFramework::CreateVulkanDescriptorPool(
            std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                        int MaxSets,VkDescriptorPool* PoolHndlPtr,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
 
    //  At the moment, this only supports uniform buffers and storage buffers. See how many we
    //  have of each.

    int UniformBuffers = 0;
    int StorageBuffers = 0;
    for (KVBufferHandle Handle : BufferHandles) {
        int Index = BufferIndexFromHandle(Handle,StatusOK);
        if (!AllOK(StatusOK)) break;
        if (I_BufferDetails[Index].BufferType == TYPE_UNIFORM) UniformBuffers++;
        if (I_BufferDetails[Index].BufferType == TYPE_STORAGE) StorageBuffers++;
    }

    if (AllOK(StatusOK)) {
        
        //  Since we support two buffer types (storage and uniform) we set up their details
        //  in an array that we will pass to vkCreateDescriptorPool(). If only one type is
        //  actually being used, NumberPoolTypes will be 1 and the second element of PoolSizes
        //  won't end up being used. All that PoolSizes is used for is to tell Vulkan how
        //  many buffers of each type will be used for each set and it can allocate the descriptor
        //  pool on that basis - together with the number of sets that will be used (MaxSets).
        
        std::array<VkDescriptorPoolSize,2> PoolSizes{};
        int NumberPoolTypes = 0;
        if (UniformBuffers > 0) NumberPoolTypes++;
        if (StorageBuffers > 0) NumberPoolTypes++;

        int Index = 0;
        if (UniformBuffers > 0) {
            PoolSizes[Index].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            PoolSizes[Index].descriptorCount = UniformBuffers;
            Index++;
        }
        if (StorageBuffers > 0) {
            PoolSizes[Index].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            PoolSizes[Index].descriptorCount = StorageBuffers;
            Index++;
        }

        VkDescriptorPoolCreateInfo PoolInfo{};
        PoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        PoolInfo.poolSizeCount = NumberPoolTypes;
        PoolInfo.pPoolSizes = PoolSizes.data();
        PoolInfo.maxSets = MaxSets;

        VkResult Result;
        Result = vkCreateDescriptorPool(I_LogicalDevice,&PoolInfo,nullptr,PoolHndlPtr);
        if (Result != VK_SUCCESS) {
            LogVulkanError ("Failed to create descriptor pool","vkCreateDescriptorPool",Result);
            StatusOK = false;
        } else {
            I_DescriptorPoolHndls.push_back(*PoolHndlPtr);
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                   A l l o c a t e  V u l k a n  D e s c r i p t o r  S e t
//
//  A Vulkan descriptor set needs to be created for the set of buffers to be used for a calculation,
//  and this routine allocates such a set from an already created Vulkan descriptor pool.
//
//  Parameters:
//     SetLayoutHndl (VkDescriptorSetLayout) The Vulkan handle for the created descriptor set
//                   layout, as returned by CreateVulkanDescriptorSetLayout().
//     PoolHndl      (VkDescriptorPool*) The Vulkan handle for the pool, as returned by
//                   CreateVulkanDescriptorPool().
//     SetHndlPtr    (VkDescriptorSet*) Receives the opaque Vulkan handle for the allocated set.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//     The descriptor set layout for the buffers must have been created by a call to the routine
//     CreateVulkanDescriptorSetLayout(). The pool must have been created by a call to the routine
//     CreateVulkanDescriptorPool().

void KVVulkanFramework::AllocateVulkanDescriptorSet(
    VkDescriptorSetLayout SetLayoutHandl,
                    VkDescriptorPool PoolHndl,VkDescriptorSet* SetHndlPtr, bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  First, allocate a single descriptor set from the pool. (Worth mentioning at this point
    //  that this set doesn't need explicitly destroying - it's destroyed when the pool is
    //  destroyed.)
    
    VkDescriptorSetLayout LocalSetLayoutHandl = SetLayoutHandl;
    VkDescriptorSetAllocateInfo AllocInfo{};
    AllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    AllocInfo.descriptorPool = PoolHndl;
    AllocInfo.descriptorSetCount = 1;
    AllocInfo.pSetLayouts = &LocalSetLayoutHandl;

    VkResult Result;
    Result = vkAllocateDescriptorSets(I_LogicalDevice,&AllocInfo,SetHndlPtr);
    if (Result != VK_SUCCESS) {
        LogVulkanError ("Failed to allocate descriptor sets","vkAllocateDescriptorSets",Result);
        StatusOK = false;
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                   S e t u p  V u l k a n  D e s c r i p t o r  S e t
//
//  A Vulkan descriptor set describes the various buffers used for a calculation. Such a set (and
//  the pool from which it has to be allocated) can be created given only the initial details of
//  the number of buffers involved, their types and access details, before the buffers themselves
//  are actually created. Once the buffers have been created (by calls to CreateBuffer()), the
//  setup of the descriptor set can be completed. In particular, up until now, the sizes of the
//  buffers have not been included in the descriptor set.
//
//  Parameters:
//     BufferHandles (std::vector<KVVulkanFramework::KVBufferHandle>&) a vector containing the
//                   Framework handles for each buffer, as returned by SetBufferDetails().
//     SetHndl       (VkDescriptorSet) The Vulkan handle for the descriptor set, as returned
//                   by CreateVulkanDescriptorSet().
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//     The descriptor set for the buffers must have been created by a call to the routine
//     CreateVulkanDescriptorSet(). The details of each buffer in question must have been set
//     using SetBufferDetails(), the vector (BufferHandles) containing the handles for each buffer
//     must have been created and filled, and all the buffers themselves must now have been
//     created by calls to CreateBuffer().

void KVVulkanFramework::SetupVulkanDescriptorSet(
        std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                                        VkDescriptorSet SetHndl, bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  We need to know how many active buffers we have to handle, in order to allocate
    //  a large enough array to pass to vkUpdateDescriptorSets().
    
    int BufferCount = 0;
    for (KVBufferHandle Handle : BufferHandles) {
        int Index = BufferIndexFromHandle(Handle,StatusOK);
        if (!AllOK(StatusOK)) continue;
        if (I_BufferDetails[Index].InUse) {
            if (I_BufferDetails[Index].BufferType == TYPE_UNIFORM) BufferCount++;
            if (I_BufferDetails[Index].BufferType == TYPE_STORAGE) BufferCount++;
        }
    }

    if (AllOK(StatusOK)) {
        
        //  What we pass to vkUpdateDescriptorSets() is an array of VkWriteDescriptorSets,
        //  which contain things like the binding (which binds the actual buffer we created
        //  to the buffer referred to by the GPU code) and the type, and also a pointer to
        //  a VkDescriptorBufferInfo structure that contain things like the VkBuffer handle
        //  for the buffer and its size. So we also need an arrray of these as well.
                
        std::vector<VkWriteDescriptorSet> WriteDescriptors{};
        WriteDescriptors.resize(BufferCount);
        std::vector<VkDescriptorBufferInfo> BufferInfo{};
        BufferInfo.resize(BufferCount);

        int WriteIndex = 0;
        for (KVBufferHandle Handle : BufferHandles) {
            
            //  For each buffer we deal with, first check this is a buffer we know how to handle.
            //  Just in case. If it's not, we skip to the next buffer.
            
            int Index = BufferIndexFromHandle(Handle,StatusOK);

            if (!AllOK(StatusOK)) break;
            VkDescriptorType Type;
            if (I_BufferDetails[Index].BufferType == TYPE_UNIFORM) {
                Type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            } else if (I_BufferDetails[Index].BufferType == TYPE_STORAGE) {
                Type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            } else {
                continue;
            }
            
            //  Set up the Buffer information structure first
            
            if (I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_CPU ||
                I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_GPU ) {
                BufferInfo[WriteIndex].buffer = I_BufferDetails[Index].SecondaryBufferHndl;
            } else {
                BufferInfo[WriteIndex].buffer = I_BufferDetails[Index].MainBufferHndl;
            }
            BufferInfo[WriteIndex].offset = 0;
            BufferInfo[WriteIndex].range = I_BufferDetails[Index].SizeInBytes;

            //  Then set up the write descriptor set, including a pointer to the Buffer
            //  information set. (It's OK that the buffer information set will go out of
            //  scope at the end of this routine - vkUpdateDescriptorSets() will copy
            //  what it needs from it, and doesn't need it to hang around.)
            
            WriteDescriptors[WriteIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            WriteDescriptors[WriteIndex].dstSet = SetHndl;
            WriteDescriptors[WriteIndex].dstBinding = I_BufferDetails[Index].Binding;
            WriteDescriptors[WriteIndex].dstArrayElement = 0;
            WriteDescriptors[WriteIndex].descriptorType = Type;
            WriteDescriptors[WriteIndex].descriptorCount = 1;
            WriteDescriptors[WriteIndex].pBufferInfo = &BufferInfo[WriteIndex];
            WriteIndex++;
        }

        //  And now we do the work we came here to do.
        
        vkUpdateDescriptorSets(I_LogicalDevice,BufferCount,WriteDescriptors.data(),0,nullptr);
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                           C r e a t e  C o m m a n d  P o o l
//
//  A Vulkan command pool is needed to supply the command buffers that can be used to run pipelines
//  on the GPU. This routine creates such a pool.
//
//  Parameters:
//     CommandPoolHndlPtr (VkCommandPool*) Receives the Vulkan handle for the command pool.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//      CreateLogicalDevice() must have been called to create the Vulkan logical device.

void KVVulkanFramework::CreateCommandPool(VkCommandPool* CommandPoolHndlPtr,bool& StatusOK)
{
    //   CreateLogicalDevice() will have created the logical device and set its handle in
    //   I_LogicalDevice, and will have identified the command queue family to be used
    //   and set its index in I_QueueFamilyIndex.
    
    if (!AllOK(StatusOK)) return;
    
    VkCommandPoolCreateInfo PoolInfo{};
    PoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    PoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    PoolInfo.queueFamilyIndex = I_QueueFamilyIndex;
    
    //  Note that you don't have to specify the number of command buffers in the pool. The pool
    //  doesn't contain the buffers it will be asked to allocate, it just coordinates them.
    
    VkResult Result;
    Result = vkCreateCommandPool(I_LogicalDevice,&PoolInfo,nullptr,CommandPoolHndlPtr);
    if (Result != VK_SUCCESS) {
        LogVulkanError ("Failed to create command pool","vkCreateCommandPool",Result);
        StatusOK = false;
    } else {
        I_Debug.Log ("Progress","Created new command pool.");
        I_CommandPoolHndls.push_back(*CommandPoolHndlPtr);
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                     C r e a t e  C o m p u t e  C o m m a n d  B u f f e r
//
//  This routine creates a single Vulkan command buffer, which may be enough for a simple compute
//  program that may only need a single such buffer.
//
//  Parameters:
//     CommandPoolHndl (VkCommandPool) The Vulkan handle for the command pool, as returned by
//                     CreateCommandPool().
//     CommandBufferHndlPtr  (VkCommandBuffer*) Receives the Vulkan handle for the command buffer.
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//
//  Pre-requisites:
//      CreateLogicalDevice() must have been called to create the Vulkan logical device,
//      and CreateCommandPool() must have been called to create the command pool.
//
//  Note:
//     This was originally provided by an early version for the Framework and is retained for
//     backwards compatability and perhaps some convenience. Command buffers are the same for
//     both compute and graphics pipelines, and this is now just a simple wrapper around
//     CreateCommandBuffers()

void KVVulkanFramework::CreateComputeCommandBuffer(
        VkCommandPool CommandPoolHndl, VkCommandBuffer* CommandBufferHndlPtr,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    std::vector<VkCommandBuffer> CommandBuffers;
    CreateCommandBuffers (CommandPoolHndl,1,CommandBuffers,StatusOK);
    *CommandBufferHndlPtr = CommandBuffers[0];
}

//  ------------------------------------------------------------------------------------------------
//
//                         C r e a t e  C o m m a n d  B u f f e r s
//
//  This routine creates a specified number of the Vulkan command buffers that used to run
//  pipelines. A simple compute program may only need a single such buffer, but a graphics
//  program running a swap chain may need one buffer for each image being used.
//
//  Parameters:
//     CommandPoolHndl (VkCommandPool) The Vulkan handle for the command pool, as returned by
//                     CreateCommandPool().
//     NumberBuffers   (int) The number of command buffers to be created.
//     CommandBuffers  (std::vector<VkCommandBuffer>&) Returns the set of command buffers
//                     requested.
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//
//  Pre-requisites:
//      CreateLogicalDevice() must have been called to create the Vulkan logical device,
//      and CreateCommandPool() must have been called to create the command pool.

void KVVulkanFramework::CreateCommandBuffers(
        VkCommandPool CommandPoolHndl, int NumberBuffers,
                            std::vector<VkCommandBuffer>& CommandBuffers,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
        
    VkCommandBufferAllocateInfo AllocInfo{};
    AllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    AllocInfo.commandPool = CommandPoolHndl;
    AllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    AllocInfo.commandBufferCount = NumberBuffers;
    
    CommandBuffers.resize(NumberBuffers);

    VkResult Result;
    Result = vkAllocateCommandBuffers(I_LogicalDevice,&AllocInfo,CommandBuffers.data());
    if (Result != VK_SUCCESS) {
        LogVulkanError("Failed to allocate command buffers","vkAllocateCommandBuffers",Result);
        StatusOK = false;
    }
    
    if (AllOK(StatusOK)) I_Debug.Logf ("Progress","Allocated %d command buffers.",NumberBuffers);
}

//  ------------------------------------------------------------------------------------------------
//
//                         R e c o r d  C o m p u t e  C o m m a n d  B u f f e r
//
//  This routine sets up a command buffer with the details of the operation it is to perform.
//  This includes binding the buffer to the pipeline to be used, and to the descriptor sets
//  that describe how it is to use the various data buffers. It also need to be told how to
//  distribute the calculations across the various workgroups that are to be used - this depends
//  on both the dimensions of the data being processed and the details of the shader code to
//  be run by the pipeline.
//
//  This is the final thing to be done before the buffer can be submitted for execution using
//  RunCommandBuffer(), and everything must have been set up properly at this point - the
//  pipeline must be set up, and the descriptor set must describe the buffers, which must all
//  have been created.
//
//  Parameters:
//     CommandBufferHndl (VkCommandBuffer) The Vulkan handle for the command buffer, as returned by
//                     either CreateCommandBuffers() or by CreateComputeCommandBuffer().
//     PipelineHndl    (VkPipeline) The Vulkan handle for the pipeline to be run, as returned by
//                     CreateComputePipeline().
//     DescriptorSetHndlPtr (VkDescriptorSet*) The address of the descriptor set that contains
//                     details of the buffers to be used by the calculation. This must have been
//                     set by AllocateVulkanDescriptorSet(). Note that this is passed by
//                     address, not by value.
//     WorkGroupCounts (uint32_t[3]) The 3-D layout of the workgroups to be used. This is an
//                     array set up on the basis of the dimensions of the data, but also has
//                     to take into account the coding of the shader program that will process
//                     the data.
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//
//  Pre-requisites:
//      The command buffer must have been created by either CreateCommandBuffers() or by
//      CreateComputeCommandBuffer(). The pipeline must have been created by CreateComputePipeline()
//      and the descriptor set must have been allocated by AllocateVulkanDescriptorSet and set up
//      by SetupVulkanDescriptorSet().  all the various pre-requisites for those routines must have
//      been fulfilled as well.

void KVVulkanFramework::RecordComputeCommandBuffer(
    VkCommandBuffer CommandBufferHndl,VkPipeline PipelineHndl,
    VkPipelineLayout PipelineLayoutHndl,VkDescriptorSet* DescriptorSetHndlPtr,
    uint32_t WorkGroupCounts[3],bool& StatusOK)
{
    //  Note that the calculation of the values in WorkGroupCounts[] has to take into account
    //  the 3D dimensions of the data to be processed by the GPU shader (which will depend on
    //  the nature of the data itself, which will be known to the calling program) and on the
    //  3D dimensions of the local workgroup (which are set explicitly by the shader code, and
    //  appear not to be accessible by the CPU code).
    
    if (!AllOK(StatusOK)) return;
    
    //  'Record' here means setting up the command buffer with the details of the operation
    //  to be performed. In particular the command buffer has to be bound to both the pipeline
    //  to be used and the descriptor sets that describe how it is to use the various data buffers.
    
    VkCommandBufferBeginInfo BeginInfo{};
    BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkResult Result;
    Result = vkBeginCommandBuffer(CommandBufferHndl,&BeginInfo);
    if (Result != VK_SUCCESS) {
        LogVulkanError ("Failed to begin recording compute command buffer","vkBeginCommandBuffer",
                                                                                           Result);
        StatusOK = false;
    } else {

        //  Note that the next three calls don't return a status, so we can't test to see
        //  if anything went wrong. The best we can do is use AllOK() to see if the validation
        //  layers reported an error.
        
        vkCmdBindPipeline(CommandBufferHndl,VK_PIPELINE_BIND_POINT_COMPUTE,PipelineHndl);
        
        vkCmdBindDescriptorSets(CommandBufferHndl,VK_PIPELINE_BIND_POINT_COMPUTE,
                                          PipelineLayoutHndl,0,1,DescriptorSetHndlPtr,0,nullptr);
        
        //  This adds the 'dispatch' stage to the compute pipeline, and will start it
        //  running the compute shader when the command buffer is finally submitted for
        //  execution. It needs to be told the number of the work groups (in 3D) that
        //  will be run. The total number of threads run will be the number of work groups
        //  specified here multiplied by the local size of the work groups as set up in
        //  the shader code. (Note that this is different to the scheme used by Metal,
        //  where both sizes are set - in a slightly different way - in the CPU code. As
        //  far as I can see, there is no way in Vulkan for the CPU code to reliably find
        //  out the local sizes set in the shader code - they may be the same as the
        //  subgroup size but that can't be relied on.)
                
        vkCmdDispatch(CommandBufferHndl,WorkGroupCounts[0],WorkGroupCounts[1],
                                       WorkGroupCounts[2]);
        
        //  Now finish off the command buffer.
        
        Result = vkEndCommandBuffer(CommandBufferHndl);
        if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
            LogVulkanError ("Failed to complete Vulkan command buffer","vkEndCommandBuffer",Result);
            StatusOK = false;
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                             G e t  D e v i c e  Q u e u e
//
//  This returns a queue that can be used to submit a command buffer to the GPU for processing.
//
//  Parameters:
//     QueueHndlPtr    (VkQueue*) Receives the Vulkan handle for the queue.
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.

void KVVulkanFramework::GetDeviceQueue(VkQueue* QueueHndlPtr,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  I_LogicalDevice and I_QueueFamilyIndex will have been set by CreateLogicalDevice().

    vkGetDeviceQueue(I_LogicalDevice,I_QueueFamilyIndex,0,QueueHndlPtr);
    
    //  vkGetDeviceQueue() returns void, with no error indications. If validation layers are
    //  on they may catch an error, but otherwise the best bet seems to be test the returned
    //  queue handle.
    
    if (QueueHndlPtr == nullptr || *QueueHndlPtr == VK_NULL_HANDLE) {
        LogError("Failed to get device queue. vkGetDeviceQueue returns null handle.");
        StatusOK = false;
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                             R u n  C o m m a n d  B u f f e r
//
//  Submits a command buffer to a queue on the GPU for processing and waits until it has completed.
//  This is what all the long series of calls to set up a GPU computation have been leading to.
//
//  Parameters:
//     QueueHndl       (VkQueue) The Vulkan handle for the queue, as returned by GetDeviceQueue().
//     CommandBufferHndl (VkCommandBuffer) The Vulkan handle for the command buffer, as returned by
//                     either CreateCommandBuffers() or by CreateComputeCommandBuffer().
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//
//  Pre-requisites:
//     The CommandBuffer must have been set up by RecordCommandBuffer(), the queue must have been
//     obtained from GetDeviceQueue(), and all their pre-requisites must have been completed.
//
//  Note:
//     The command buffer must be re-recorded before it can be re-used, but so long as the layout
//     of the buffers remains unchanged, the data in them can be modified and the command buffer
//     re-recorded and re-run without other changes being needed.

void KVVulkanFramework::RunCommandBuffer(
    VkQueue QueueHndl,VkCommandBuffer CommandBufferHndl,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    VkCommandBuffer LocalCommandBufferHndl = CommandBufferHndl;
    VkSubmitInfo SubmitInfo = {};
    SubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    SubmitInfo.commandBufferCount = 1;
    SubmitInfo.pCommandBuffers = &LocalCommandBufferHndl;

    //  We need to set up a fence so we can wait on it for the computation to complete.
    
    VkFence Fence;
    VkFenceCreateInfo FenceCreateInfo = {};
    FenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    FenceCreateInfo.flags = 0;
    VkResult Result;
    Result = vkCreateFence(I_LogicalDevice,&FenceCreateInfo,nullptr,&Fence);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError("Failed to set up fence","vkCreateFence",Result);
        StatusOK = false;
    } else {

        //  Submit the command buffer to the queue, together with the fence.
        
        Result = vkQueueSubmit(QueueHndl,1,&SubmitInfo,Fence);
        if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
                LogVulkanError("Failed to submit compute queue","vkQueueSubmit",Result);
            StatusOK = false;
        } else {

            //  And wait for it to complete.
            
            Result = vkWaitForFences(I_LogicalDevice,1,&Fence,VK_TRUE,100000000000);
            if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
                LogVulkanError ("Failed to wait for compute to complete","vkWaitForFences",Result);
                StatusOK = false;
            }
        }
        
        //  And we don't need the fence any more.
        
        vkDestroyFence(I_LogicalDevice,Fence,nullptr);
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                       R e a d  S p i r V  F i l e  (Internal routine)
//
//  This internal routine reads SPIR-V code from a specified file into memory, and returns the
//  address of the start of that memory and the length of the code in bytes.
//
//  Parameters:
//     ShaderFilename (const std::string&) The name of file containing the SPIR-V code for the
//                    GPU program to be run by the pipeline.
//     LengthInBytes  (long*) Receives the length of the SPIR-V code in bytes.
//     StatusOK       (bool&) A reference to an inherited status variable. If passed false,
//                    this routine returns immediately. If something goes wrong, the variable
//                    will be set false.
//  Returns:
//     (uint32_t*)    The address of the start of the memory into which the code has been read.
//                    This memory will have been allocated using new[] and should be deleted
//                    (using delete[]) by the calling routine once it is no longer needed. If
//                    there is an error, this routine will return a null pointer, so the caller
//                    needs to test for that rather than blindly run delete[] on a null pointer.
//  Pre-requisites:
//     The file name specified must be a string that can be passed to fopen(), and the file should
//     contain valid SPIR-V code. This routine uses no Vulkan calls, so could in principle be
//     called before any Vulkan initialisation.
                                                   
uint32_t* KVVulkanFramework::ReadSpirVFile(
        const std::string& Filename,long* LengthInBytes,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return nullptr;
    
    //  This is complicated slightly by the Vulkan specification requiring that SPIR-V binary code
    //  be presented to vkCreateShaderModule() - as it is here in CreateShaderModule() - as a
    //  pointer to an array of uint32_t values. In this routine we determine the size of the code
    //  in the specified file, allocate an array to hold it, and read into that array. However,
    //  we have to allow for the case where the file size is not a round number of uint32_t
    //  values. (In practice, any valid SPIR-V file will be, but we need to allow for any file
    //  we might be passed. But we should be able to handle any file at this point. If it's not
    //  a valid file, it's up to vkCreateShaderModule() to spot that later - which it will do,
    //  and will explain why if validation is enabled.)
    
    uint32_t* Buffer = nullptr;
    
    FILE* SpirVFile = fopen(Filename.c_str(), "rb");
    if (SpirVFile == nullptr) {
        LogError("Could not find or open file: %s.",Filename.c_str());
        StatusOK = false;
    } else {

        //  Get the file size by seeking to the end and getting our file position.
        
        fseek(SpirVFile,0,SEEK_END);
        *LengthInBytes = ftell(SpirVFile);
    
        //  Round up the size to a whole number of uint32_t values and allocate an array of that
        //  size. We zero out the last value, because if there is any padding it's probably nice
        //  not to leave it as random values. Then we read from the file. Note that we allow
        //  for a zero length file.
        
        if (*LengthInBytes > 0) {
            long LengthInUint32s = (*LengthInBytes + sizeof(uint32_t) - 1)/sizeof(uint32_t);
            Buffer = new uint32_t[LengthInUint32s];
            Buffer[LengthInUint32s - 1] = 0;
            fseek(SpirVFile,0,SEEK_SET);
            long Length = fread(Buffer,sizeof(char),*LengthInBytes,SpirVFile);
            if (Length != *LengthInBytes) {
                LogError("Error reading from file: %s.",Filename.c_str());
                LogError("Read %ld bytes, expected to read %ld bytes.",Length,*LengthInBytes);
                StatusOK = false;
            } else {
                I_Debug.Logf("Progress","Read shader code from '%s'",Filename.c_str());
            }
        }
        fclose(SpirVFile);
    }
    return Buffer;
}

//  ------------------------------------------------------------------------------------------------
//
//                       C r e a t e  S h a d e r  M o d u l e  (Internal routine)
//
//  This internal routine is passed the memory address of some SPIR-V code (which probably comes
//  from a from a specified file, as read using ReadSpirVFile()), creates a Vulkan shader module
//  from the SPIR-V code, and returns the Vulkan handle for that shader module. The shader module
//  can then be used in a pipeline, either for computation or for graphics.
//
//  Parameters:
//     Code           (uint32t*) The memory address of the start of the code (usually this
//                    will have been returned by ReadSpirVFile()).
//     LengthInBytes  (long) The length of the SPIR-V code in bytes.
//     StatusOK       (bool&) A reference to an inherited status variable. If passed false,
//                    this routine returns immediately. If something goes wrong, the variable
//                    will be set false.
//  Returns:
//     (VkShaderModule) The Vulkan handle for the newly created shader module.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//     Code and LengthInBytes should describe valid SPIR-V code.

VkShaderModule KVVulkanFramework::CreateShaderModule(
                                        uint32_t* Code,long LengthInBytes,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return VK_NULL_HANDLE;
    
    VkShaderModuleCreateInfo CreateInfo{};
    CreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    CreateInfo.codeSize = LengthInBytes;
    CreateInfo.pCode = Code;

    VkShaderModule ShaderModule;
    VkResult Result;
    Result = vkCreateShaderModule(I_LogicalDevice,&CreateInfo,nullptr,&ShaderModule);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError("Failed to create shader module","vkCreateShaderModule",Result);
        ShaderModule = VK_NULL_HANDLE;
        StatusOK = false;
    }
    return ShaderModule;
}

//  ------------------------------------------------------------------------------------------------
//
//                       G e t  M e m o r y  T y p e  I n d e x  (Internal routine)
//
//  This is an internal utility used by CreateVulkanBuffer(). A GPU typically provides a number of
//  different types of memory, each with different properties - some will be local to the GPU (and
//  so fast for it to access), some will be visible to the CPU, some won't, etc. Which type a
//  program should use for any given purpose depends on the details of just what properties it
//  would prefer the memory to have.
//
//  Parameters:
//     MemoryRequirements  (VkMemoryRequirements) The memory requirements for the buffer, as
//                    returned by vkGetBufferMemoryRequirements().
//     PropertyFlags  (VkMemoryPropertyFlags) Describes the memory properties for the buffer,
//                    GPU local, shared, etc. that the buffer needs to have, as specified by
//                    the program in its call to SetBufferDetails().
//     StatusOK       (bool&) A reference to an inherited status variable. If passed false,
//                    this routine returns immediately. If something goes wrong, the variable
//                    will be set false.
//  Returns:
//     (uint32_t)     An index into the set of memory types available. This routine has determined
//                    that this index specifies a suitable memory type to use for the buffer.
//
//  Pre-requisites:
//     FindSuitableDevice() must have been called to select the GPU device to be used.
//
//  Note:
//     This routine isn't trying to find the most suitable memory type - that would need much
//     closer interaction with the higher levels of the program. It merely returns the index of
//     the first memory type that meets the requirements.

uint32_t KVVulkanFramework::GetMemoryTypeIndex(
    VkMemoryRequirements MemoryRequirements,VkMemoryPropertyFlags PropertyFlags,bool& StatusOK)
{
    //  The physical device must have been identified and its handle set in I_SelectedDevice.
    
    if (!AllOK(StatusOK)) return 0;
    
    bool Found = false;
        
    //  The VkMemoryRequirements structure for a resource has a field memoryTypeBits that works
    //  with the list of memory types returned by vkGetPhysicalDeviceMemoryProperties(). Each bit
    //  of memoryTypeBits indicates whether the corresponding memory type returned by that routine
    //  is supported for the resource. (In this case, the resource is the buffer itself, with
    //  its requirements passed in MemoryRequirements.)
    
    uint32_t Index = 0;
    uint32_t SupportedMemoryTypeMask = MemoryRequirements.memoryTypeBits;
    
    //  We get the set of memory properties for each memory type supported by the device,
    //  and will check each one against what we need.
    
    VkPhysicalDeviceMemoryProperties MemoryProperties;
    vkGetPhysicalDeviceMemoryProperties(I_SelectedDevice, &MemoryProperties);
    
    if (I_Debug.Active("Properties")) ListMemoryProperties(&MemoryProperties);

    for (uint32_t I = 0; I < MemoryProperties.memoryTypeCount; I++) {
        
        //  First, is this memory type supported by the resource in question? (See above.)
        
        if (SupportedMemoryTypeMask & (1 << I)) {
            
            //  If so, are all the properties we've asked for provided by this memory type?
            //  (The test checks that all the bits in PropertyFlags are set in the property
            //  flags for the memory type, ignoring flags we don't care about.)
            
            if ((MemoryProperties.memoryTypes[I].propertyFlags & PropertyFlags) == PropertyFlags) {
                Index = I;
                Found = true;
                break;
            }
        }
    }
    
    if (!Found) {
        LogError ("Unable to find a memory type that meets requirements.");
        StatusOK = false;
    }
    return Index;
}

//  ------------------------------------------------------------------------------------------------
//
//                       L i s t  M e m o r y  P r o p e r t i e s  (Internal routine)
//
//  This is an internal diagnostic utility that can be used to list the properties of the
//  various memory types supported by the selected GPU (see GetMmeoryTypeIndex() for a little
//  more context).
//
//  Parameters:
//     Properties     (VkPhysicalDeviceMemoryProperties*) A structure describing all the memory
//                    types supported by the device. This should have been returned by a call to
//                    vkGetPhysicalDeviceMemoryProperties().
//  Output:
//     A listing of the available memory types and their properties.
//
//  Pre-requisites:
//     Strictly, none, although in practice, Properties will have been returned by a call to
//     vkGetPhysicalDeviceMemoryProperties().

void KVVulkanFramework::ListMemoryProperties(const VkPhysicalDeviceMemoryProperties* Properties)
{
    I_Debug.Log ("Properties","Memory properties:");
    
    //  A device has one or more memory 'heaps'. List them first. Some heaps are much larger
    //  than others and can support much larger memory allocations.
    
    I_Debug.Logf ("Properties","Heaps: %d",Properties->memoryHeapCount);
    for (unsigned int Heap = 0; Heap < Properties->memoryHeapCount; Heap++) {
        std::string FlagString = "";
        VkMemoryHeapFlags Flags = Properties->memoryHeaps[Heap].flags;
        if (Flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) FlagString += "DeviceLocal ";
        if (Flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) FlagString += "MultiInstance ";
        I_Debug.Logf ("Properties","Heap %d size %d %s",Heap,
                                             Properties->memoryHeaps[Heap].size,FlagString.c_str());
    }
    
    //  It also has a number of memory types, each associated with a specific heap and
    //  with its own specific properties.
    
    I_Debug.Logf ("Properties","Memory types: %d",Properties->memoryTypeCount);
    for (unsigned int Type = 0; Type < Properties->memoryTypeCount; Type++) {
        std::string FlagString = "";
        VkMemoryPropertyFlags Flags = Properties->memoryTypes[Type].propertyFlags;
        if (Flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) FlagString += "DeviceLocal ";
        if (Flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) FlagString += "HostVisible ";
        if (Flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) FlagString += "HostCoherent ";
        if (Flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) FlagString += "HostCached ";
        if (Flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) FlagString += "LazilyAllocated ";
        if (Flags & VK_MEMORY_PROPERTY_PROTECTED_BIT) FlagString += "PropertyProtected ";
        if (Flags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD) FlagString += "DeviceCoherentAMD ";
        if (Flags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD) FlagString += "DeviceUncachedAMD ";
        if (Flags & VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV) FlagString += "RDMACapableNV ";
        I_Debug.Logf ("Properties","Type %d Heap %d %s",Type,
                                       Properties->memoryTypes[Type].heapIndex,FlagString.c_str());
    }
}

//  ------------------------------------------------------------------------------------------------
//
//          G e t  I n d e x  F o r  Q u e u e  F a m i l y  T o  U s e  (Internal routine)
//
//  This is an internal utility used by CreateLogicalDevice(). Devices support multiple queues,
//  bundled up into queue 'families' in which each member has the same capabilities (ie can do
//  graphics, can do compute, can do other things like memory transfers). Each queue family has
//  its own index number. This routine selects a queue family to use for the sort of general
//  application the Framework is designed to handle, and returns its index so CreateLogicalDevice()
//  can use it.
//
//  Parameters:
//     UseGraphics    (bool) Set true if the application needs to use graphics.
//     UseCompute     (bool) Set true if the application needs to use the GPU for computation.
//     StatusOK       (bool&) A reference to an inherited status variable. If passed false,
//                    this routine returns immediately. If something goes wrong, the variable
//                    will be set false.
//  Returns:
//     (uint32_t)     An index into the set of queue families available. This routine has
//                    determined that this index specifies a suitable family to use.
//
//  Pre-requisites:
//     FindSuitableDevice() must have been called to select the GPU device to be used.
//
//  Note:
//     This routine isn't trying to find the most suitable queue family - that would need much
//     closer interaction with the higher levels of the program. It merely returns the index of
//     the first family that meets the requirements.

uint32_t KVVulkanFramework::GetIndexForQueueFamilyToUse(
               bool UseGraphics,bool UseCompute,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return 0;
    
    bool ListQueueProperties = true;
    
    uint32_t FamilyIndex = 0;
    
    //  Get the details of all the queue families supported by the device.
    
    uint32_t NumberFamilies;
    vkGetPhysicalDeviceQueueFamilyProperties(I_SelectedDevice,&NumberFamilies,nullptr);
    std::vector<VkQueueFamilyProperties> QueueFamilies(NumberFamilies);
    vkGetPhysicalDeviceQueueFamilyProperties(I_SelectedDevice,&NumberFamilies,QueueFamilies.data());

    //  This section lists the details of what each queue family supports. It gives an idea of
    //  what might be tested for (possibly even at the point when the device is selected, for
    //  programs that have specifically heavy requirements).
    
    if (ListQueueProperties) {
        int Index = 0;
        for (const VkQueueFamilyProperties& Props : QueueFamilies) {
            std::string PropString = "";
            if (Props.queueFlags & VK_QUEUE_GRAPHICS_BIT) PropString += " Graphics";
            if (Props.queueFlags & VK_QUEUE_COMPUTE_BIT) PropString += " Compute";
            if (Props.queueFlags & VK_QUEUE_TRANSFER_BIT) PropString += " Transfer";
            if (Props.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) PropString += " Binding";
            if (UseGraphics) {
                VkBool32 PresentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(I_SelectedDevice,Index,I_Surface,
                                                                         &PresentSupport);
                if (PresentSupport) PropString += " Present";
            }
            I_Debug.Logf("Properties","Queue family index %d Queues %d %s",Index,
                                                            Props.queueCount,PropString.c_str());
            Index++;
        }
    }

    //  Now we look for a suitable queue family. We just take the first one that actually has
    //  queues and which supports both compute and graphics, if that's what's been asked for.
    //  If we're using graphics, we want present support as well. Do we really expect all
    //  graphics queues to support present, or might we need a separate present queue?
    
    bool Found = false;
    int Index = 0;
    for (const VkQueueFamilyProperties &Props : QueueFamilies) {
        bool Suitable = true;
        if (Props.queueCount > 0) {
            if (UseGraphics && (Props.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) Suitable = false;
            if (UseCompute && (Props.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) Suitable = false;
            if (UseGraphics) {
                VkBool32 PresentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(I_SelectedDevice,Index,I_Surface,
                                                                              &PresentSupport);
                if (!PresentSupport) Suitable = false;
            }
            if (Suitable) {
                I_Debug.Logf("Properties","Selected queue at index: %d",Index);
                FamilyIndex = Index;
                Found = true;
                break;
            }
        }
        Index++;
    }
    
    //  You can imagine being clever if we don't find a suitable queue, expecially if we have
    //  more unusual requirements - we could return two different queue families, with different
    //  capabilities, but then we'd need a slightly different calling sequence just to get that
    //  more nuanced information returned. We might need to use separate graphics and present
    //  queues (which is what the Vulkan tutorial does).

    if (!Found) StatusOK = false;
    return FamilyIndex;
}

//  ------------------------------------------------------------------------------------------------
//
//                       R a t e  D e v i c e  (Internal routine)
//
//  This is an internal utility used by FindSuitableDevice(). If a machine has multiple GPU
//  devices, we need to pick one for use. This routine assigns a rough score to each device
//  based on its capabilities. Since we don't really have much detail about what the program
//  intends to do with the device, this routine looks mostly for what might be expected to be
//  the most powerful device.
//
//  Parameters:
//     DeviceHndl     (VkPhysicalDevice) A Vulkan handle for the physical device in question.
//
//  Returns:
//     (int)          A rough score. FindSuitableDevice() will assume zero indicates an unusable
//                    device, and will simply select the device given the highest score.
//
//  Pre-requisites:
//     FindSuitableDevice() must have been called to select the GPU device to be used.
//
//  Note:
//      This is more of an example than anything, but it's a reasonable way of scoring a physical
//      GPU device. We assume that if this routine has been called at all, then the device has
//      passed basic tests like those for support for the required extensions, so is suitable for
//      use. A score of zero implies unusable, so we give it more than that. We give it 1 in most
//      cases, another 10 if it's a discrete GPU, and another 10 if it supports double precision.
//      Anything fancier requires application-specific code.

int KVVulkanFramework::RateDevice (VkPhysicalDevice DeviceHndl)
{
    int Score = 1;
    VkPhysicalDeviceProperties Properties;
    vkGetPhysicalDeviceProperties(DeviceHndl,&Properties);
    if (Properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) Score += 10;
    VkPhysicalDeviceFeatures Features;
    vkGetPhysicalDeviceFeatures(DeviceHndl,&Features);
    if (Features.shaderFloat64) Score += 10;

    return Score;
}

//  ------------------------------------------------------------------------------------------------
//
//                       D e v i c e  E x t e n s i o n s  O K   (Internal routine)
//
//  This is an internal utility used by FindSuitableDevice(). Vulkan makes use of a number of
//  'extensions'; for example, a program using the GPU for display will need to support an
//  extension called 'VK_KHR_swapchain'. This routine checks a list of named extensions supported
//  by a given physical device against a list of those required by the program and returns
//  true if the device supports all those required.
//
//  Parameters:
//     GraphicsExtensions (const std::vector<const char*>&) The set of named extensions required
//                        by the program.
//     DeviceExtensions   (const std::vector<VkExtensionProperties>&) The set of extensions
//                        supported by the device - these details include their names.
//
//  Returns:
//     (bool)      True if the device supports all the required extensions, false otherwise.
//

bool KVVulkanFramework::DeviceExtensionsOK(const std::vector<const char*>& GraphicsExtensions,
                          const std::vector<VkExtensionProperties>& DeviceExtensions)
{
    bool AllPresent = true;
    for (const char* NamePtr : GraphicsExtensions) {
        I_Debug.Logf("Device","Checking for extension %s",NamePtr);
        bool Found = false;
        for (const VkExtensionProperties& Property : DeviceExtensions) {
            if (!strcmp(NamePtr,Property.extensionName)) {
                Found = true;
                break;
            }
        }
        if (!Found) {
            I_Debug.Logf("Device","Device does not support required extension '%s'.",NamePtr);
            AllPresent = false;
            break;
        }
    }
    return AllPresent;
}

//  ------------------------------------------------------------------------------------------------
//
//                     S h o w  D e v i c e  D e t a i l s   (Internal routine)
//
//  This is an internal utility used by FindSuitableDevice() which may call it as a diagnostic.
//  The details listed may be useful when testing a program. However, the main reason for having
//  this code here is to show just a little of the information that's available about a GPU.
//  What's listed here is just a subset. The full list of device properties can be found in
//  mind-boggling detail in the Vulkan documentation.
//
//  Parameters:
//     DeviceHndl     (VkPhysicalDevice) A Vulkan handle for the physical device in question.
//
//  Output:
//     Some possibly interesting details about the device.

void KVVulkanFramework::ShowDeviceDetails (VkPhysicalDevice DeviceHndl)
{
    
    VkPhysicalDeviceProperties Properties;
    vkGetPhysicalDeviceProperties(DeviceHndl,&Properties);
    I_Debug.Logf("Device","Device: %s",Properties.deviceName);
    I_Debug.Logf("Device","Max compute workgroup count: %d, %d, %d",
             Properties.limits.maxComputeWorkGroupCount[0],
             Properties.limits.maxComputeWorkGroupCount[1],
             Properties.limits.maxComputeWorkGroupCount[2]);
    I_Debug.Logf("Device","Max compute workgroup invocations: %d",
             Properties.limits.maxComputeWorkGroupInvocations);
    I_Debug.Logf("Device","Max compute workgroup size: %d %d %d",
             Properties.limits.maxComputeWorkGroupSize[0],
             Properties.limits.maxComputeWorkGroupSize[1],
             Properties.limits.maxComputeWorkGroupSize[2]);
    I_Debug.Logf("Device","Max storage buffer range: %d",
             Properties.limits.maxStorageBufferRange);
    VkPhysicalDeviceFeatures Features;
    vkGetPhysicalDeviceFeatures(DeviceHndl,&Features);
    I_Debug.Logf("Device","Double precision support: %s",
             ((Features.shaderFloat64) ? "Yes" : "No"));
}

//  ------------------------------------------------------------------------------------------------
//
//                     C l e a n u p  V u l k a n  G r a p h i c s
//
//  This routine cleans up the graphics aspects of Vulkan. A program that used Vulkan for nothing
//  other than computation does not need to call this routine. However, a program that uses
//  Vulkan in conjunction with a windowing system like GFLW has to close both down in a careful
//  sequence. This routine, CleanupVulkanGraphics() needs to be called before the graphics surface
//  is destroyed, which will probably happen when the window system being used is cleaned up. So
//  the cleanup sequence needs to be 1) CleanupVulkanGraphics(), 2) Cleanup Window System, 3)
//  CleanupVulkan(). The Window system will usually need access to the Vulkan instance to do its
//  cleanup, so can't be called after CleanupVulkan(), but Vulkan's use of graphics, particularly
//  the swap chain, needs to be cleaned up before the surface is destroyed. Hence this three part
//  cleanup sequence.

void KVVulkanFramework::CleanupVulkanGraphics (void)
{
    //  The main thing we have to do is cleanup the swap chain. Otherwise the windowing system
    //  will have problems deleting the display surface.
    
    CleanupSwapChain();
    
    //  And we can also get rid of the various other non-swap chain graphics resources - the
    //  various semaphores and the fences.
    
    for (VkSemaphore Semaphore : I_ImageSemaphoreHndls) {
        vkDestroySemaphore(I_LogicalDevice,Semaphore,nullptr);
    }
    I_ImageSemaphoreHndls.clear();
    for (VkSemaphore Semaphore : I_RenderSemaphoreHndls) {
        vkDestroySemaphore(I_LogicalDevice,Semaphore,nullptr);
    }
    I_RenderSemaphoreHndls.clear();
    for (VkFence Fence : I_FenceHndls) {
        vkDestroyFence(I_LogicalDevice,Fence,nullptr);
    }
    I_FenceHndls.clear();
}

//  ------------------------------------------------------------------------------------------------
//
//                               C l e a n u p  V u l k a n
//
//  This routine closes down Vulkan, having released all its resources. It can be called explicitly,
//  by a program that no longer needs Vulkan, or left to be called automatically from the
//  Framework's  destructor.
//
//  Note:
//     If validation is enabled, the validation layers are very good at spotting resources that
//     have not been closed down when the Vulkan instance is deleted.

void KVVulkanFramework::CleanupVulkan (void)
{
    //  This may well be called twice, once explicitly and once from the destructor, so when
    //  something is destroyed, either its handle must be set to null, or the vector that contains
    //  it and other handles must be cleared so the code doesn't attempt to destroy already
    //  deleted items the second time round.
    
    //  Make sure the Vulkan graphics resources have been released.
    
    CleanupVulkanGraphics();
    
    //  Now all the rest. Shader modules
    
    for (VkShaderModule ShaderModuleHndl : I_ShaderModuleHndls) {
        if (ShaderModuleHndl != VK_NULL_HANDLE) {
            vkDestroyShaderModule(I_LogicalDevice,ShaderModuleHndl,nullptr);
        }
    }
    I_ShaderModuleHndls.clear();

    //  Descriptor set layouts
    
    for (VkDescriptorSetLayout LayoutHndl : I_DescriptorSetLayoutHndls) {
        vkDestroyDescriptorSetLayout(I_LogicalDevice,LayoutHndl,nullptr);
    }
    I_DescriptorSetLayoutHndls.clear();
    
    //  Descriptor pools and command pools.
    
    for (VkDescriptorPool PoolHndl : I_DescriptorPoolHndls) {
        vkDestroyDescriptorPool(I_LogicalDevice,PoolHndl,nullptr);
    }
    I_DescriptorPoolHndls.clear();
    for (VkCommandPool PoolHndl : I_CommandPoolHndls) {
        vkDestroyCommandPool(I_LogicalDevice,PoolHndl,nullptr);
    }
    I_CommandPoolHndls.clear();

    //  All the buffers
    
    for (T_BufferDetails Details : I_BufferDetails) {
        if (Details.InUse) {
            vkFreeMemory(I_LogicalDevice,Details.MainBufferMemoryHndl,nullptr);
            vkDestroyBuffer(I_LogicalDevice,Details.MainBufferHndl,nullptr);
            if (Details.BufferAccess == ACCESS_STAGED_CPU ||
                Details.BufferAccess == ACCESS_STAGED_GPU ) {
                vkFreeMemory(I_LogicalDevice,Details.SecondaryBufferMemoryHndl,nullptr);
                vkDestroyBuffer(I_LogicalDevice,Details.SecondaryBufferHndl,nullptr);
            }
        }
    }
    I_BufferDetails.clear();
    
    //  Pipelines
    
    for (T_PipelineDetails Details : I_PipelineDetails) {
        vkDestroyPipelineLayout(I_LogicalDevice,Details.PipelineLayoutHndl,nullptr);
        vkDestroyPipeline(I_LogicalDevice,Details.PipelineHndl,nullptr);
    }
    I_PipelineDetails.clear();

    //  Render passes
    
    if (I_RenderPass != VK_NULL_HANDLE) vkDestroyRenderPass(I_LogicalDevice,I_RenderPass, nullptr);
    I_RenderPass = VK_NULL_HANDLE;

    //  Logical devices
    
    if (I_LogicalDevice != VK_NULL_HANDLE) vkDestroyDevice(I_LogicalDevice,nullptr);
    I_LogicalDevice = VK_NULL_HANDLE;
    
    //  And, finally, the instance itself, together with any disgnostic layers
    
    if (I_Instance != VK_NULL_HANDLE) {
        
        //  Delete the VkDebugUtilsMessengerEXT object used for handling diagnostics. As with its
        //  creation, we have to first locate the extension routine that can do this, and then
        //  call it.
        
        if (I_DebugMessenger != VK_NULL_HANDLE) {
            auto Function = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(I_Instance,
                                                        "vkDestroyDebugUtilsMessengerEXT");
            if (Function != nullptr) Function(I_Instance,I_DebugMessenger,nullptr);
            I_DebugMessenger = VK_NULL_HANDLE;
        }
        
        //  And now that we don't need the Vulkan instance any more, we can delete it.
        //  The physical device (or at least, the Vulkan handle to it) is regarded as created
        //  by the instance and is destroyed with it - we don't need to destroy I_PhysicalDevice.
        
        vkDestroyInstance(I_Instance,nullptr);
        I_Instance = VK_NULL_HANDLE;
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                 S e t u p  D e b u g  M e s s e n g e r  I n f o   (Internal routine)
//
//  This is an internal utility used by CreateVulkanInstance() which calls it as part of the
//  process of enabling validation layers, should that be required. It is passed a structure
//  of type VkDebugUtilsMessengerCreateInfoEXT, which is used to tell the validation layers
//  which routine to call when they have something to report, and under what circumstances
//  they should do so - basically what sort of validation and other messages should be reported.
//  This sets the static routine DebugUtilsCallback() as the routine for the validation layers to
//  call.
//
//  Parameters:
//     DebugInfo  (VkDebugUtilsMessengerCreateInfoEXT&) The structure to be set up.

void KVVulkanFramework::SetupDebugMessengerInfo (VkDebugUtilsMessengerCreateInfoEXT& DebugInfo)
{
    //  This enables the most useful messages. The callback routine has to be a static routine,
    //  as the Vulkan diagnostics have no idea of the KVVulkanFramework class, but the .pUserData
    //  field can be used to pass the address of the current KVVulkanFramework object to the
    //  callback routine. There are a often a lot of information messages generated, enabled by
    //  the VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT being set, but these are suppressed by
    //  default unless enabled explicitly using EnableValidationLevels().
    
    DebugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    DebugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    DebugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    DebugInfo.pfnUserCallback = DebugUtilsCallback;
    DebugInfo.pUserData = this;
}

//  ------------------------------------------------------------------------------------------------
//
//                         A d d  I n s t a n c e  E x t e n s i o n s
//
//  This routine is provided to enable the interaction between Vulkan and a windowing system being
//  used to provide the display surface Vulkan needs for graphics output. Such a windowing system
//  (such as GLFW) may - probably will - need Vulkan to support certain specific extensions, and
//  these will need to be explicitly enabled when the initial Vulkan Instance is created. So this
//  needs to be called before CreateVulkanInstance() if Vulkan graphics facilities are needed.
//  It is passed a list of named Vulkan extensions that the windowing system requires. GLFW has
//  a routine, glfwGetRequiredInstanceExtensions() that supplies the names needed for such a list.
//
//  Parameters:
//     ExtensionNames  (const std::vector<const char*>&) The list of named extensions that need
//                     to be enabled.
//
//  Note:
//     This routine must be called before CreateVulkanInstance().

void KVVulkanFramework::AddInstanceExtensions(
        const std::vector<const char*>& ExtensionNames,bool& StatusOK)
{
    //  All we have to do is add these names to I_RequiredInstanceExtensions.
    
    if (!AllOK(StatusOK)) return;
    I_Debug.Log ("Instance","Adding Required Instance Extensions.");
    for (const char* NamePtr : ExtensionNames) I_RequiredInstanceExtensions.push_back(NamePtr);
}

//  ------------------------------------------------------------------------------------------------
//
//                        A d d  G r a p h i c s  E x t e n s i o n s
//
//  This routine is similar to AddInstanceExtensions() but whereas AddInstanceExtensions() is used
//  to specify extensions that the Vulkan instance must support, AddGraphicsExtensions() is used
//  to specify extensions that any physical GPU to be used must support. This is used internally
//  to require swap chain support to be supported by any device to be used for graphics display,
//  but could also be used if any windowing system were to have specific device-level extension
//  requirements. (Note that GLFW does not have any such requirements.)
//
//  Parameters:
//     ExtensionNames  (const std::vector<const char*>&) The list of named extensions that need
//                     to be enabled.
//  Note:
//     This routine must be called before FindSuitableDevice().

void KVVulkanFramework::AddGraphicsExtensions(
    const std::vector<const char*>& ExtensionNames,bool& StatusOK)
{
    //  All we have to do is add these names to I_RequiredGraphicsExtensions.
    
    if (!AllOK(StatusOK)) return;
    I_Debug.Log ("Instance","Adding to Required Graphics Extensions.");
    for (const char* NamePtr : ExtensionNames) I_RequiredGraphicsExtensions.push_back(NamePtr);
}

//  ------------------------------------------------------------------------------------------------
//
//                    G e t  D i a g n o s t i c  L a y e r s    (internal routine)
//
//  There are a couple of places where the code needs to know which diagnostic layers are to be
//  enabled. This routine returns a vector of the names of the enabled layers.
//
//  Parameters: None.
//
//  Returns:
//     (const std::vector<const char*>&)  The list of diagnostic layers to be enabled.

const std::vector<const char*>& KVVulkanFramework::GetDiagnosticLayers(void)
{
    //  If we are enabling diagnostics, we will aim to use the VK_LAYER_KHRONOS_validation
    //  layer, if it is available. This is described as providing all the useful standard
    //  validation. For the moment at least, this name is hard-coded here.

    static const std::vector<const char*> LayerNames = {"VK_LAYER_KHRONOS_validation"};
    return LayerNames;
}

//  ------------------------------------------------------------------------------------------------
//
//                       L o g  V a l i d a t i o n  E r r o r    (internal routine)
//
//  This is the default routine called by DebugUtilsCallback() to log an error message.
//  It simply outputs it to the error stream, prefaced by a string to indicate that it
//  represents an error.
//
//  Parameters:
//     Message     (const char*) The address of a nul-terminated error string.
//
//  Note:
//     The message is ignored unless error messages have been enabled through EnableValidation()
//     which enables them by default, or through EnableValidationLevels() which provides more
//     explicit control.

void KVVulkanFramework::LogValidationError(const char* Message)
{
    if (I_EnableValidationErrors) std::cerr << "*** Error: " << Message << '\n';
}

//  ------------------------------------------------------------------------------------------------
//
//                       L o g  V a l i d a t i o n  W a r n i n g    (internal routine)
//
//  This is the default routine called by DebugUtilsCallback() to log a warning message.
//  It simply outputs it to the error stream, prefaced by a string to indicate that it
//  represents a warning.
//
//  Parameters:
//     Message     (const char*) The address of a nul-terminated warning.
//
//  Note:
//     The message is ignored unless warnings have been enabled through EnableValidation()
//     which enables them by default, or through EnableValidationLevels() which provides more
//     explicit control.

void KVVulkanFramework::LogValidationWarning(const char* Message)
{
    if (I_EnableValidationWarnings) std::cerr << "* Warning: " << Message << '\n';
}

//  ------------------------------------------------------------------------------------------------
//
//                         L o g  V a l i d a t i o n  I n f o    (internal routine)
//
//  This is the default routine called by DebugUtilsCallback() to log a informational message.
//  It simply outputs it to the output stream, prefaced by a string to indicate that it
//  represents an informational message.
//
//  Parameters:
//     Message     (const char*) The address of a nul-terminated warning.
//
//  Note:
//     The message is ignored unless informational messages have been enabled explicitly through
//     EnableValidationLevels().

void KVVulkanFramework::LogValidationInfo(const char* Message)
{
    if (I_EnableValidationInformation) std::cout << "Information: " << Message << '\n';
}

//  ------------------------------------------------------------------------------------------------
//
//                         D e b u g  U t i l s  C a l l b a c k    (internal routine)
//
//  This is the static routine set up to be called by the validation layers when they have
//  something to report. The calling sequence for this is defined by Vulkan.
//
//  Parameters:
//     MessageSeverity   (VkDebugUtilsMessageSeverityFlagBitsEXT) A Vulkan code indicating
//                       how the message has been classed - error, warning, information, etc.
//     MessageType       (VkDebugUtilsMessageTypeFlagsEXT) A Vulkan code indicating the type
//                       of the message - this is currently ignored by this routine. (It is
//                       used to indicate the sort of occurrence that has triggered the
//                       message - a violation of Vulkan rules, sub-optimal usage, etc...)
//     CallbackData      (const VkDebugUtilsMessengerCallbackDataEXT*) A Vulkan structure that
//                       includes the text of the message.
//     UserData          (void*) User-supplied data. This must the the address of the Framework,
//                       to allow Framework routines such as LogValidationError() to be called by
//                       this static routine.
//
//  Returns:
//     (VkBool)          Returned as VK_TRUE if the Vulkan call that triggered the message
//                       should be aborted, and VK_FALSE if not.

VKAPI_ATTR VkBool32 VKAPI_CALL KVVulkanFramework::DebugUtilsCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT MessageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT /*MessageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* CallbackData,
    void* UserData)
 {
    if (UserData) {
        
        //  Get the address of the Framework, and call the appropriate routine.
        
        KVVulkanFramework* FrameworkObject = (KVVulkanFramework*) UserData;
        if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
            FrameworkObject->LogValidationInfo(CallbackData->pMessage);
        } else if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
            FrameworkObject->LogValidationInfo(CallbackData->pMessage);
        } else if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            FrameworkObject->LogValidationWarning(CallbackData->pMessage);
        } else if (MessageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
            FrameworkObject->SetValidationError(true);
            FrameworkObject->LogValidationError(CallbackData->pMessage);
        } else {
            
            //  This catches any other message types, such as the vebose information type.
            
            FrameworkObject->LogValidationInfo(CallbackData->pMessage);
        }
    }
    
    //  The Vulkan tutorial says this should always return false - it's only when it's being used
    //  to test the validation layers themselves that it should return true.
    
    return VK_FALSE;
}

//  ------------------------------------------------------------------------------------------------
//
//                           S e t  V a l i d a t i o n  E r r o r    (internal routine)
//
//  This routine exists solely to allow the static routine DebugUtilsCallback() to flag that
//  the Vulcan validation layers have detected an error. It sets the Framework's internal flag
//  that indicates this. (The parameter allows it to be set or cleared, but at the moment there's
//  no call to have it cleared externally.) Technically this is not an internal routine, as it
//  needs to be callable by DebugUtilsCallback(), but it is not intended to be called from
//  outside the Framework code.
//
//  Parameters:
//     Set           (bool) True if the internal 'validation error detected' flag is to be set.
//                   False if it is to be cleared.

void KVVulkanFramework::SetValidationError(bool Set)
{
    I_ValidationErrorFlagged = Set;
}

//  ------------------------------------------------------------------------------------------------
//
//                                  L o g  E r r o r      (internal routine)
//
//  Logs an error message, formatting it like printf(). The supplied message will be formatted and
//  output, with a new line character appended. This also sets the internal 'error' flag tested
//  by AllOK().
//
//  Parameters:
//     Format        (const char* const) A printf() style formatting string. This can be followed
//                   by a variable number of additional arguments as required.


void KVVulkanFramework::LogError (const char* const Format, ...)
{
    char Message[1024];
    va_list Args;
    va_start (Args,Format);
    vsnprintf (Message,sizeof(Message),Format,Args);
    cerr << "[" << I_Debug.GetSubSystem() << "] *** Error: " << Message << " ***\n";
    I_ErrorFlagged = true;
}

//  ------------------------------------------------------------------------------------------------
//
//                             L o g  V u l k a n  E r r o r      (internal routine)
//
//  Should be called when a Vulkan routine returns an error. It logs the details of the error,
//  including the name of the Vulkan routine and a readable version of the status it returned.
//
//  Parameters:
//     Text        (const std::string&) The main text to be output. This should be used to
//                 give some idea of the circumstances in which the error occurred.
//     Routine     (const std::string&) The name of the Vulkan routine returning the error code.
//     Result      (VkResult) The Vulkan status code returned by the routine.

void KVVulkanFramework::LogVulkanError(
        const std::string& Text,const std::string& Routine, VkResult Result)
{
    std::string ResultText(string_VkResult(Result));
    std::string Message = Text + " " + Routine + " returned code " + ResultText + ".";
    LogError(Message.c_str());
    I_ErrorFlagged = true;
}

//  ------------------------------------------------------------------------------------------------
//
//              D e v i c e  H a s  P o r t a b i l i t y  S u b s e t    (internal routine)
//
//  A tricky issue is to do with the portability subset. Mostly this an issue for code running
//  on Macintoshes using MoltenVK, but in principle it's more general. Vulkan allows for
//  devices that support a limited set of features, the portability subset. MoltenVK is a layer
//  that uses the Macintosh native Metal system to emulate Vulkan, and it implements the
//  portability subset. Vulkan doesn't want people to be using devices that only support a
//  subset to do so blindly - not realising they are using a limited implementation. So if
//  a device only supports the portability subset, Vulkan requires that the code explicitly
//  request that subset. That way, Vulkan assumes you know what you're doing. So we look to
//  see if this is such a device, and if so (we're probably on a Mac) we will need to add it
//  to the enabled extensions. This routine checks the list of extensions supported by the
//  selected device and returns true if they include the portability subset.
//
//  Parameters:
//     Extensions   (std::vector<VkExtensionProperties>&) Lists the details of the extensions
//                  supported by the device, including their names.
//
//  Returns:
//     (bool)        True if the list of device extensions includes the portability subset.

bool KVVulkanFramework::DeviceHasPortabilitySubset(std::vector<VkExtensionProperties>& Extensions)
{
    bool Found = false;
    for (const VkExtensionProperties& Property : Extensions) {
        if (!strcmp("VK_KHR_portability_subset",Property.extensionName)) {
            Found = true;
            break;
        }
    }
    return Found;
}

//  ------------------------------------------------------------------------------------------------
//
//                                  M a p  B u f f e r
//
//  This routine maps the CPU memory allocated for a buffer and returns a pointer to it.
//  The pointer will remain valid so long as the buffer remains unchanged. However, once
//  the buffer is deleted, its address range will no longer be accessible. If the buffer
//  is resized, it will generally be unmapped, and should be remapped before attempting
//  to access its data. This routine can be called multiple times for a given buffer - the
//  first call will result in a call to Vulkan to actually map the buffer, subsequent calls
//  will simply return the mapped address. A buffer will be unmapped when the Framework is
//  shut down or the buffer deleted, and can be explicitly unmapped through a call to
//  UnmapBuffer().
//
//  Parameters:
//     BufferHandle  (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Returns:
//     (void*)       The address of the mapped buffer.
//
//  Pre-requisites:
//     SetBufferDetails() must have been called to describe how the buffer will be used
//     and to return the buffer handle that is passed to this routine and the buffer and its
//     associated memory should already have been created using CreateBuffer().
//
//  Note:
//     If the buffer is local to the GPU (ie if the call to SetBufferDetails() specified "LOCAL"
//     as the access string), its data cannot be mapped and this routine will return with bad
//     status.

void* KVVulkanFramework::MapBuffer(KVBufferHandle BufferHndl,long* SizeInBytes,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return nullptr;
    
    *SizeInBytes = 0;
    int Index = BufferIndexFromHandle(BufferHndl,StatusOK);
    void* MappedAddress = nullptr;
    if (AllOK(StatusOK)) {
        
        //  See if the buffer has already been mapped, in which case we know the mapped address.

        if (I_BufferDetails[Index].MappedAddress) {
            MappedAddress = I_BufferDetails[Index].MappedAddress;
        } else {
            
            //  If not, we have to map it now. Note that we map the whole of the allocated
            //  memory (which can be more than the current size of the buffer, if the buffer
            //  was ever resized down).
            
            if (vkMapMemory(I_LogicalDevice,I_BufferDetails[Index].MainBufferMemoryHndl,0,
                I_BufferDetails[Index].MemorySizeInBytes,0,&MappedAddress) != VK_SUCCESS) {
                MappedAddress = nullptr;
                StatusOK = false;
            } else {
                
                //  Record the mapped address in the buffer details.
                
                I_BufferDetails[Index].MappedAddress = MappedAddress;
            }
        }
        if (AllOK(StatusOK)) *SizeInBytes = I_BufferDetails[Index].SizeInBytes;
    }
    return MappedAddress;
}

//  ------------------------------------------------------------------------------------------------
//
//                                  U n m a p  B u f f e r
//
//  This routine unmaps the CPU memory allocated for a buffer. Following this, its address
//  range as returned by MapBuffer() will no longer be accessible (although until the buffer is
//  deleted it can always be remapped if necessary). A buffer will be unmapped automatically when
//  the Framework is shut down or the buffer deleted, and this routine does not need to be called
//  as a preliminary to deleting a buffer.
//
//  Parameters:
//     BufferHandle  (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Pre-requisites:
//     MapBuffer() must previously have been called to map the buffer.
//
//  Note:
//     If the buffer has not in fact been mapped, this routine returns without indicating an error.

void KVVulkanFramework::UnmapBuffer(KVBufferHandle BufferHndl,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    int Index = BufferIndexFromHandle(BufferHndl,StatusOK);
    if (AllOK(StatusOK)) {
        
        //  Only unmap the buffer if it's actually been mapped.
        
        if (I_BufferDetails[Index].MappedAddress) {
            vkUnmapMemory(I_LogicalDevice,I_BufferDetails[Index].MainBufferMemoryHndl);
            I_BufferDetails[Index].MappedAddress = nullptr;
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                                  S y n c  B u f f e r
//
//  This routine synchronizes a staged buffer. Such a buffer is implemented using two separate
//  buffers of the same size, one local to the CPU and one local to the GPU, and the program
//  using them is responsible for indicating when they need to be brought into sync. These are
//  usually used when there are significant overheads associated with the use of shared buffers
//  (ones that both the CPU and GPU can access, and which rely on the underlying system to ensure
//  that the CPU and GPU both see the same data when they do access them). In such cases, a program
//  may for example, create a 'staging' buffer on the CPU, write into it and then copy its contents
//  in one go - presumably this is a relatively efficient process - to the corresponding GPU buffer.
//  Alternatively, the GPU may be used to fill a buffer local to the GPU, and this can then be
//  copied over to the corresponding CPU buffer. So the buffer handle passed should be that for
//  a buffer created as either "STAGED_CPU" or "STAGED_GPU", and which was used will determine the
//  direction of transfer set up by this routine - a "STAGED_CPU" buffer is synched by copying the
//  CPU side buffer over to the GPU side buffer, and a "STAGED_GPU" buffer is synched by a transfer
//  in the opposite direction.
//
//  Parameters:
//     BufferHandle  (KVBufferHandle) An opaque handle used by the Framework to refer to the buffer,
//                   as returned by SetBufferDetails().
//     CommandPoolHndl (VkCommandPool) A Vulkan handle specifying the command pool to be used
//                   to set up the required transfer.
//     QueueHandl    (VkQueue) A Vulkan handle specifying the queue to be used for the required
//                   transfer.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//  Pre-requisites:
//     o SetBufferDetails() must have been called to return the buffer handle that is passed to
//     this routine, the buffer should have access "STAGED_GPU" or "STAGED_CPU" and the CPU and
//     GPU buffers and their associated memory should already have been created using
//     CreateBuffer().
//     o The transfer is performed by setting up a transfer operation on the GPU, and this requires
//     allocation of a command buffer from a pool, and a queue to run that command on. The command
//     pool should be one created using CreatCommandPool(), and the queue should be obtained using
//     by GetDeviceQueue().
//
//  Note:
//     If the buffer is not in fact a staged buffer, this routine returns without indicating an
//     error. This makes it easier to experiment with the use of shared buffers as opposed to
//     staged ones - just include the call to SyncBuffer() in both cases where needed, and this
//     will work for either shared or staged buffers.

void KVVulkanFramework::SyncBuffer(KVBufferHandle BufferHndl,VkCommandPool CommandPoolHndl,
                                                           VkQueue QueueHndl,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    int Index = BufferIndexFromHandle(BufferHndl,StatusOK);
    if (AllOK(StatusOK)) {
        
        //  If this isn't a staged buffer, quietly do nothing - a sync is a null operation.
        
        if (I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_CPU ||
            I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_GPU) {
            
            //  The sync operation consists of a copy of data between the CPU-visible buffer and
            //  the local buffer used by the GPU. Which way depends on whether the buffer access
            //  type (STAGED_GPU or STAGED_CPU) - but that comes right at the end. First we need
            //  a command buffer for the copy.
            
            VkCommandBufferAllocateInfo AllocInfo{};
            AllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            AllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            AllocInfo.commandPool = CommandPoolHndl;
            AllocInfo.commandBufferCount = 1;
            
            VkCommandBuffer CommandBuffer;
            vkAllocateCommandBuffers(I_LogicalDevice,&AllocInfo,&CommandBuffer);
            
            //  Then we need to set up the command buffer. All it's going to run is a copy
            //  command. So we begin the command buffer setup..
            
            VkCommandBufferBeginInfo BeginInfo{};
            BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            BeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            
            vkBeginCommandBuffer(CommandBuffer,&BeginInfo);
            
            //  Then we set up the details of the copy command. All this needs is the source and
            //  destination buffers involved, the number of bytes to copy, and the offsets within
            //  each buffer. Which is the source buffer and which the destination depends on the
            //  whether this is a buffer where the CPU creates the data which then has to be
            //  copied to the GPU (STAGED_CPU) or where the GPU computes the data which then has
            //  to be copied to the CPU (STAGED_GPU). There's only one copy region involved.
            
            VkBufferCopy CopyRegion{};
            CopyRegion.size = I_BufferDetails[Index].SizeInBytes;
            CopyRegion.srcOffset = 0;
            CopyRegion.dstOffset = 0;
            VkBuffer SrcBufferHndl,DstBufferHndl;
            if (I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_CPU) {
                SrcBufferHndl = I_BufferDetails[Index].MainBufferHndl;
                DstBufferHndl = I_BufferDetails[Index].SecondaryBufferHndl;
            } else {
                SrcBufferHndl = I_BufferDetails[Index].SecondaryBufferHndl;
                DstBufferHndl = I_BufferDetails[Index].MainBufferHndl;
                
            }
            vkCmdCopyBuffer(CommandBuffer,SrcBufferHndl,DstBufferHndl,1,&CopyRegion);
            
            vkEndCommandBuffer(CommandBuffer);
            
            //  Then we submit the command buffer on the specified queue, and wait for the
            //  copy to finish.
            
            VkSubmitInfo SubmitInfo{};
            SubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            SubmitInfo.commandBufferCount = 1;
            SubmitInfo.pCommandBuffers = &CommandBuffer;
            
            vkQueueSubmit(QueueHndl,1,&SubmitInfo,VK_NULL_HANDLE);
            vkQueueWaitIdle(QueueHndl);
            
            //  Remembering to release the command buffer once we've finished with it.
            
            vkFreeCommandBuffers(I_LogicalDevice,CommandPoolHndl,1,&CommandBuffer);
            
            //  Just to note that there hasn't been any status checking in this sequence, and
            //  there really ought to be.
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                S w a p  C h a i n  S u p p o r t  A d e q u a t e    (internal routine)
//
//  This internal routine is called by FindSuitableDevice(), to see if a given GPU provides
//  support for the swap chain that will be needed for graphics display.
//
//  Parameters:
//     DeviceHndl   (VkPhysicalDevice) A Vulkan handle for the physical device in question.
//
//  Returns:
//     (bool)       True if the device provides sufficient support for the way the Framework
//                  will use the swap chain.
//
//  Pre-requisites:
//    If graphics are enabled, the window handler must have created the window surface to be
//    used and EnableGraphics() must have already been called. This routine will only be called if
//    graphics has been enabled.

bool KVVulkanFramework::SwapChainSupportAdequate(VkPhysicalDevice DeviceHndl,bool &StatusOK)
{
    //  This regards a device's swap chain support as adequate if it supports any surface
    //  format and at least one present mode that will work with the current graphics surface.
    //  That's a pretty low bar. ('Present' here is about presenting the image to the display.)
    
    if (!AllOK(StatusOK)) return false;
    bool SwapChainAdequate = true;
    if (I_GraphicsEnabled) {
        uint32_t FormatCount,PresentModeCount;
        if (I_Surface == VK_NULL_HANDLE) {
            LogError("Cannot check swap chain, because no graphics surface has been specified");
            StatusOK = false;
        } else {
            I_Debug.Log("Swapchain","Using surface to check swap chain.");
            
            //  I_Surface will have been set by EnableGraphics().
            
            vkGetPhysicalDeviceSurfaceFormatsKHR(DeviceHndl,I_Surface,&FormatCount,nullptr);
            vkGetPhysicalDeviceSurfacePresentModesKHR(DeviceHndl,I_Surface,
                                                                 &PresentModeCount,nullptr);
            if ((FormatCount == 0) || (PresentModeCount == 0)) SwapChainAdequate = false;
        }
    }
    return SwapChainAdequate;
}

//  ------------------------------------------------------------------------------------------------
//
//                             C r e a t e  S w a p  C h a i n
//
//  Vulkan uses a 'swap chain' as an abstraction that hides the grisly details of how images
//  are actually displayed on a screen. It acts like a chain of 'images' that can be displayed
//  in turn. The program acquires access to an available image from the chain, renders what it
//  wants to display to that image, ie sets the contents of the image, and then presents it for
//  display. Once the image has been displayed it can be returned to the swap chain for reuse.
//  Hiding all this behind the swap chain abstraction allows Vulkan a deal of flexibility
//  as to what actually happens to make this work. But it means we're dealing with this somewhat
//  opaque concept rather than with anything easier to visualise. The Framework supports a
//  fairly straightforward way of working with a swap chain. This routine creates a swap chain
//  able to handle a specified number of images. If the swap chain can only handle a smaller
//  number of images, this routine will set it to handle the most it can, and will return that
//  smaller number.
//
//  Parameters:
//     RequestedImages (uint32_t) The number of images the caller would like the swap chain
//                     to support.
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//  Returns:
//     (uint32_t)      The actual number of images the created swap chain can support.
//
//  Pre-requisites:
//    Graphics must have been enabled, using EnableGraphics(), and the logical device must have
//    been created using CreateLogicalDevice().

uint32_t KVVulkanFramework::CreateSwapChain(uint32_t RequestedImages,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return 0;
    
    //  Setting up a swap chain is quite complicated. This code pretty much follows what's
    //  described in the vulkan tutorial, which is worth looking at for more details about
    //  the choices made. (vulkan-tutorial.com)
    
    //  We pick the colour format to use, on the basis of what the selected device supports,
    //  and also pick the way data is to be presented.
    
    VkSurfaceFormatKHR SurfaceFormat = PickSwapSurfaceFormat(StatusOK);
    VkPresentModeKHR PresentMode = PickSwapPresentMode(StatusOK);
    
    //  We need to know something of what the display surface being is are capable of -
    //  number of images it can handle at once, any size restrictions, etc.
    
    VkSurfaceCapabilitiesKHR Capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(I_SelectedDevice,I_Surface,&Capabilities);
    
    //  Swap extent is to do with the resolution of the swap chain images, and generally
    //  should match the resolution of the window used for the display.
    
    VkExtent2D Extent = PickSwapExtent(Capabilities,StatusOK);

    //  We use one more than the minimum number of images the device requires - this gives it
    //  some more flexibility when overlapping setup and display. Basically, we set up an image
    //  for display, then leave it to the swap chain to handle getting it displayed, dealing
    //  with things like the device frame rate and so on, so the chain is usually handling a
    //  queue of images.
    
    uint32_t ImageCount = RequestedImages;
    if (ImageCount == 0) ImageCount = Capabilities.minImageCount + 1;
    if (Capabilities.maxImageCount > 0 && ImageCount > Capabilities.maxImageCount) {
        ImageCount = Capabilities.maxImageCount;
    }
    if (ImageCount < Capabilities.minImageCount) ImageCount = Capabilities.minImageCount;
    I_Debug.Logf("Swapchain","Swap chain min image count: %d, max image count: %d",
                                    Capabilities.minImageCount,Capabilities.minImageCount);
    I_Debug.Logf("Swapchain","Using swap chain imageCount = %d",ImageCount);
    I_ImageCount = ImageCount;

    //  And now we can fill in the structure Vulkan needs to specify the swap chain in full.
    
    VkSwapchainCreateInfoKHR CreateInfo{};
    CreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    CreateInfo.surface = I_Surface;
    CreateInfo.minImageCount = ImageCount;
    CreateInfo.imageFormat = SurfaceFormat.format;
    CreateInfo.imageColorSpace = SurfaceFormat.colorSpace;
    CreateInfo.imageExtent = Extent;
    CreateInfo.imageArrayLayers = 1;
    CreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    //  If we end up using different queue families for graphics and present, then we need to
    //  specify the sharing mode as VK_SHARING_MODE_CONCURRENT and specify the queue families
    //  explicitly at this point using .queueFamilyIndexCount and .pQueueFamilyIndices. For
    //  the moment, we assume we use the same queue family for both, and at this point we
    //  don't even need to call GetIndexForQueueFamilyToUse() to find out what it is.
    
    CreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;

    CreateInfo.preTransform = Capabilities.currentTransform;
    CreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    CreateInfo.presentMode = PresentMode;
    CreateInfo.clipped = VK_TRUE;
    
    //  The swap chain will need to be re-created if the window size changes, because it's been
    //  set up knowing the image size it has to handle. If this happens, it's handled by
    //  RecreateSwapChain(), and this calls CleanupSwapChain() which explicitly waits for
    //  the device to be idle. It's possible to be cleverer and create a new swap chain while
    //  the old one is still running, and if you do this you have to tell vkCreateSwapchainKHR()
    //  about the old one. We aren't doing this.
    
    CreateInfo.oldSwapchain = VK_NULL_HANDLE;

    //  And finally we create the swap chain.
    
    VkResult Result;
    Result = vkCreateSwapchainKHR(I_LogicalDevice,&CreateInfo,nullptr,&I_SwapChain);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError("Failed to create swap chain","vkCreateSwapchainKHR",Result);
        ImageCount = 0;
        StatusOK = false;
    } else {

        //  Now that we know how many images the swap chain holds, we need to be able to
        //  access them as the program runs. We find out just how many were actually created,
        //  and get their Vulkan handles in I_SwapChainImages.
        
        vkGetSwapchainImagesKHR(I_LogicalDevice,I_SwapChain,&ImageCount,nullptr);
        I_SwapChainImages.resize(ImageCount);
        vkGetSwapchainImagesKHR(I_LogicalDevice,I_SwapChain,&ImageCount,I_SwapChainImages.data());
        I_SwapChainImageFormat = SurfaceFormat.format;
        I_SwapChainExtent = Extent;
    }
    
    return ImageCount;
}

//  ------------------------------------------------------------------------------------------------
//
//                         R e c r e a t e  S w a p  C h a i n    (internal routine)
//
//  The swap chain needs to be recreated if the display window size changes. This can happen
//  quite frequently, not just as everything closes down, so it has its own separate routine.
//
//  Parameters:
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.

void KVVulkanFramework::RecreateSwapChain (bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  This assumes the render pass doesn't need to be recreated - it may if the display window
    //  is moved to a different resolution display.
    
    CleanupSwapChain();
    I_ImageCount = CreateSwapChain(I_ImageCount,StatusOK);
    CreateImageViews(StatusOK);
    CreateFramebuffers(StatusOK);
}

//  ------------------------------------------------------------------------------------------------
//
//                         C l e a n u p  S w a p  C h a i n    (internal routine)
//
//  The swap chain needs to be recreated if the display window size changes. This can happen
//  quite frequently, not just as everything closes down, so it has its own separate routine.

void KVVulkanFramework::CleanupSwapChain(void)
{
    if (I_LogicalDevice) vkDeviceWaitIdle(I_LogicalDevice);
    
    for (VkFramebuffer Framebuffer : I_SwapChainFramebuffers) {
        vkDestroyFramebuffer(I_LogicalDevice,Framebuffer,nullptr);
    }
    I_SwapChainFramebuffers.clear();
    
    for (VkImageView ImageView : I_SwapChainImageViews) {
        vkDestroyImageView(I_LogicalDevice,ImageView,nullptr);
    }
    I_SwapChainImageViews.clear();

    if (I_SwapChain != VK_NULL_HANDLE) vkDestroySwapchainKHR(I_LogicalDevice,I_SwapChain,nullptr);
    I_SwapChain = VK_NULL_HANDLE;
}

//  ------------------------------------------------------------------------------------------------
//
//            P i c k  S w a p  C h a i n  S u r f a c e  F o r m a t    (internal routine)
//
//  This is an internal routine used by CreateSwapChain(). It picks a format to be used for the
//  swap chain images based on what the selected device supports. It returns a VkSurfaceFormatKHR,
//  specifying a format and a colour space. The format specifies how colours are stored - number of
//  bits for blue, green, red and alpha, and in which order, and the colour space to be used - the
//  standard one is SRGB.
//
//  Parameters:
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//  Returns:
//     (VkSurfaceFormatKHR) The selected format.
//
//  Pre-requisites:
//    Graphics must have been enabled, using EnableGraphics(), and the physical device to use
//    must have been selected using FindSuitableDevice().
//
//  Note:
//    More specialised applications may want more control over the format than this provides.

VkSurfaceFormatKHR KVVulkanFramework::PickSwapSurfaceFormat(bool& StatusOK)
{
    //  If we're passed bad status, bail out, but we need to return something, so we need to
    //  set some dummy initial value we can return.
    
    VkSurfaceFormatKHR ChosenFormat{VK_FORMAT_UNDEFINED,VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    if (!AllOK(StatusOK)) return ChosenFormat;
    
    //  Check through all the available formats supported by the selected device. The preferred
    //  option is support for 8-Bit SRGB mode, and if we find that we take it. Otherwise, we
    //  settle for the first available option.
    
    uint32_t FormatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(I_SelectedDevice,I_Surface,&FormatCount,nullptr);
    if (FormatCount > 0) {
        std::vector<VkSurfaceFormatKHR> PossibleFormats(FormatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(I_SelectedDevice,I_Surface,
                                                             &FormatCount,PossibleFormats.data());
        ChosenFormat = PossibleFormats[0];
        
        //  If enabled, list the possible formats.
        
        if (I_Debug.Active("Swapchain")) {
            I_Debug.Logf("Swapchain","Device supports %d format(s)",FormatCount);
            for (const auto& Format : PossibleFormats) {
                I_Debug.Logf("Swapchain","Format: %s",string_VkFormat(Format.format));
            }
        }
        
        //  Now select the one we want.
        
        for (const auto& Format : PossibleFormats) {
            if (Format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                             Format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                ChosenFormat = Format;
                break;
            }
        }
        I_Debug.Logf("Swapchain","Chosen format: %s",string_VkFormat(ChosenFormat.format));
    } else {
        LogError(
         "Cannot pick graphics format. vkGetPhysicalDeviceSurfaceFormatsKHR reports zero formats.");
        StatusOK = false;
    }
    return ChosenFormat;
}

//  ------------------------------------------------------------------------------------------------
//
//                P i c k  S w a p  P r e s e n t  M o d e   (internal routine)
//
//  This is an internal routine used by CreateSwapChain(). It picks a presentation mode to be
//  used when images are to be shown on the screen. For example, the chain can simply present
//  images in its queue when and as the display is refreshed (FIFO mode), or can force an image to
//  be submitted immediately (which may mean it gets mixed in with part of the previous image,
//  causing 'tearing'). 'Mailbox' mode is a compromise that doesn't let the image queue overflow
//  (it replaces waiting images in the queue with new ones, rather than displaying them all
//  as best it can).
//
//  Parameters:
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//  Returns:
//     (VkPresentModeKHR) The selected presentation mode.
//
//  Pre-requisites:
//    Graphics must have been enabled, using EnableGraphics(), and the physical device to use
//    must have been selected using FindSuitableDevice().
//
//  Note:
//    More specialised applications may want more control over the mode than this provides.

VkPresentModeKHR KVVulkanFramework::PickSwapPresentMode(bool& StatusOK)
{
    //  FIFO mode is guaranteed to be available, but isn't the preferred mode. If bad status
    //  is passed, bail out and since we need to return a value, return FIFO.
    
    VkPresentModeKHR ChosenMode = VK_PRESENT_MODE_FIFO_KHR;
    if (!AllOK(StatusOK)) return ChosenMode;
    
    //  Look at the modes available. If Mailbox mode is available, we'll take that. If not,
    //  we just settle for the first one that's supported.
    
    uint32_t ModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(I_SelectedDevice,I_Surface,&ModeCount,nullptr);
    if (ModeCount > 0) {
        std::vector<VkPresentModeKHR> PossibleModes(ModeCount);
        ChosenMode = PossibleModes[0];
        if (I_Debug.Active("Swapchain")) {
            I_Debug.Logf("Swapchain","Device supports %d present mode(s)",ModeCount);
            for (const auto& Mode : PossibleModes) {
                I_Debug.Logf("Swapchain","Mode: %s",string_VkPresentModeKHR(Mode));
            }
        }
        for (const auto& Mode : PossibleModes) {
            if (Mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                ChosenMode = Mode;
                break;
            }
        }
        I_Debug.Logf("Swapchain","Chosen mode: %s",string_VkPresentModeKHR(ChosenMode));
    }
    return ChosenMode;
}

//  ------------------------------------------------------------------------------------------------
//
//                   P i c k  S w a p  E x t e n t  (internal routine)
//
//  This is an internal routine used by CreateSwapChain(). It returns the dimensions to be used
//  for the swap chain images. Normally, these should match the resolution of the window and this
//  is given by the currentExtent member of the structure describing the device capabilities.
//  However, some window managers are more flexible and let us specify a resolution, in which
//  case we pick dimensions that match the framebuffer being used by the windowing system. Either
//  way, this routine returns the image extent to be used as a VkExtent2D structure.
//
//  Parameters:
//     Capabilities    (const VkSurfaceCapabilitiesKHR&) Describes the capabilities of the
//                     display surface being used, particularly what image dimensions it can handle.
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//  Returns:
//     (VkExtent2D) The image dimensions to use.
//
//  Pre-requisites:
//    Graphics must have been enabled, using EnableGraphics(), the physical device to use
//    must have been selected using FindSuitableDevice(), and SetFrameBufferSize() should have
//    been called to supply the current window size.

VkExtent2D KVVulkanFramework::PickSwapExtent(
                const VkSurfaceCapabilitiesKHR& Capabilities,bool& StatusOK)
{
    VkExtent2D Extent{0,0};
    if (!AllOK(StatusOK)) return Extent;

    //  Unless the window manager is being more flexible (indicated by its returning a specific
    //  magic number of the width), we simply return the reported extent.
    
    if (Capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        Extent = Capabilities.currentExtent;
        
    } else {
        
        //  This handles the special case where the window manager allows us to pick the extent
        //  and this will need us to know the frame buffer size being used by the windowing
        //  system (which should have been set by a call to SetFrameBufferSize()).
        
        if (I_FrameBufferWidth == 0 || I_FrameBufferHeight == 0) {
             LogError("Cannot specify swap chain extent, as frame buffer size not specified.");
            StatusOK = false;
        } else {
            
            uint32_t Width = I_FrameBufferWidth;
            uint32_t Height = I_FrameBufferHeight;

            Width = std::clamp(Width,Capabilities.minImageExtent.width,
                                                          Capabilities.maxImageExtent.width);
            Height = std::clamp(Height,Capabilities.minImageExtent.height,
                                                         Capabilities.maxImageExtent.height);
            Extent.width = Width;
            Extent.height = Height;
        }
    }
    return Extent;
}

//  ------------------------------------------------------------------------------------------------
//
//                            C r e a t e  I m a g e  V i e w s
//
//  Each image being used by the swap chain needs an image view, describing how it is to be
//  handled. This routine sets up the image views needed by the swap chain, and should generally
//  be called directly after CreateSwapChain().
//
//  Parameters:
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//
//  Pre-requisites:
//    The swap chain should have been set up, using CreateSwapChain().
//
//  Notes:
//    This should be called by a program as part of the basic graphics sequence, but is also
//    called internally by as part of the resizing process that has to happen if the display window
//    is resized.

void KVVulkanFramework::CreateImageViews(bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  This routine sets the image views up as fairly straightforward 2D images.
    
    int NumberImages = int(I_SwapChainImages.size());
    I_SwapChainImageViews.resize(NumberImages);

    for (int Index = 0; Index < NumberImages; Index++) {
        VkImageViewCreateInfo CreateInfo{};
        CreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        CreateInfo.image = I_SwapChainImages[Index];
        CreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        CreateInfo.format = I_SwapChainImageFormat;
        CreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        CreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        CreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        CreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        CreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        CreateInfo.subresourceRange.baseMipLevel = 0;
        CreateInfo.subresourceRange.levelCount = 1;
        CreateInfo.subresourceRange.baseArrayLayer = 0;
        CreateInfo.subresourceRange.layerCount = 1;

        VkResult Result;
        Result = vkCreateImageView(I_LogicalDevice,&CreateInfo,nullptr,
                                                            &I_SwapChainImageViews[Index]);
        if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
            
            //  If we fail to create one image view after creating some OK, cleanup the ones we
            //  did create. (It seems unlikely, but it keeps things tidy.)
            
            LogVulkanError("Failed to create image views.","vkCreateImageView",Result);
            StatusOK = false;
            if (Index > 0) {
                for (int I = 0; I < Index; I++) {
                    vkDestroyImageView(I_LogicalDevice,I_SwapChainImageViews[I],nullptr);
                }
                I_SwapChainImageViews.clear();
            }
            break;
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                            C r e a t e  R e n d e r  P a s s
//
//  A graphics pipeline needs a render pass object, which describes all the processing performed
//  during the various rendering operations. This routine creates a simple render pass object,
//  and needs to be called before CreateGraphicsPipeline().
//
//  Parameters:
//     StatusOK        (bool&) A reference to an inherited status variable. If passed false,
//                     this routine returns immediately. If something goes wrong, the variable
//                     will be set false.
//
//  Pre-requisites:
//    Graphics must have been enabled, using EnableGraphics(), and the logical device must have
//    been created using CreateLogicalDevice().
//
//  Note:
//    A render pass can be quite complicated, but this sets up a very simple one based on the
//    example in vulkan-tutorial.com.

void KVVulkanFramework::CreateRenderPass(bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  A render pass object describes all the processing performed during the various rendering
    //  operations. This can involve a number of different subpasses, but we set up a simple
    //  render pass with only one subpass. This will be a graphics subpass, working with one
    //  straightforward colour attachment.
    
    VkAttachmentDescription ColourAttachment{};
    ColourAttachment.format = I_SwapChainImageFormat;
    ColourAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    ColourAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ColourAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ColourAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ColourAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ColourAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ColourAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ColourAttachmentRef{};
    ColourAttachmentRef.attachment = 0;
    ColourAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    //  Note that when we set up the RenderPassInfo structure below, we specify an array of colour
    //  attachments with just one element, which will be at index 0. This refers to the single
    //  image we assume the shader program will be using, having specified location = 0. Our
    //  single subpass refers to one colour attachment, and that too specifies attachment index 0.
    //  So the structure we have here effectively forces the fragment shader to specify location 0.
    //  I think I have that right, but this gets quite intricate!
    
    VkSubpassDescription Subpass{};
    Subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    Subpass.colorAttachmentCount = 1;
    Subpass.pColorAttachments = &ColourAttachmentRef;

    VkSubpassDependency Dependency{};
    Dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    Dependency.dstSubpass = 0;
    Dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    Dependency.srcAccessMask = 0;
    Dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    Dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo RenderPassInfo{};
    RenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    RenderPassInfo.attachmentCount = 1;
    RenderPassInfo.pAttachments = &ColourAttachment;
    RenderPassInfo.subpassCount = 1;
    RenderPassInfo.pSubpasses = &Subpass;
    RenderPassInfo.dependencyCount = 1;
    RenderPassInfo.pDependencies = &Dependency;

    VkResult Result;
    Result = vkCreateRenderPass(I_LogicalDevice,&RenderPassInfo,nullptr,&I_RenderPass);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError("Failed to create render pass.","vkCreateRenderPass",Result);
        StatusOK = false;
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                   C r e a t e  S h a d e r  M o d u l e  F r o m  F i l e
//
//  This routine creates a Vulkan shader module for use by a pipeline, given the name of a file
//  containing its SPIR-V semi-compiled code.
//
//  Parameters:
//     ShaderFilename (const std::string&) The name of file containing the SPIR-V code for a
//                    shader module to be run by the GPU.
//     ModuleHndlPtr  (VkShaderModule*) Receives the Vulkan handle for the newly created module.
//     StatusOK       (bool&) A reference to an inherited status variable. If passed false,
//                    this routine returns immediately. If something goes wrong, the variable
//                    will be set false.
//
//  Pre-requisites:
//    The logical device must have been created using CreateLogicalDevice(). The file name 
//    specified must be a string that can be passed to fopen(), and the file should contain valid
//    code.

void KVVulkanFramework::CreateShaderModuleFromFile(
        const std::string& ShaderFilename,VkShaderModule* ModuleHndlPtr,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    VkShaderModule ShaderModuleHndl = VK_NULL_HANDLE;
    uint32_t* ShaderCode = nullptr;
    
    //  Read the SPIR-V code from the file (this will be into an array allocated using new[])
    
    long LengthInBytes;
    ShaderCode = ReadSpirVFile(ShaderFilename,&LengthInBytes,StatusOK);
    
    //  Create the module from the code, and release the memory used for the code. Then record
    //  the handle for the module so it can be deleted later.
    
    ShaderModuleHndl = CreateShaderModule(ShaderCode,LengthInBytes,StatusOK);
    if (ShaderCode) delete[] ShaderCode;
    if (AllOK(StatusOK)) {
        I_ShaderModuleHndls.push_back(ShaderModuleHndl);
    } else {
        if (ShaderModuleHndl != VK_NULL_HANDLE) { vkDestroyShaderModule(I_LogicalDevice,ShaderModuleHndl,nullptr);
            ShaderModuleHndl = VK_NULL_HANDLE;
        }
    }
    *ModuleHndlPtr = ShaderModuleHndl;
}

//  ------------------------------------------------------------------------------------------------
//
//                         C r e a t e  G r a p h i c s  P i p e l i n e
//
//  This routine creates a relatively simple Vulkan pipeline that will run a GPU graphics program
//  using a vertex and a fragment shader, operating on a set of one or more data buffers that
//  contain, for example, vertex positions and colours) in a specified layout. At this point, the
//  sizes of the buffer(s) do not need to be known, but their type (uniform, storage, etc) and the
//  way they are accessed (local, shared, etc) must be known. The set of buffers used should be
//  passed to this routine as a vector of Framework buffer handles, as returned by the routine
//  SetBufferDetails(). The shader and fragment modules should have been created from
//  SPIR-V code, probably using CreateShaderModuleFromFile(). This routine creates both a layout
//  for the pipeline and then the pipeline itself, and returns Vulkan handles for both.
//
//  Parameters:
//     VertexShaderHndl (VkShaderModule) The vertex shader module to be run by the pipeline.
//     VertexStageName (const std::string&) The entry name of the routine in the vertex shader to
//                   be run by the pipeline (often this will be 'main', but it does not have to be).
//     FragmentShaderHndl (VkShaderModule) The vertex shader module to be run by the pipeline.
//     FragmentStageName (const std::string&) The entry name of the routine in the fragment shader
//                   to be run by the pipeline (again, often this will be 'main' but need not be).
//     VertexType    (const std::string&) A string giving the tolology used for the vertices - this
//                   can be one of "TRIANGLE_LIST", "TRIANGLE_STRIP", "LINE_LIST" or "LINE_STRIP".
//     BufferHandles (std::vector<KVVulkanFramework::KVBufferHandle>&) a vector containing the
//                   Framework handles for each buffer, as returned by SetBufferDetails().
//     PipelineLayoutHndlPtr (VkPipelineLayout*) Receives the opaque Vulkan handle for the
//                   created pipeline layout.
//     PipelineHndlPtr (VkPipeline*) Receives the opaque Vulkan handle for the created pipeline.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//     The buffer handles should have been created by calls to SetBufferDetails(), and if any
//     buffer contains vertex data, SetVertexBufferDetails() should also have been called to add
//     details of the buffer layout. The shader modules will usually have been created through
//     calls to CreateShaderModuleFromFile()

void KVVulkanFramework::CreateGraphicsPipeline(
    VkShaderModule VertexShaderHndl,const std::string& VertexStageName,
    VkShaderModule FragmentShaderHndl,const std::string& FragmentStageName,
    const std::string& VertexType,
    const std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
    VkPipelineLayout* PipelineLayoutHndlPtr,VkPipeline* PipelineHndlPtr,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;

    *PipelineLayoutHndlPtr = nullptr;
    *PipelineHndlPtr = nullptr;

    //  Build up the sets of binding and attribute descriptions associated with the buffers.
    
    std::vector<VkVertexInputBindingDescription> BindingDescriptions;
    std::vector<VkVertexInputAttributeDescription> AttributeDescriptions;
    for (KVVulkanFramework::KVBufferHandle BufferHandle : BufferHandles) {
        int Index = BufferIndexFromHandle(BufferHandle,StatusOK);
        if (AllOK(StatusOK)) {
            I_Debug.Log ("Buffers","Adding binding description:");
            VkVertexInputBindingDescription BDescr = I_BufferDetails[Index].BindingDescr;
            I_Debug.Logf ("Buffers","Binding = %d, stride = %d, %s",BDescr.binding,BDescr.stride,
                          BDescr.inputRate == VK_VERTEX_INPUT_RATE_VERTEX? "Vertex" : "Instance");
            BindingDescriptions.push_back(I_BufferDetails[Index].BindingDescr);
            for (const VkVertexInputAttributeDescription& AttributeDescr : 
                                                   I_BufferDetails[Index].AttributeDescrs) {
                I_Debug.Log ("Buffers","Adding attribute description:");
                I_Debug.Logf ("Buffers","Location %d, binding %d, format %d, offset %d",
                        AttributeDescr.location,AttributeDescr.binding,AttributeDescr.format,
                        AttributeDescr.offset);
                AttributeDescriptions.push_back(AttributeDescr);
            }
        }
    }
    
    //  See what sort of vertices we're being passed.
    
    VkPrimitiveTopology VertexTopology;
    if (AllOK(StatusOK)) {
        if (VertexType == "TRIANGLE_LIST") VertexTopology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        else if (VertexType == "TRIANGLE_STRIP") VertexTopology = 
                                                           VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        else if (VertexType == "LINE_LIST") VertexTopology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        else if (VertexType == "LINE_STRIP") VertexTopology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        else {
           LogError ("Unrecognised vertex type '%s' for graphics pipeline.",VertexType.c_str());
           StatusOK = false;
        }
    }
    
    if (AllOK(StatusOK)) {
        
        //  This sequence basically follows that shown in vulkan-tutorial.com. First, a lot
        //  of fiddly setup, mostly setting sensible defauts for all the many pipeline options.
        
        VkPipelineShaderStageCreateInfo VertexShaderStageInfo{};
        VertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        VertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        VertexShaderStageInfo.module = VertexShaderHndl;
        VertexShaderStageInfo.pName = VertexStageName.c_str();

        VkPipelineShaderStageCreateInfo FragmentShaderStageInfo{};
        FragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        FragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        FragmentShaderStageInfo.module = FragmentShaderHndl;
        FragmentShaderStageInfo.pName = FragmentStageName.c_str();

        VkPipelineShaderStageCreateInfo ShaderStages[] =
                                            {VertexShaderStageInfo,FragmentShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo VertexInputInfo{};
        VertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VertexInputInfo.vertexBindingDescriptionCount =
                                       static_cast<uint32_t>(BindingDescriptions.size());
        VertexInputInfo.vertexAttributeDescriptionCount =
                                       static_cast<uint32_t>(AttributeDescriptions.size());
        I_Debug.Logf ("Progress",
            "Graphics pipeline set up with %d binding descriptions, %d attribute descriptions\n",
                              int(BindingDescriptions.size()),int(AttributeDescriptions.size()));
        VertexInputInfo.pVertexBindingDescriptions = BindingDescriptions.data();
        VertexInputInfo.pVertexAttributeDescriptions = AttributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo InputAssembly{};
        InputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        InputAssembly.topology = VertexTopology;
        InputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo ViewportState{};
        ViewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        ViewportState.viewportCount = 1;
        ViewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo Rasterizer{};
        Rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        Rasterizer.depthClampEnable = VK_FALSE;
        Rasterizer.rasterizerDiscardEnable = VK_FALSE;
        Rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        Rasterizer.lineWidth = 1.0f;
        Rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        Rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        Rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo Multisampling{};
        Multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        Multisampling.sampleShadingEnable = VK_FALSE;
        Multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState ColourBlendAttachment{};
        ColourBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        ColourBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo ColourBlending{};
        ColourBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        ColourBlending.logicOpEnable = VK_FALSE;
        ColourBlending.logicOp = VK_LOGIC_OP_COPY;
        ColourBlending.attachmentCount = 1;
        ColourBlending.pAttachments = &ColourBlendAttachment;
        ColourBlending.blendConstants[0] = 0.0f;
        ColourBlending.blendConstants[1] = 0.0f;
        ColourBlending.blendConstants[2] = 0.0f;
        ColourBlending.blendConstants[3] = 0.0f;

        std::vector<VkDynamicState> DynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo DynamicState{};
        DynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        DynamicState.dynamicStateCount = static_cast<uint32_t>(DynamicStates.size());
        DynamicState.pDynamicStates = DynamicStates.data();

        //  Finally create the pipeline layout
        
        VkPipelineLayoutCreateInfo PipelineLayoutInfo{};
        PipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        PipelineLayoutInfo.setLayoutCount = 0;
        PipelineLayoutInfo.pushConstantRangeCount = 0;

        VkResult Result;
        Result = vkCreatePipelineLayout(I_LogicalDevice,&PipelineLayoutInfo,nullptr,
                                                                   PipelineLayoutHndlPtr);
        if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
            LogVulkanError("Failed to create pipeline layout.","vkCreatePipelineLayout",Result);
            StatusOK = false;
        } else {
            
            //  And now the pipeline itself.
            
            VkGraphicsPipelineCreateInfo PipelineInfo{};
            PipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            PipelineInfo.stageCount = 2;
            PipelineInfo.pStages = ShaderStages;
            PipelineInfo.pVertexInputState = &VertexInputInfo;
            PipelineInfo.pInputAssemblyState = &InputAssembly;
            PipelineInfo.pViewportState = &ViewportState;
            PipelineInfo.pRasterizationState = &Rasterizer;
            PipelineInfo.pMultisampleState = &Multisampling;
            PipelineInfo.pColorBlendState = &ColourBlending;
            PipelineInfo.pDynamicState = &DynamicState;
            PipelineInfo.layout = *PipelineLayoutHndlPtr;
            PipelineInfo.renderPass = I_RenderPass;
            PipelineInfo.subpass = 0;
            PipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

            Result = vkCreateGraphicsPipelines(
                    I_LogicalDevice,VK_NULL_HANDLE,1,&PipelineInfo,nullptr,PipelineHndlPtr);
            if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
                LogVulkanError("Failed to create graphics pipeline.","vkCreateGraphicsPipelines",
                                                                                          Result);
                StatusOK = false;
            }
        }
    }
    
    //  If the pipeline and its layout were created and bound successfully, record the details
    //  so they can be shut down properly at the end of the program. If things went wrong,
    //  release these if they were allocated before the problem was spotted.
    
    if (AllOK(StatusOK)) {
        T_PipelineDetails PipelineDetails;
        PipelineDetails.PipelineHndl = *PipelineHndlPtr;
        PipelineDetails.PipelineLayoutHndl = *PipelineLayoutHndlPtr;
        I_PipelineDetails.push_back(PipelineDetails);
    } else {
        if (*PipelineLayoutHndlPtr != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(I_LogicalDevice,*PipelineLayoutHndlPtr,nullptr);
        }
        if (*PipelineHndlPtr != VK_NULL_HANDLE) {
            vkDestroyPipeline(I_LogicalDevice,*PipelineHndlPtr,nullptr);
        }
    }


}

//  ------------------------------------------------------------------------------------------------
//
//                           C r e a t e  S y n c  O b j e c t s
//
//  This routine creates the synchronisation objects (sempahores and fences) to be used to control
//  execution of the simple graphics pipeline created using CreateGraphicsPipeline() and swap chain
//  created using CreateSwapChain(). We need two semaphores for each image in the chain, one for
//  acquisition of a new image, one for its rendering. We also need one fence for each command
//  buffer (again, one for each image) to make sure we don't attempt to reuse the command buffer
//  while it is still in use.
//
//  Parameters:
//     ImageCount    (int) The number of images in question, ie the number of each type of
//                   sync object required.
//     ImageSemaphores (std::vector<VkSemaphore>&) A vector of Vulkan handles for semaphores to be
//                   used to indicate that an image has been acquired from the swapchain. It should
//                   be passed empty, and will be returned containing the Vulkan handles for each
//                   of the created semaphores.
//     RenderSemaphores (std::vector<VkSemaphore>&) A vector of Vulkan handles for semaphores to be
//                   used to indicate that an image has been rendered and is ready for display. It
//                   should be passed empty, and will be returned containing the Vulkan handles for
//                   each of the created semaphores.
//     Fences        (std::vector<VkFence>&) A vector of Vulkan handles for fences to be used to
//                   indicate that the execution of the command buffer in use has completed and it
//                   can be re-recorded. It should be passed empty, and will be returned containing
//                   the Vulkan handles for each of the created fences.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device.
//
//  Note:
//     The calling program probably isn't going to need to do anything with the vectors of
//     semaphores and fences that are returned to it, and does not need to retain them. The
//     Framework keeps track independently of the various sync objects. (Since they are only
//     handles, you can have multiple copies of the handles for any one actual Vulkan object.)

void KVVulkanFramework::CreateSyncObjects(int ImageCount,std::vector<VkSemaphore>& ImageSemaphores,
        std::vector<VkSemaphore>& RenderSemaphores,std::vector<VkFence>& Fences,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    ImageSemaphores.resize(ImageCount);
    RenderSemaphores.resize(ImageCount);
    Fences.resize(ImageCount);

    VkSemaphoreCreateInfo SemaphoreInfo{};
    SemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo FenceInfo{};
    FenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    FenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    int ImageSemCount = 0, RenderSemCount = 0, FenceCount = 0;
    for (int Index = 0; Index < ImageCount; Index++) {
        VkResult CreateStatus =  vkCreateSemaphore(I_LogicalDevice,&SemaphoreInfo,nullptr,&ImageSemaphores[Index]);
        if (CreateStatus == VK_SUCCESS) ImageSemCount++;
        CreateStatus =  vkCreateSemaphore(I_LogicalDevice,&SemaphoreInfo,nullptr,&RenderSemaphores[Index]);
        if (CreateStatus == VK_SUCCESS) RenderSemCount++;
        CreateStatus = vkCreateFence(I_LogicalDevice,&FenceInfo,nullptr,&Fences[Index]);
        if (CreateStatus == VK_SUCCESS) FenceCount++;
        if ((ImageSemCount != Index + 1) || (RenderSemCount != Index + 1) ||
                                                             (FenceCount != Index + 1)) {
            LogError("Failed to create synchronization objects for image %d in swap chain.",
                                                                                  Index + 1);
            StatusOK = false;
            break;
        }
    }
    
    //  If anything went wrong, delete the objects we've created so far, if any.
    
    if (!AllOK(StatusOK)) {
        for (int Index = 0; Index < ImageSemCount; Index++) {
            vkDestroySemaphore(I_LogicalDevice,ImageSemaphores[Index],nullptr);
        }
        for (int Index = 0; Index < RenderSemCount; Index++) {
            vkDestroySemaphore(I_LogicalDevice,RenderSemaphores[Index],nullptr);
        }
        for (int Index = 0; Index < FenceCount; Index++) {
            vkDestroyFence(I_LogicalDevice,Fences[Index],nullptr);
        }
        ImageSemaphores.clear();
        RenderSemaphores.clear();
        Fences.clear();
    } else {
        for (int Index = 0; Index < ImageSemCount; Index++) {
            I_ImageSemaphoreHndls.push_back(ImageSemaphores[Index]);
            I_RenderSemaphoreHndls.push_back(RenderSemaphores[Index]);
            I_FenceHndls.push_back(Fences[Index]);
        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                           C r e a t e  F r a m e b u f f e r s
//
//  A swap chain needs framebuffers into which it can render its images. This routine creates
//  those necessary framebuffers, one for each image in the swap chain created by CreateSwapChain().
//
//  Parameters:
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     CreateLogicalDevice() must have been called to set up the selected GPU as a Vulkan device,
//     and the swap chain must have been created using CreateSwapChain().

void KVVulkanFramework::CreateFramebuffers(bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  We get the number of images handled by the swap chain, and set the size of our internal
    //  vector of framebuffer handles to match.
    
    size_t ImageCount = I_SwapChainImageViews.size();
    I_SwapChainFramebuffers.resize(ImageCount);

    for (int Index = 0; Index < int(ImageCount); Index++) {
        
        //  Since we're setting the attachment count to 1, we don't need to set up an array
        //  of their addresses - we can just the the address of the one image view directly.

        VkFramebufferCreateInfo FramebufferInfo{};
        FramebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        FramebufferInfo.renderPass = I_RenderPass;
        FramebufferInfo.attachmentCount = 1;
        FramebufferInfo.pAttachments = &(I_SwapChainImageViews[Index]);
        FramebufferInfo.width = I_SwapChainExtent.width;
        FramebufferInfo.height = I_SwapChainExtent.height;
        FramebufferInfo.layers = 1;

        VkResult Result;
        Result = vkCreateFramebuffer(I_LogicalDevice,&FramebufferInfo,nullptr,
                                                          &I_SwapChainFramebuffers[Index]);
        if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
            
            //  If we fail to create one frame buffer after creating some OK, cleanup the ones we
            //  did create. (It seems unlikely, but it keeps things tidy.)
            
            LogVulkanError("Failed to create frame buffer.","vkCreateFramebuffer",Result);
            StatusOK = false;
            if (Index > 0) {
                for (int I = 0; I < Index; I++) {
                    vkDestroyFramebuffer(I_LogicalDevice,I_SwapChainFramebuffers[I],nullptr);
                }
                I_SwapChainFramebuffers.clear();
            }
            break;

        }
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                                     D r a w  F r a m e
//
//  This routine actually draws a frame using the swap chain. It needs to be given the number of
//  the frame to be drawn, a command buffer to be used, the pipeline to be run, and the handles
//  for the various buffers involved. It also needs to be told the number of vertices that will
//  have to be drawn. This routine records the command buffer and presents it for execution. It
//  does not wait for the command buffer to be executed. DrawFrameMulti() supports a slightly
//  more complex drawing operation with multiple pipeline/buffer combinations.
//
//  Parameters:
//     CurrentFrame      (int) The index number of the frame to be drawn, starting from 0.
//     CommandBufferHndl (VkCommandBuffer) The Vulkan handle for the command buffer, as returned by
//                       either CreateCommandBuffers() or by CreateComputeCommandBuffer().
//     VertexCount       (int) The number of vertices to be drawn.
//     BufferHandles     (std::vector<KVVulkanFramework::KVBufferHandle>&) a vector containing the
//                       Framework handles for each buffer, as returned by SetBufferDetails().
//     PipelineHndl      (VkPipeline) The Vulkan handle for the pipeline.
//     StatusOK          (bool&) A reference to an inherited status variable. If passed false,
//                       this routine returns immediately. If something goes wrong, the variable
//                       will be set false.
//
//  Pre-requisites:
//     Pretty much everything must have been set up by now, since this is what all the graphics
//     set-up has been leading towards. The buffer handles should have been created by calls to
//     SetBufferDetails(). If any buffer contains vertex data, SetVertexBufferDetails() should
//     also have been called to add details of the buffer layout. The swap chain should have been
//     set up using CreateSwapChain(), the command buffer should have been created by by either
//     CreateCommandBuffers() or by CreateComputeCommandBuffer(). The buffers should contain the
//     data - generally the vertex colours and positions for what is to be drawn. The pipeline
//     should have been created using CreateGraphicsPipeline().

void KVVulkanFramework::DrawFrame (int CurrentFrame,VkCommandBuffer CommandBufferHndl,
                int VertexCount,const std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                                                            VkPipeline PipelineHndl,bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    //  DrawGraphicsFrame() was introduced when it became apparent that DrawFrame() was
    //  a little too limited, with its single pipeline that could only handle one type of
    //  data - eg triangles, or lines. It's now simpler to implement DrawFrame() as a call
    //  to DrawGraphicsFrame().
    
    int VertexCounts[1] = {VertexCount};
    VkPipeline Pipelines[1] = {PipelineHndl};
    const std::vector<KVVulkanFramework::KVBufferHandle> BufferSets[1] = {BufferHandles};
    
    DrawGraphicsFrame(CurrentFrame,CommandBufferHndl,1,VertexCounts,BufferSets,Pipelines,StatusOK);
}

//  ------------------------------------------------------------------------------------------------
//
//                             S e t  V e r t e x  B u f f e r  D e t a i l s
//
//  This routine adds information about a buffer used to contain vertex information for use
//  by a graphics pipeline. Such a buffer should be set up initially using a call to
//  SetBufferDetails(), and this call is used to add details about the layout of the vertex
//  information in a vertex buffer. There may be different types of information associated
//  with a vertex - it may, for example, have position information, and colour information.
//  These may be interleaved, or may be supplied in two separate vertex buffers. This is a
//  complex topic, and is described in more detail in the overview comments for this code.
//
//  Parameters:
//     BufferHndl    (KVBufferHandle) The Framework handles for the buffer, as returned by
//                   SetBufferDetails().
//     Stride        (long) The number of bytes in the buffer between successive elements of
//                   the buffer.
//     VertexRate    (bool) True if the buffer is indexed by vertex. This should be true unless
//                   the buffer is being used for instanced rendering, where multiple instances
//                   of the same object are rendered with different parameters.
//     NumberAttributes (long) The number of different attributes interleaved in the data. Each
//                   attribute (eg colour data, position coordinates, etc) has a format, a location
//                   and an offset, supplied in the Locations, FormatStrings and Offset array
//                   parameters.
//     Locations     (long[]) An array of NumberAttributes values, giving the 'locations' for the
//                   data in question - the location values used by the shader code to access the
//                   data.
//     FormatStrings (const char*[]) An array of NumberAttributes strings, giving the format used
//                   for the data for each vertex. This can be one of "float","vec2","vec3","vec4",
//                   "R32_SFLOAT","R32G32_SFLOAT","R32G32B32_SFLOAT" or "R32G32B32A32_SFLOAT",.
//                   "float","vec2","vec3" and "vec4" represent 1,2,3 or 4 single precision floats
//                   respectively and are used for general numeric values like positions.
//                   The others are used for colour values. The 'SFLOAT' indicates signed floating
//                   point, and 'R32G32', for example, means a 32-bit red value followed by
//                   a 32-bit blue value.
//     Offsets       (long[]) An array of NumberAttributes values, giving the offsets in bytes
//                   into the array at which the data starts. These offsets can be used to allow
//                   a single buffer to hold multiple types of data - since most GPUs only support
//                   a limited number of buffers this can be an advantage in some cases where a
//                   very large number of data sets are being used.
//     StatusOK      (bool&) A reference to an inherited status variable. If passed false,
//                   this routine returns immediately. If something goes wrong, the variable
//                   will be set false.
//
//  Pre-requisites:
//     The buffer should have been already set up using SetBufferDetails().
//
//  Notes:

void KVVulkanFramework::SetVertexBufferDetails(
        KVBufferHandle BufferHndl,long Stride,bool VertexRate,long NumberAttributes,
        long Locations[],const char* FormatStrings[],long Offsets[],bool& StatusOK)
{
    //  For a Vertex buffer, this associates the binding and attribute descriptions that the
    //  pipeline needs with the buffer handle. Passing the handle to CreateGraphicsPipeline()
    //  will allow the pipeline to be set up as needed to use the buffer.
    
    if (!AllOK(StatusOK)) return;
    
    int Index = BufferIndexFromHandle(BufferHndl,StatusOK);
    if (AllOK(StatusOK)) {
        
        //  Each buffer should be associated with just one binding, and even if it's being
        //  used in an interleaved form it should only have one stride value and one input rate.
        
        I_BufferDetails[Index].BindingDescr.binding = I_BufferDetails[Index].Binding;
        I_BufferDetails[Index].BindingDescr.stride = Stride;
        if (VertexRate) I_BufferDetails[Index].BindingDescr.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        else I_BufferDetails[Index].BindingDescr.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        
        //  If a buffer is interleaved, each interleaved group will be associated with a separate
        //  location in the shader code, and will need its own set of attributes.
        
        for (int I = 0; I < NumberAttributes; I++) {
            VkVertexInputAttributeDescription AttributeDescr;
            AttributeDescr.binding = I_BufferDetails[Index].Binding;
            AttributeDescr.location = Locations[I];
            AttributeDescr.offset = Offsets[I];
            std::string FormatString = FormatStrings[I];
            VkFormat Format;
            if (FormatString == "float") Format = VK_FORMAT_R32_SFLOAT;
            else if (FormatString == "vec3") Format = VK_FORMAT_R32G32B32_SFLOAT;
            else if (FormatString == "vec4") Format = VK_FORMAT_R32G32B32A32_SFLOAT;
            else if (FormatString == "vec2") Format = VK_FORMAT_R32G32_SFLOAT;
            else if (FormatString == "R32G32B32_SFLOAT") Format = VK_FORMAT_R32G32B32_SFLOAT;
            else if (FormatString == "R32G32B32A32_SFLOAT") Format = VK_FORMAT_R32G32B32A32_SFLOAT;
            else if (FormatString == "R32_SFLOAT") Format = VK_FORMAT_R32_SFLOAT;
            else if (FormatString == "R32G32_SFLOAT") Format = VK_FORMAT_R32G32_SFLOAT;
            else {
                LogError("Format string '%s' for vertex buffer unrecognised.",FormatString.c_str());
                StatusOK = false;
                break;
            }
            AttributeDescr.format = Format;
            I_BufferDetails[Index].AttributeDescrs.push_back(AttributeDescr);
        }
        if (!AllOK(StatusOK)) I_BufferDetails[Index].AttributeDescrs.clear();
    }
}

//  ------------------------------------------------------------------------------------------------
//
//                           D r a w  G r a p h i c s  F r a m e
//
//  This routine actually draws a frame using the swap chain. It needs to be given the number of
//  the frame to be drawn, a command buffer to be used, and a set of pipelines to be run, and
//  the handles for the various buffers involved - each pipeline has its own set of buffers and
//  its own number of vertices. This routine records the command buffer and presents it for
//  execution. It does not wait for the command buffer to be executed.
//
//  Parameters:
//     CurrentFrame      (int) The index number of the frame to be drawn, starting from 0.
//     CommandBufferHndl (VkCommandBuffer) The Vulkan handle for the command buffer, as returned by
//                       either CreateCommandBuffers() or by CreateComputeCommandBuffer().
//     Stages            (int) The number of pipeline/buffer combinations to be run.
//     VertexCounts      (int[]) The number of vertices to be drawn by each pipeline.
//     BufferSets        (std::vector<KVVulkanFramework::KVBufferHandle>[]) an array of vectors of
//                       buffer handles, each giving the Framework handles for each buffer used
//                       by the corresponding pipeline, as returned by SetBufferDetails().
//     PipelineHndls     (VkPipeline[]) The Vulkan handles for the pipelines.
//     StatusOK          (bool&) A reference to an inherited status variable. If passed false,
//                       this routine returns immediately. If something goes wrong, the variable
//                       will be set false.
//
//  Pre-requisites:
//     Pretty much everything must have been set up by now, since this is what all the graphics
//     set-up has been leading towards. The buffer handles should have been created by calls to
//     SetBufferDetails(). If any buffer contains vertex data, SetVertexBufferDetails() should
//     also have been called to add details of the buffer layout. The swap chain should have been
//     set up using CreateSwapChain(), the command buffer should have been created by by either
//     CreateCommandBuffers() or by CreateComputeCommandBuffer(). The buffers should contain the
//     data - generally the vertex colours and positions for what is to be drawn. The pipelines
//     should have been created using CreateGraphicsPipeline().
//
//  Note:
//     This is really just a version of DrawFrame(), but which takes arrays of Vertex counts,
//     pipelines and Buffer handle vectors instead of just a single pipeline/buffer combination.

void KVVulkanFramework::DrawGraphicsFrame (int CurrentFrame,VkCommandBuffer CommandBufferHndl,
   int Stages, int VertexCounts[],const std::vector<KVVulkanFramework::KVBufferHandle> BufferSets[],
                                                VkPipeline PipelineHndls[],bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    MsecTimer Timer; // DEBUG
    vkWaitForFences(I_LogicalDevice,1,&I_FenceHndls[CurrentFrame],VK_TRUE,UINT64_MAX);
    
    uint32_t ImageIndex;
    VkResult Result = vkAcquireNextImageKHR(I_LogicalDevice,I_SwapChain,UINT64_MAX,
                                I_ImageSemaphoreHndls[CurrentFrame],VK_NULL_HANDLE,&ImageIndex);
    
    if (Result == VK_ERROR_OUT_OF_DATE_KHR) {
        I_Debug.Log("Buffers","Frame buffer needs resizing");
        RecreateSwapChain(StatusOK);
        return;
    } else if (Result != VK_SUCCESS && Result != VK_SUBOPTIMAL_KHR) {
        LogVulkanError("Failed to acquire swap chain image","vkAcquireNextImageKHR",Result);
    }
    
    vkResetFences(I_LogicalDevice,1,&I_FenceHndls[CurrentFrame]);
    
    //  The command buffer will need re-recording - they're only good for one submission.
    
    RecordGraphicsCommandBuffer(CommandBufferHndl,Stages,PipelineHndls,ImageIndex,
                                                        VertexCounts,BufferSets,StatusOK);
    
    VkSubmitInfo SubmitInfo{};
    SubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
    VkSemaphore WaitSemaphores[] = {I_ImageSemaphoreHndls[CurrentFrame]};
    VkPipelineStageFlags WaitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    SubmitInfo.waitSemaphoreCount = 1;
    SubmitInfo.pWaitSemaphores = WaitSemaphores;
    SubmitInfo.pWaitDstStageMask = WaitStages;
    
    SubmitInfo.commandBufferCount = 1;
    SubmitInfo.pCommandBuffers = &CommandBufferHndl;
    
    VkSemaphore SignalSemaphores[] = {I_RenderSemaphoreHndls[CurrentFrame]};
    SubmitInfo.signalSemaphoreCount = 1;
    SubmitInfo.pSignalSemaphores = SignalSemaphores;
    
    VkQueue Queue;
    vkGetDeviceQueue(I_LogicalDevice,I_QueueFamilyIndex,0,&Queue);   //?? Is this the right place?
    
    Result = vkQueueSubmit(Queue,1,&SubmitInfo,I_FenceHndls[CurrentFrame]);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError("Failed to submit draw command buffer","vkQueueSubmit",Result);
        StatusOK = false;
    }
    
    VkPresentInfoKHR PresentInfo{};
    PresentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    
    PresentInfo.waitSemaphoreCount = 1;
    PresentInfo.pWaitSemaphores = SignalSemaphores;
    
    VkSwapchainKHR SwapChains[] = {I_SwapChain};
    PresentInfo.swapchainCount = 1;
    PresentInfo.pSwapchains = SwapChains;
    
    PresentInfo.pImageIndices = &ImageIndex;
    
    Result = vkQueuePresentKHR(Queue,&PresentInfo);
    
    if (Result == VK_ERROR_OUT_OF_DATE_KHR || Result == VK_SUBOPTIMAL_KHR) {
        RecreateSwapChain(StatusOK);
    } else if (Result != VK_SUCCESS) {
        LogVulkanError("Failed to present swap chain image","vkQueuePresentKHR",Result);
        StatusOK = false;
    }
    
}

//  ------------------------------------------------------------------------------------------------
//
//          R e c o r d  G r a p h i c s  C o m m a n d  B u f f e r   (internal routine)
//
//  This internal routine is used by DrawGraphicsFrame() to set up the command buffer that it
//  is using for the frame. This sets up the command buffer for a more complex drawing operation,
//  with multiple pipelines, for example one drawing triangles and another drawing lines. This
//  can handle any number of pipelines, each with their own set of buffers.
//
//  Parameters:
//     CommandBufferHndl (VkCommandBuffer) The Vulkan handle for the command buffer, as returned by
//                       either CreateCommandBuffers() or by CreateComputeCommandBuffer().
//     Stages            (int) The number of pipeline/buffer combinations to be run.
//     PipelineHndls     (VkPipeline[]) The Vulkan handles for the pipelines.
//     ImageNumber       (int) The image number to be used, as provided by vkAcquireNextImageKHR().
//     VertexCounts      (int[]) The number of vertices to be drawn by each pipeline.
//     BufferSets        (std::vector<KVVulkanFramework::KVBufferHandle>[]) an array of vectors of
//                       buffer handles, each giving the Framework handles for each buffer used
//                       by the corresponding pipeline, as returned by SetBufferDetails().
//     StatusOK          (bool&) A reference to an inherited status variable. If passed false,
//                       this routine returns immediately. If something goes wrong, the variable
//                       will be set false.
//
//  Pre-requisites:
//     This is expected to be called from DrawGraphicsFrame(), so has the same rather extensive
//     set of requirements.
//

void KVVulkanFramework::RecordGraphicsCommandBuffer(
        VkCommandBuffer CommandBufferHndl,int Stages,VkPipeline PipelineHndls[],int ImageNumber,
        int VertexCounts[], const std::vector<KVVulkanFramework::KVBufferHandle> BufferSets[],
        bool& StatusOK)
{
    if (!AllOK(StatusOK)) return;
    
    vkResetCommandBuffer(CommandBufferHndl,0);
    
    VkCommandBufferBeginInfo BeginInfo{};
    BeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    
    VkResult Result;
    Result = vkBeginCommandBuffer(CommandBufferHndl,&BeginInfo);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError ("Failed to begin recording command buffer.","vkBeginCommandBuffer",Result);
        StatusOK = false;
    }
    
    VkRenderPassBeginInfo RenderPassInfo{};
    RenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    RenderPassInfo.renderPass = I_RenderPass;
    RenderPassInfo.framebuffer = I_SwapChainFramebuffers[ImageNumber];
    RenderPassInfo.renderArea.offset = {0, 0};
    RenderPassInfo.renderArea.extent = I_SwapChainExtent;
    
    VkClearValue ClearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    RenderPassInfo.clearValueCount = 1;
    RenderPassInfo.pClearValues = &ClearColor;
    
    vkCmdBeginRenderPass(CommandBufferHndl,&RenderPassInfo,VK_SUBPASS_CONTENTS_INLINE);
    
    VkViewport Viewport{};
    Viewport.x = 0.0f;
    Viewport.y = 0.0f;
    Viewport.width = (float) I_SwapChainExtent.width;
    Viewport.height = (float) I_SwapChainExtent.height;
    Viewport.minDepth = 0.0f;
    Viewport.maxDepth = 1.0f;

    VkRect2D Scissor{};
    Scissor.offset = {0, 0};
    Scissor.extent = I_SwapChainExtent;

    for (int Stage = 0; Stage < Stages; Stage++) {
        vkCmdBindPipeline(CommandBufferHndl,VK_PIPELINE_BIND_POINT_GRAPHICS,PipelineHndls[Stage]);
        
        vkCmdSetViewport(CommandBufferHndl,0,1,&Viewport);
        
        vkCmdSetScissor(CommandBufferHndl,0,1,&Scissor);
        
        //  We have to set up the binding for the various buffers to be used, so the shader code
        //  knows where to gets its data. The main complication is that these buffers have to be
        //  accessible by the GPU, so if we are using staged buffers (with one on the CPU and one
        //  on the GPU) we have to make sure we use the right one - the secondary buffer.
        
        const std::vector<KVVulkanFramework::KVBufferHandle> BufferHandles = BufferSets[Stage];
        for (KVVulkanFramework::KVBufferHandle BufferHandle : BufferHandles) {
            int Index = BufferIndexFromHandle(BufferHandle,StatusOK);
            if (AllOK(StatusOK)) {
                VkDeviceSize Offset = 0;
                const char* WhichBuffer = "main";
                VkBuffer* VulkanBufferPtr = &I_BufferDetails[Index].MainBufferHndl;
                if (I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_CPU ||
                    I_BufferDetails[Index].BufferAccess == ACCESS_STAGED_GPU ) {
                    VulkanBufferPtr = &I_BufferDetails[Index].SecondaryBufferHndl;
                    WhichBuffer = "secondary";
                }
                I_Debug.Logf ("Buffers",
                        "Stage %d, Binding %s VkBuffer %p to binding %ld, offset %lld",Stage,
                              WhichBuffer,*VulkanBufferPtr,I_BufferDetails[Index].Binding,Offset);
                vkCmdBindVertexBuffers(CommandBufferHndl,I_BufferDetails[Index].Binding,
                                       1,VulkanBufferPtr,&Offset);
            }
        }
        
        //  This is the important command - it tells the command buffer to actually draw things.
        
        vkCmdDraw(CommandBufferHndl,VertexCounts[Stage],1,0,0);
    }
    vkCmdEndRenderPass(CommandBufferHndl);
    
    Result = vkEndCommandBuffer(CommandBufferHndl);
    if (Result != VK_SUCCESS || !AllOK(StatusOK)) {
        LogVulkanError ("Failed to record command buffer.","vkEndCommandBuffer",Result);
        StatusOK = false;
    }
}

/* -------------------------------------------------------------------------------------------------
 
                              P r o g r a m m i n g  N o t e s
 
    o   I need to revisit the use of inherited status as opposed to throwing exceptions. But I
        do think inherited status makes it easier to provide useful diagnostics that give context
        to a problem - so long as people take the trouble to use it to do so. It would be possible
        to make StatusOK an internal variable that can be tested, but that has design issues too.
 
    o   The graphics interface needs looking at - the framework remembers semaphores and fences,
        but not command buffers. Is there usually a 1-1 correspondence between all three?
 
    o   Actually, the whole API needs going over once I've actually used it for more than one
        application.
 
    o   I think there'd be something to be said for separating the creation of the shader modules
        (and the reading of the spirv files) from the creation of the pipeline. I think I've
        done that now.
 
    o   Sizes in bytes (lots of SizeInBytes variables used) should be size_t instead of long.
 
    o   I think it's the case that the creation of the graphics pipeline only uses the layout
        and bindings of the vertex buffers. The actual buffers used can be replaced with others
        of the same attributes (but, for example, of different sizes) at the time the command
        buffers are recorded - ie much later, as part of DrawFrame(). In fact, the buffers
        don't need to have been created at all - just the attribute descriptions and binding
        descriptions.
 
    o   ResizeBuffer() has become a superset of CreateBuffer(), and maybe CreateBuffer() isn't
        needed any more. But I like the idea of distinguishing between then. Maybe CreateBuffer()
        should simply call ResizeBuffer()?
        
    o   Initial testing under Linux on my Framework laptop (an AMD 7840U chip with an integrated
        Radeon 780M GPU) showed the compute time for my Mandelbrot test program being quite 
        impressive, but the rendering time was really very slow (I was getting 2fps or so on
        a 1024 square image). It turned out that the bottleneck was getting the results from
        the GPU calculation back from the GPU so the CPU could do the histogram equalisation 
        and set the colours for the triangle vertices, and then actually writing to the
        colour buffer. Both the image and the colour buffer were set as host visible and host
        coherent. What made the difference was setting the host cached flag as well. This
        got the frame rate up to over 100fps! Setting the cached flag seemed to have no effect
        on my M2 laptop, which has unified memory. The effects of the cached flag seem ill-defined
        and it might be that on other systems the results may differ. (30/1/24: I've now added
        support for staged buffers but haven't yet tested them on Linux)
 
    o   Initial testing with staged buffers on OSX suggests that they make little difference,
        but slow things down, slightly for the compute part of the Mandelbrot test, somewhat more
        noticeably for the rendering part. This was on an Intel Mac mini, where I thought it might
        have helped, but evidently not.
 
    o   The code doesn't support having a STAGED_BOTH type of staged buffer, where a buffer can
        be filled on the CPU side, synched so the data is copied the corresponding GPU buffer,
        then modified on the GPU side and copied back so the CPU side buffer is back in synch.
        It should only require setting both the destination and source transfer bits for both
        of the pair of buffers, but will need a version of SyncBuffers() that can specify the
        direction of transfer.
 
    o   I don't think CreateImageViews() needs to be an external routine - it really might as
        well be called directly from CreateSwapChain(). In fact, the whole graphics basic
        sequence could well be combined into one simple external call. This needs a bit of
        consideration once the Framework has bene used for more than one graphics application.
 
    o   DrawGraphicsFrame() was added as something of a messy extension when I realised I needed
        to support more complex drawing operations with multiple sets of both lines and triangles.
        It is a complete superset of DrawFrame() - if you call it with Stages set to 1 it does
        exactly what DrawFrame() does - so having both looks a bit odd. Indeed, I've now implemented
        DrawFrame() as a call to DrawGraphicsFrame(). But I rather feel that what's wanted
        is to split DrawGraphicsFrame() into a number of smaller routines that could be used by
        a calling program in an even more flexible way. And I think the calling sequence for
        DrawGraphicsFrame() is messy, with its arrays of things, including arrays of vectors(!).
        So is the name, actually. It would be nice to rename it to just DrawFrame().
        It depends on what I think the future of this code (which was originally only cobbled
        together as a demonstration of using Vulkan) might be. If it has a future at all.
 */
