//
//                           K V  V u l k a n  F r a m e w o r k . h
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
//     on the excellent website www.vulkan-tutorials.com. I found that invaluable, and
//     recommend it to anyone looking to get into Vulkan.
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
//                    options other than just a list of triangles. KS.
//     30th Jan 2024. Added support for the use of 'staged' buffers, implemented using two
//                    buffers, one visible to the CPU and a local GPU buffer, explicitly
//                    synched using the new SyncBuffer() routine. KS.
//      4th Feb 2024. Added ListMemoryProperties(). KS.
//      9th Mar 2024. Added support for DeviceSupportsDouble(). KS.
//     11th Mar 2024. Added support for EnableValidation() and EnableValidationLevels(). KS.
//      3rd May 2024. Started to seriously flesh out the comments, particularly the details
//                    for individual variables. The StageName parameter for CreateComputePipeline()
//                    is now passed as a reference. KS.
//     10th Jun 2024. Use of DebugHandler added to control debug logging, AllOK() added.
//      2nd Aug 2024. RecreateSwapChain() is now private. KS.
//     13th Aug 2024. Added GetDebugOptions(). KS.
//      6th Sep 2024. Introduced DrawGraphicsFrame(). KS.
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

#ifndef __KVVulkanFramework__
#define __KVVulkanFramework__

#include <vulkan/vulkan.h>

#include <vector>
#include <string>
#include <string.h>

#include "DebugHandler.h"

class KVVulkanFramework {
public:
    //  Defines a useful constant - something to indicate an unallocated buffer.
    typedef long KVBufferHandle;
    static const KVBufferHandle KV_NULL_HANDLE = 0;
    
    //  Constructor and destructor.
    //  ---------------------------
    KVVulkanFramework(void);
    ~KVVulkanFramework();
    
    //  Debugging support.
    //  ------------------
    //  Returns a string listing the named levels supported by the inbuilt Debug handler.
    static std::string GetDebugOptions(void);
    //  Sets the sub-system name to be used by the inbuilt Debug handler.
    void SetDebugSystemName (const std::string& Name);
    //  Specifies which of the supported levels are to be enabled.
    void SetDebugLevels (const std::string& Levels);
    //  Enables or disables the Vulkan validation layers.
    void EnableValidation(bool Enable);
    //  Enables or disables specific levels of Vulkan validation.
    void EnableValidationLevels (bool EnableErrors,bool EnableWarnings,bool EnableInformation);
    
    //  Initial setup.
    //  --------------
    //  Enables graphics, if required, and supplies the windowing system surface to use.
    void EnableGraphics(VkSurfaceKHR SurfaceHndl,bool& StatusOK);
    //  Sets the size of the frame buffer used by the window in use. Can change with window size.
    void SetFrameBufferSize(int Width,int Height,bool& StatusOK);
    //  Specifies any required Vulkan instance extensions.
    void AddInstanceExtensions(const std::vector<const char*>& ExtensionNames,bool& StatusOK);
    //  Specifies any required Vulkan instance extensions relating to graphics support.
    void AddGraphicsExtensions(const std::vector<const char*>& ExtensionNames,bool& StatusOK);
    //  Create the 'Instance' used by the program to interact with Vulkan.
    void CreateVulkanInstance (bool& StatusOK);
    //  Locate a suitable GPU device to be used.
    void FindSuitableDevice (bool& StatusOK);
    //  Create the 'logical device' used to interact with the actual GPU being used.
    void CreateLogicalDevice (bool& StatusOK);
    //  Returns true if the selected GPU supports double precision floating point operations.
    bool DeviceSupportsDouble (void);
    //  Returns the Vulkan instance being used.
    VkInstance GetInstance(void);
    
    //  Closing down.
    //  -------------
    //  Close down graphics - must be done before closing window.
    void CleanupVulkanGraphics(void);
    //  Close down Vulkan.
    void CleanupVulkan(void);
    
    //  Buffer support.
    //  ---------------
    //  Set the details describing a GPU buffer and get a Framework handle for it.
    KVBufferHandle SetBufferDetails (long Binding,const std::string& Type,
                                              const std::string& Access,bool& StatusOK);
    //  Adds additional details for a graphics buffer holding vertex information.
    void SetVertexBufferDetails(KVBufferHandle BufferHndl,long Stride,bool VertexRate,
                      long NumberAttributes,long Locations[],const char* FormatStrings[],
                                                           long Offsets[],bool& StatusOK);
    //  Create the actual Vulkan buffer and associated memory given a Framework handle.
    void CreateBuffer (KVBufferHandle BufferHndl,long SizeInBytes,bool& StatusOK);
    //  Delete a buffer.
    void DeleteBuffer (KVBufferHandle BufferHndl,bool& StatusOK);
    //  True if CreateBuffer() has been called for a buffer.
    bool IsBufferCreated (KVBufferHandle BufferHndl,bool& StatusOK);
    //  Synchronises a staged buffer - and is a null operation for an unstaged buffer.
    void SyncBuffer(KVBufferHandle BufferHndl,VkCommandPool CommandPoolHndl,VkQueue QueueHndl,
                                                                               bool& StatusOK);
    //  Gets a pointer the CPU can use to access the data held in a buffer.
    void* MapBuffer(KVBufferHandle BufferHndl,long* SizeInBytes,bool& StatusOK);
    //  Close down the mapping for a buffer.
    void UnmapBuffer(KVBufferHandle BufferHndl,bool& StatusOK);
    //  Change the size of a buffer and its associated memory.
    void ResizeBuffer(KVBufferHandle BufferHndl,long NewSizeInBytes,bool& StatusOK);
    
    //  Creating shaders.
    //  -----------------
    //  Read a file containing shader code and create the shader.
    void CreateShaderModuleFromFile(const std::string& ShaderFilename,
                                    VkShaderModule* ModuleHndlPtr,bool& StatusOK);
    
    //  Descriptor sets.
    //  ----------------
    //  Create a pool that can be used to supply descriptors for a given group of buffers.
    void CreateVulkanDescriptorPool(std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                                    int MaxSets,VkDescriptorPool* PoolHndlPtr,bool& StatusOK);
    //  Create a layout that specifies the configuration of a set of buffers.
    void CreateVulkanDescriptorSetLayout(
                    std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                                         VkDescriptorSetLayout* SetLayoutHndlPtr,bool& StatusOK);
    //  Get a descriptor set matching a specified layout from a descriptor pool.
    void AllocateVulkanDescriptorSet(VkDescriptorSetLayout SetLayoutHandl,VkDescriptorPool PoolHndl,
                                                   VkDescriptorSet* SetHndlPtr, bool& StatusOK);
    //  Fill in a descriptor set with the details of a set of buffers.
    void SetupVulkanDescriptorSet(std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                                                        VkDescriptorSet SetHndl, bool& StatusOK);
    
    //  Setting up and running computation on the GPU.
    //  ----------------------------------------------
    //  Create a pool that can be used to supply command buffers for execution on the GPU.
    void CreateCommandPool(VkCommandPool* CommandPoolHndlPtr,bool& StatusOK);
    //  Get a command buffer from a pool. (Could be dropped for CreateCommandBuffers()).
    void CreateComputeCommandBuffer(
            VkCommandPool CommandPoolHndl, VkCommandBuffer* CommandBufferHndlPtr,bool& StatusOK);
    //  Get a number of command buffers from a pool (graphics chains need a number of such buffers)
    void CreateCommandBuffers(VkCommandPool CommandPoolHndl, int NumberBuffers,
                            std::vector<VkCommandBuffer>& CommandBuffers,bool& StatusOK);
    //  Create a compute pipeline using a given shader and a given buffer layout.
    void CreateComputePipeline(const std::string& ShaderFilename,const std::string& StageName,
                 VkDescriptorSetLayout* SetLayoutHndlPtr,VkPipelineLayout* PipelineLayoutHndlPtr,
                                                  VkPipeline* PipelineHndlPtr,bool& StatusOK);
    //  Set up a compute command buffer given a pipeline and a buffer descriptor set.
    void RecordComputeCommandBuffer(
        VkCommandBuffer CommandBufferHndl,VkPipeline PipelineHndl,
        VkPipelineLayout PipelineLayoutHndl,VkDescriptorSet* DescriptorSetHndlPtr,
                                            uint32_t WorkGroupCounts[3],bool& StatusOK);
    //  Get a queue to run a command buffer on the GPU.
    void GetDeviceQueue(VkQueue* QueueHndlPtr,bool& StatusOK);
    //  Run a command buffer and wait for it to complete.
    void RunCommandBuffer(VkQueue QueueHndl,VkCommandBuffer CommandBufferHndl,bool& StatusOK);
    
    //  Setting up and running graphics on the GPU.
    //  -------------------------------------------
    //  Create a chain of images for display on successive frames.
    uint32_t CreateSwapChain(uint32_t RequestedImages,bool& StatusOK);
    //  Sets up the various images in a swap chain.
    void CreateImageViews(bool& StatusOK);
    //  Create a simple 'render pass' descrining the processing by a graphics command buffer.
    void CreateRenderPass(bool& StatusOK);
    //  Creates the actual frame buffers used by the images in a swap chain.
    void CreateFramebuffers(bool& StatusOK);
    //  Create a graphics pipeline, given a shader, a pipeline and a set of buffers.
    void CreateGraphicsPipeline(
        VkShaderModule VertexShaderHndl,const std::string& VertexStageName,
        VkShaderModule FragmentShaderHndl,const std::string& FragmentStageName,
        const std::string& VertexType,
        const std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
        VkPipelineLayout* PipelineLayoutHndlPtr,VkPipeline* PipelineHndlPtr,bool& StatusOK);
    //  Create the semaphores needed to synchronise operation of a swap chain of images.
    void CreateSyncObjects(int ImageCount,std::vector<VkSemaphore>& ImageSemaphores,
        std::vector<VkSemaphore>& RenderSemaphores,std::vector<VkFence>& Fences,bool& StatusOK);
    //  Draw a frame using a single graphics pipeline. (Could use DrawGraphicsFrame() instead).
    void DrawFrame (int CurrentFrame,VkCommandBuffer CommandBufferHndl,int VertexCount,
                    const std::vector<KVVulkanFramework::KVBufferHandle>& BufferHandles,
                    VkPipeline PipelineHndl,bool& StatusOK);
    //  Draw a frame using a number of graphics pipelines and associated buffers.
    void DrawGraphicsFrame (int CurrentFrame,VkCommandBuffer CommandBufferHndl, int Stages,
        int VertexCounts[],const std::vector<KVVulkanFramework::KVBufferHandle> BufferSets[],
                              VkPipeline PipelineHndls[],bool& StatusOK);
private:
    //  The framework is mostly a fairly transparent interface to Vulkan, but it does try to
    //  make buffer access a little higher level. In particular, it tries to hide a lot of the
    //  detailed housekeeping needed for buffers, and in particular to simplify the use of
    //  staged buffers. A buffer may be one of:
    //  Local - visible only to the GPU. (Most efficient, but not accessible by the CPU)
    //  Shared - visible to both GPU and CPU, with the necessary coordination handled behind the
    //           scened by Vulkan. (Flexible, but potentially inefficient).
    //  Staged - actually involves two buffers, one visible only to the GPU and one only visible
    //           to the CPU. (Potentially more efficient than a shared buffer, especially with
    //           an integrated GPU with its own memory allocation, or a discrete GPU, but
    //           probably unnecessary and even inefficient with a GPU with unified memory.)
    //  With a staged buffer, the framework tries to hide the double nature of the buffers, so
    //  the application sees the CPU-visible buffer, but when the framework works with the GPU,
    //  it makes sure the GPU sees the GPU-visible buffer. The application has to choose when
    //  the two buffers are synched, using SyncBuffer(). SyncBuffer() does nothing for shared
    //  buffers, so an application can be written for staged buffers and then changed to use
    //  shared buffers just by changing the access code when the buffers are created.
    //
    //  Internally, a local buffer has only one vulkan buffer, local to the GPU and this is the
    //  main buffer. A shared buffer has only one vulkan buffer, visible to both CPU and GPU, and
    //  again this is the main buffer. A staged buffer has a main buffer, visible to the CPU, and
    //  also a secondary buffer, visible to the GPU. It can be used to setup data on the CPU and
    //  then synch this with the GPU ('staged_cpu') or to produce data on the GPU and then sync
    //  this with the CPU ('staged_gpu'). Really, you don't want to be trying to use one staged
    //  buffer to go both ways - CPU to GPU and then back GPU to CPU. You're probably better off
    //  with a shared buffer if you want to do that.
    //
    //  The application can ignore the double nature of a staged buffer, and treat the 'buffer'
    //  as presented by the framework as just a single buffer accessible by both, but which
    //  needs to have the SynchBuffer() routine called once the CPU or GPU has done whatever
    //  it needs to do to the buffer.
    
    typedef enum {TYPE_UNKNOWN,TYPE_UNIFORM,TYPE_STORAGE,TYPE_VERTEX} KVBufferType;
    typedef enum {ACCESS_UNKNOWN,ACCESS_LOCAL,ACCESS_SHARED,
                                ACCESS_STAGED_CPU,ACCESS_STAGED_GPU} KVBufferAccess;
    //  The framework uses a vector (I_BufferDetails) of structures of type T_BufferDetails to
    //  keep track of all the buffers currently in use.
    typedef struct T_BufferDetails {
        //  Indicates this entry is in use - ie it refers to an existing buffer.
        bool InUse;
        //  Framework code for the buffer type (uniform, vertex, etc)
        KVBufferType BufferType;
        //  Framework code for the buffer access method (shared, local, staged, etc)
        KVBufferAccess BufferAccess;
        //  Handle used by the application to refer to this buffer
        KVBufferHandle Handle;
        //  Binding is specified by the application and has to match the shader setup.
        long Binding;
        //  The number of bytes of the buffer currently in use - may be less than those allocated.
        long SizeInBytes;
        //  The number of bytes actually allocated. (May allow for resizing without reallocation.)
        long MemorySizeInBytes;
        //  The address the CPU can use to access a buffer visible to the CPU.
        void* MappedAddress;
        //  The Vulkan handle for the main vulkan buffer.
        VkBuffer MainBufferHndl;
        //  The Vulkan handle for the main buffer memory.
        VkDeviceMemory MainBufferMemoryHndl;
        //  Usage flags for the main vulkan buffer.
        VkBufferUsageFlags MainUsageFlags;
        //  Property flags for the main buffer
        VkMemoryPropertyFlags MainPropertyFlags;
        //  The Vulkan handle for the secondary vulkan buffer (used for staged buffers)
        VkBuffer SecondaryBufferHndl;
        //  The Vulkan handle for the secondary vulkan buffer (used for staged buffers)
        VkDeviceMemory SecondaryBufferMemoryHndl;
        //  Usage flags for the secondary vulkan buffer (used for staged buffers)
        VkBufferUsageFlags SecondaryUsageFlags;
        //  Property flags for the secondary buffer (used for staged buffers)
        VkMemoryPropertyFlags SecondaryPropertyFlags;
        //  Binding description for the buffer as used by a graphics pipeline.
        VkVertexInputBindingDescription BindingDescr;
        //  Attribute description for the buffer as used by a graphics pipeline.
        std::vector<VkVertexInputAttributeDescription> AttributeDescrs;
    } BufferDetails;
    //  Internally the framework keeps track of any pipelines that it sets up in a vector
    //  (I_PipelineDetails) of structures of type T_PipelineDetails. It needs to keep a
    //  note of the handle for each pipeline (as returned by Vulkan) and the handle for the
    //  layout used by that pipeline. This is mainly to allow these to be deleted cleanly
    //  when the framework closes down.
    typedef struct T_PipelineDetails {
        VkPipeline PipelineHndl;              // Handle to the pipeline.
        VkPipelineLayout PipelineLayoutHndl;  // Handle to the layout it uses.
    } PipelineDetails;

    //  Error logging and handling
    //  This is the message callback set up when Vulkan validation is enabled.
    static VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT MessageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT MessageType,
        const VkDebugUtilsMessengerCallbackDataEXT* CallbackData,
        void* UserData);
    //  Logs a formatted error message
    void LogError (const char* const Format, ...);
    //  Logs an error message from the Vulkan validation layers, if enabled.
    void LogValidationError(const char* Message);
    //  Logs a warning message from the Vulkan validation layers, if enabled.
    void LogValidationWarning(const char* Message);
    //  Logs an informational message from the Vulkan validation layers, if enabled.
    void LogValidationInfo(const char* Message);
    //  Notes that a validation error has occurred.
    void SetValidationError(bool Set);
    //  Checks that no error has been flagged so far.
    bool AllOK (bool& StatusOK);
    //  Called when a Vulkan routine returns error status to log a suitable message.
    void LogVulkanError(const std::string& Text,const std::string& Routine, VkResult Result);
    //  Sets up the Vulkan validation error reporting.
    void SetupDebugMessengerInfo (VkDebugUtilsMessengerCreateInfoEXT& DebugInfo);
    //  Recreates a swap chain when the display window size changes.
    void RecreateSwapChain(bool& StatusOK);
    //  Closes down a swap chain.
    void CleanupSwapChain(void);
    //  Checks that a GPU supports the extensions required by the program.
    bool DeviceExtensionsOK(const std::vector<const char*>& GraphicsExtensions,
                            const std::vector<VkExtensionProperties>& DeviceExtensions);
    //  Checks that a GPU provides the swap chain support required by the program.
    bool SwapChainSupportAdequate(VkPhysicalDevice DeviceHndl,bool& StatusOK);
    //  Outputs some diagnostic information about a GPU device.
    void ShowDeviceDetails (VkPhysicalDevice DeviceHndl);
    //  Picks a swap chain surface format from those supported.
    VkSurfaceFormatKHR PickSwapSurfaceFormat(bool& StatusOK);
    //  Picks a presentation mode to be used by a swap chain from those supported.
    VkPresentModeKHR PickSwapPresentMode(bool& StatusOK);
    //  Returns the dimensions to be used for the swap chain images.
    VkExtent2D PickSwapExtent(const VkSurfaceCapabilitiesKHR& Capabilities,bool& StatusOK);
    //  Outputs some diagnostic information about the memory types supported by a GPU device.
    void ListMemoryProperties(const VkPhysicalDeviceMemoryProperties* Properties);
    //  Checks to see if a device supports the portability subset (ie not all Vulkan facilities).
    bool DeviceHasPortabilitySubset(std::vector<VkExtensionProperties>& Extensions);
    //  Lists the diagnostic layer names used by the Framework.
    const std::vector<const char*>& GetDiagnosticLayers(void);
    //  Attempts to classify the suitability of given GPU for the program in question.
    int RateDevice (VkPhysicalDevice DeviceHndl);
    //  Select a suitable device queue family and return its index number.
    uint32_t GetIndexForQueueFamilyToUse(bool UseGraphics,bool UseCompute,bool& StatusOK);
    //  Select a memory type from those supported and return its index.
    uint32_t GetMemoryTypeIndex(VkMemoryRequirements MemoryRequirements,
                                      VkMemoryPropertyFlags PropertyFlags,bool& StatusOK);
    //  Read a shader file in SPIR-V format into memory.
    uint32_t* ReadSpirVFile(const std::string& Filename,long* LengthInBytes, bool& StatusOK);
    //  Create a Vulkan shader module from SPIR-V code in memory.
    VkShaderModule CreateShaderModule(uint32_t* Code,long LengthInBytes,bool& StatusOK);
    //  Create a Vulkan buffer and its associated memory.
    void CreateVulkanBuffer(VkDeviceSize SizeInBytes,VkBufferUsageFlags UsageFlags,
                           VkMemoryPropertyFlags PropertyFlags,VkBuffer* BufferHndlPtr,
                                   VkDeviceMemory* BufferMemoryHndlPtr,bool& StatusOK);
    //  Record a graphics command buffer with a number of pipeline/buffer combinations.
    void RecordGraphicsCommandBuffer(
            VkCommandBuffer CommandBufferHndl,int Stages,VkPipeline PipelineHndls[],int ImageNumber,
            int VertexCounts[],const std::vector<KVVulkanFramework::KVBufferHandle> BufferSets[],
            bool& StatusOK);
    //  Given a buffer Handle, get its index into the internal vector of buffer details.
    int BufferIndexFromHandle (KVBufferHandle Handle,bool& StatusOK);

    //   Instance variables.
    DebugHandler I_Debug;
    VkInstance I_Instance;
    VkSurfaceKHR I_Surface;
    VkDebugUtilsMessengerEXT I_DebugMessenger;
    VkPhysicalDevice I_SelectedDevice;
    bool I_DeviceHasPortabilitySubset;
    bool I_GraphicsEnabled;
    bool I_DeviceSupportsDouble;
    bool I_EnableValidationErrors;
    bool I_EnableValidationWarnings;
    bool I_EnableValidationInformation;
    bool I_ValidationErrorFlagged;
    bool I_ErrorFlagged;
    uint32_t I_FrameBufferWidth;
    uint32_t I_FrameBufferHeight;
    VkExtent2D I_SwapChainExtent;
    int I_ImageCount;
    VkSwapchainKHR I_SwapChain;
    VkFormat I_SwapChainImageFormat;
    std::vector<VkImage> I_SwapChainImages;
    std::vector<VkImageView> I_SwapChainImageViews;
    std::vector<VkFramebuffer> I_SwapChainFramebuffers;
    VkRenderPass I_RenderPass;
    VkDevice I_LogicalDevice;
    bool I_DiagnosticsEnabled;
    uint32_t I_QueueFamilyIndex;  //  Note this assumes we only need one queue family.
    std::vector<const char*> I_RequiredInstanceExtensions;
    std::vector<const char*> I_RequiredGraphicsExtensions;
    std::vector<T_BufferDetails> I_BufferDetails;
    std::vector<T_PipelineDetails> I_PipelineDetails;
    std::vector<VkSemaphore> I_ImageSemaphoreHndls;
    std::vector<VkSemaphore> I_RenderSemaphoreHndls;
    std::vector<VkFence> I_FenceHndls;
    std::vector<VkDescriptorSetLayout> I_DescriptorSetLayoutHndls;
    std::vector<VkDescriptorPool> I_DescriptorPoolHndls;
    std::vector<VkCommandPool> I_CommandPoolHndls;
    std::vector<VkShaderModule> I_ShaderModuleHndls;
    static const std::string I_DebugOptions;
};

#endif

/*
                                 P r o g r a m m i n g  N o t e s
 
    o   Naming of variables like I_SemaphoreHndls and I_SwapChainImages is inconsistent.
        They are in fact all vectors of handles to Vulkan objects, and I think it helps
        to make this clear in the names. It should at least be consistent. This only applies
        to the naming of instance variables, so is entirely internal. The various routine
        parameters are all now named properly, I believe, with 'Hndl' used for any Vulkan
        handles, and 'HndlPtr' used for any pointers to Vulkan handles.
 
    o   CreateComputePipeline() still takes a shader name, while CreateGraphicsPipeline()
        now takes shader module handles.
 
    o   CreateComputePipeline() is passed a VkDescriptorSetLayout which is then used to create
        the pipeline layout to be used, which references the passed descriptor set layout.
        CreateGraphicsPipeline() doesn't (at present) use any descriptor sets (would it if
        it used uniform buffers, for example?), so isn't passed any such thing.
 
    o   RecordGraphicsCommandBuffer() is passed the buffer handles, which I think makes it easy to
        let it use different buffers even though the pipeline is the same. I can see people
        wanting to do this with compute command buffers as well. It is a work in progress...
 
    o   Notes on using buffers:
 
        All buffers have a 'binding' number associated with them.
 
        For GPU compute shaders, buffers are usually going to be uniform or storage buffers.
        Their binding numbers are set into the descriptors, one for each buffer, in the
        descriptor set that will be used when the command buffer to be used is recorded using
        vkCmdBindDescriptorSets(). The binding numbers are referenced explicitly in the shader
        code, for example in a line like:
        layout(std140, binding = 1) uniform buf1
        used to describe the contents of the buffer.
 
        For a vertex buffer, used in a vertex shader, the binding numbers for each buffer
        are bound to the buffers themselves when the command buffer to be used is recorded,
        using vkCmdBindVertexBuffers(). The shader code does not refer directly to buffers,
        but uses location values for data it processes. Location values are tied to binding
        numbers through the use of 'attribute descriptions' - structures that include a binding
        number and location number, together with an offset and a description of the contents
        of each buffer element (in the sense of 'this is a set of three floating point numbers
        and starts at such and such an offset from the start of the vertex data'.) Working with
        this are 'binding descriptors' - structures that contain a binding number together with
        some details of the size of the data element for each vertex (strictly, the 'stride'
        from the start of the data for one vertex to the start of the data to the next - which
        padding may cause to be larger than the size of the actual data) and a flag describing
        how to move through the data. The location value used by the shader is specified in
        the shader code in lines like:
        layout(location = 0) in vec2 inPosition;
        layout(location = 1) in vec3 inColor;
 
        Each buffer has one binding descriptior, but may have more than one attribute descriptor.
        Buffers with, for example, position and colour data interleaved would have one attribute
        descriptor for the position data and another for the colour values, giving different
        data formats and different offset values.
        The binding and attribute descriptors are passed to the pipeline through a structure
        of type VkPipelineVertexInputStateCreateInfo which is one of the elements specified
        when the pipeline layout is set up using vkCreatePipelineLayout(). The pipeline layout
        is in turn used to create the pipeline using vkCreateGraphicsPipelines().
 
        For example:
        As above, a vertex shader might expect to be passed XY positions and RGB colours for
        each vertex:
        layout(location = 0) in vec2 inPosition;
        layout(location = 1) in vec3 inColor;
        A vertex buffer might contain a set of vertex description structures giving both position
        and colour details for each vertex. In this case, there would be a single binding
        descriptor for that one buffer, set up in the C++ code as follows:
 
        struct Vertex {
            glm::vec2 pos;
            glm::vec3 color;
        };
        VkVertexInputBindingDescription bindingDescription;
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
 
        and two attribute descriptors, set up as:
 
        VkVertexInputAttributeDescription positionAttributes;
        positionAttributes.binding = 0;
        positionAttributes.location = 0;
        positionAttributes.format = VK_FORMAT_R32G32_SFLOAT;
        positionAttributes.offset = offsetof(Vertex, pos);
        VkVertexInputAttributeDescription colourAttributes;
        colourAttributes.binding = 0;
        colourAttributes.location = 1;
        colourAttributes.format = VK_FORMAT_R32G32B32_SFLOAT;
        colourAttributes.offset = offsetof(Vertex, pos);

        and the data in the buffer itself is just an array of Vertex structures.
 
        Note that the position data is just two XY float values (which is what glm::vec2 is),
        but the format code used is R32G32_SFLOAT, which is also two floating point values.
        Vulcan doesn't specify codes like X32Y32_SFLOAT so the equivalent colour codes have
        to be used instead.
 
        When the shader code references the inColor data, which is tied to location 1 in the
        shader code, the colourAttributes structure above ties location 1 to binding 0, and
        the actual buffer used will be tied to binding 0 by the call to vkCmdBindDescriptorSets()
        when the command buffer is recorded. The bindingDescription structure above says that
        the buffer with binding 0 is a set of items of size sizeof(Vertex) and the offset
        value in colourAttributes describes how the colour data is offset from the start
        of each item.
 
        Alternatively, two buffers might be used. One buffer holding all the position data
        as an array of glm::vec2 XY values, with a second holding all the colour data as an
        array of glm::vec3 RGB values. In this case, the shader code is unchanged, but there
        are now two binding descriptors, one for each buffer, and two attribute descriptors,
        this time each associated with a different buffer (ie with a different binding value).

        VkVertexInputBindingDescription positionBinding;
        positionBinding.binding = 0;
        positionBinding.stride = sizeof(glm::vec2);
        positionBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputBindingDescription colourBinding;
        colourBinding.binding = 1;
        colourBinding.stride = sizeof(glm::vec3);
        colourBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription positionAttributes;
        positionAttributes.binding = 0;
        positionAttributes.location = 0;
        positionAttributes.format = VK_FORMAT_R32G32_SFLOAT;
        positionAttributes.offset = 0;
        VkVertexInputAttributeDescription colourAttributes;
        colourAttributes.binding = 1;
        colourAttributes.location = 1;
        colourAttributes.format = VK_FORMAT_R32G32B32_SFLOAT;
        colourAttributes.offset = 0;
 
        Now, colourAttributes ties the location 1 colour data to the buffer with binding 1.
        The structure of each individual buffer is now simpler, and offsets in both cases
        are zero.
 
    o   Basic sequence for running a GPU computation. This runs backwards, starting with the
        actual computation, and then shows what you need to have done to get to that point.
 
        To run a computation..
 
        You call RunCommandBuffer()
        This needs a command buffer, properly setup, and a compute queue to run it in.
 
        To setup a command buffer, you call RecordComputeCommandBuffer().
        This needs the command buffer, a defined compute pipeline, a pipeline layout to
        describe the pipeline, a descriptor set that describes the memory buffers to be used,
        and a specification for the configuration of the GPU work groups to be used. You also
        need to have created the buffers themselves.
        The work group configuration is an array set up on the basis of the data dimensions.
        
        To get a compute queue you call GetDeviceQueue(). This needs a logical device to
        have been set up, but that happens early on - see below.

        To get the command buffer, you call CreateComputeCommandBuffer(). This needs a command
        pool, which can look after a number of command buffers.
 
        To get the command pool, you call CreateCommandPool(), (which needs a logical device).
 
        To setup the descriptor set, you call SetupVulkanDescriptorSet(). This needs the
        descriptor set itself, and a set of handles that refer to the various buffers involved.
        
        To get the descriptor set, you call AllocateVulkanDescriptorSet(). This needs a set
        layout, which depends on the buffers being used (we'll come to that) and a descriptor pool.
 
        To get the descriptor pool, you call CreateVulkanDescriptorPool(), which needs to
        know how many buffers are involved and their types, which it can get from the same set
        of buffer handles passed to SetupVulkanDescriptorSet().
 
        To create the compute pipeline, you call CreateComputePipeline(). This is passed the
        name of the file with the compiled shader code to be run, the name of the entry point
        in that code to use, and the same set layout needed by AllocateVulkanDescriptorSet(),
        to describe the buffers in use. This creates the pipeline, and also the pipeline layout
        that will eventually be used by RecordComputeCommandBuffer().
        
        To create that descriptor set layout, you call CreateVulkanDescriptorSetLayout(). This
        needs that same set of buffer handles that describe the buffers in use.
 
        To get that set of buffer handles, you need to call SetBufferDetails() for each of the
        memory buffers involved. Note that a lot of the pipeline setup just involves descriptions
        of the buffers, but it's not until just before you have to run the pipeline that you
        actually have to create the buffers and allocate actual memory for them. When you do
        actually want to do this, you call CreateBuffer() for each buffer. To make the buffer
        memory accessible to the CPU, you can call MapBuffer() (and there's an UnmapBuffer()).
 
        Almost all of these calls work with a logical Vulkan device, which the framework keeps
        track of internally, but has to be created using CreateLogicalDevice(). And this needs
        you to have picked an actual physical GPU to use. To do this, you call the framework's
        FindSuitableDevice(). This is a place where raw Vulkan gives you lots of options to
        look at all the GPUs available and pick the most suitable. Assuming most people only
        have one GPU, the framework just picks any GPU available that supports computation
        (and, optionally, graphics, which isn't covered in this section).
 
        To call FindSuitableDevice(), you have to have created a Vulkan 'instance', which is
        just an abstract entity that code can use to communicate with Vulkan. You create this
        with a call to CreateVulkanInstance(). For code that doesn't involve graphics, this is
        the first call to the framework that you need to make. (For graphics, interaction with
        the windowing system complicates this a little.)
 
        So, in actual order of calling, here's what you call:
 
        CreateVulkanInstance()
        FindSuitableDevice()
        CreateLogicalDevice()
        SetBufferDetails()  for each buffer
        CreateBuffer()      for each buffer (can be deferred)
        MapBuffer()         for each buffer (can also be deferred)
        CreateVulkanDescriptorSetLayout()
        CreateComputePipeline()
        CreateVulkanDescriptorPool()
        AllocateVulkanDescriptorSet()
        SetupVulkanDescriptorSet()
        CreateCommandPool()
        CreateComputeCommandBuffer()
        GetDeviceQueue()
        RecordComputeCommandBuffer()
        RunCommandBuffer()
 
        Having done all that, repeating the calculation is easy. The command buffer needs to be
        re-recorded, and then it can be re-run. So you just repeat those last two calls. Typically,
        one buffer will contain parameter values (usually a 'uniform' buffer, as these values
        apply to every instance of the calculation), and these can be changed before repeating
        the calculation.
 */
