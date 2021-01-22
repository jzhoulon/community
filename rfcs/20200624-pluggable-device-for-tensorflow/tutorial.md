#

# Contents

1. [Introduction](#Introduction)

2. [Getting started](#Getting started)

   1. [Plugin Implementation](#_Toc62041058)

   2. [Device Runtime](#_Toc62041059)

[Kernel/Op 5](#_Toc62041060)

[Graph optimization 14](#_Toc62041061)

[Plugin build 14](#_Toc62041062)

[Plugin installation 14](#_Toc62041063)

[Plugin Running 14](#_Toc62041064)

# **Tutorial: How to create a TensorFlow plugin**

# Introduction

This tutorial is intended for developers who wish to extend TensorFlow to support a new device for the current TensorFlow stack through Modular TensorFlow. Plugin provides a decoupled way to add a new device to TensorFlow and has benefits:

- Simpler process: Does not have to add a new build toolchain to TensorFlow
- Faster time-to-solution: Does not need code review from the TensorFlow team.
- Lower maintenance efforts: Only C-API-related changes could break the integration. Unrelated TensorFlow changes would not break the code.

The article describes how to implement, build, install and run the plugin. The plugin implementation section covers device runtime registration, kernel registration as well as graph optimizer registration.

Developers are also recommended to read the Modular TensorFlow design RFC to have a better understanding of the architecture.

- [Modular TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20190305-modular-tensorflow.md)
- [Kernel and Op Implementation and Registration API](https://github.com/tensorflow/community/blob/master/rfcs/20190814-kernel-and-op-registration.md)
- [StreamExecutor C API](https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api.md)
- [Adding Pluggable Device for TensorFlow](https://github.com/tensorflow/community/blob/master/rfcs/20200624-pluggable-device-for-tensorflow.md)
- [Modular TensorFlow Graph C API](https://github.com/tensorflow/community/blob/master/rfcs/20201027-modular-tensorflow-graph-c-api.md)

The build environment in this tutorial is based on Linux, however, it is also expected to work on other OS(Windows, MacOS, and etc).

# Getting started

In this section, you will learn how to implement, build, install, and run a plugin.

## Plugin Implementation

Modular TensorFlow provides a set of C API as an ABI-stable way to register a custom device runtime, kernels/ops and graph optimizer. This will simplify the distribution of plugins and allow plugin authors to distribute binary artifacts without necessarily publishing plugin source code.

![](RackMultipart20210122-4-xmi1k9_html_37f3c7c5a9bc4409.png)

We anticipate three basic functionalities within a device plug-in module: device runtime, kernel/op, graph optimizer.

### Device Runtime

StreamExecutor is TensorFlow&#39;s main device manager, responsible for work execution and memory management. It provides a set of methods (such as Memcpy) that can be customized for a particular device. Modular TensorFlow proposed a C API wrapper of a subset of methods in StreamExecutorInterface as an ABI-stable way to register a custom StreamExecutor platform. The API can be found in [tensorflow/c/experimental/stream\_executor/stream\_executor.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/experimental/stream_executor/stream_executor.h). Plugins need to include implementation of the interfaces declared in this file.

Here we will introduce how to register a device runtime through StreamExecutor C API. Before that, we will have some conventions:

- Struct defined in StreamExecutor C API: struct prefix indicates whether fields should be filled by the plugin or core implementation
  - SE\_: set/filled by core unless explicit marked otherwise.
  - SP\_: set/filled by plugin unless explicit marked otherwise.
- Struct with Plugin prefix: these are structs defined in plugin, plugin can choose whatever name/definition they want.
- Function with plugin\_ prefix: these are functions defined in plugin, plugin can choose whatever function name they want.

- **SE\_InitPlugin**

Plugins need to define SE\_InitPlugin function and populates SE\_PlatformRegistrationParams::SP\_Platform and SE\_PlatformRegistrationParams::SP\_PlatformFns. When this plugin is loaded by TF at runtime, SE\_InitPlugin method will be called and a new StreamExecutor platform will be registered by Core TensorFlow.

Example:

![](RackMultipart20210122-4-xmi1k9_html_afef03ee042fd451.gif)

As you may see in the example, plugin needs to populate the platform and platform\_fns.

- platform-\&gt;struct\_size: plugin needs to set it as SP\_PLATFORM\_STRUCT\_SIZE (defined in stream\_executor.h). This field is used for the StreamExecutor C API version check between Core TensorFlow and the plugin.
- platform-\&gt;type: This field allows plugin authors to register a new device type to the Core TensorFlow, this device type will be visible in front-end, such as tf.device(&quot;device type&quot;)
- platfom-\&gt;name: This filed allows plugin authors to register a new StreamExecutor platform name to the Core TensorFlow. This name should be a unique name, you can&#39;t choose a name like &quot;CUDA&quot;, &quot;ROCM&quot; which are first party platform name.
- platform-\&gt;visible\_device\_count: Core TensorFlow will query this number to decide how many physical devices are discovered by plugin&#39;s device runtime.
- platform\_fns-\&gt;create\_device: a callback for creating SP\_Device. plugin authors need to define function that populate the SP\_Device:

![](RackMultipart20210122-4-xmi1k9_html_5ea104d9f460a058.gif)

- platform\_fns-\&gt;destroy\_device: a callback for destroying SP\_Device. plugin authors need to define function that destroy the SP\_Device:

![](RackMultipart20210122-4-xmi1k9_html_d776be3f11ea5773.gif)

- platform\_fns-\&gt;create\_stream\_executor: a callback for creating SP\_StreamExecutor. plugin authors need to define function that populates SP\_StreamExecutor.

#include&quot;tensorflow/c/experimental/stream\_executor/stream\_executor.h&quot;

void plugin\_create\_stream\_executor(const SP\_Platform\* platform,

SE\_CreateStreamExecutorParams\* params,

TF\_Status\* const status) {

params-\&gt;stream\_executor-\&gt;struct\_size = SP\_STREAMEXECUTOR\_STRUCT\_SIZE;

params-\&gt;stream\_executor-\&gt;allocate = plugin\_allocate;

params-\&gt;stream\_executor-\&gt;deallocate = plugin\_deallocate;

params-\&gt;stream\_executor-\&gt;host\_memory\_allocate= plugin\_host\_memory\_allocate;

params-\&gt;stream\_executor-\&gt;host\_memory\_deallocate =

plugin\_host\_memory\_deallocate;

params-\&gt;stream\_executor-\&gt;get\_allocator\_stats = plugin\_get\_allocator\_stats;

params-\&gt;stream\_executor-\&gt;device\_memory\_usage = plugin\_device\_memory\_usage;

params-\&gt;stream\_executor-\&gt;create\_stream = plugin\_create\_stream;

params-\&gt;stream\_executor-\&gt;destroy\_stream = plugin\_destroy\_stream;

params-\&gt;stream\_executor-\&gt;create\_stream\_dependency =

plugin\_create\_stream\_dependency;

params-\&gt;stream\_executor-\&gt;get\_stream\_status = plugin\_get\_stream\_status;

params-\&gt;stream\_executor-\&gt;create\_event = plugin\_create\_event;

params-\&gt;stream\_executor-\&gt;destroy\_event = plugin\_destroy\_event;

params-\&gt;stream\_executor-\&gt;get\_event\_status = plugin\_get\_event\_status;

params-\&gt;stream\_executor-\&gt;record\_event = plugin\_record\_event;

params-\&gt;stream\_executor-\&gt;wait\_for\_event = plugin\_wait\_for\_event;

params-\&gt;stream\_executor-\&gt;create\_timer = plugin\_create\_timer;

params-\&gt;stream\_executor-\&gt;destroy\_timer = plugin\_destroy\_timer;

params-\&gt;stream\_executor-\&gt;start\_timer = plugin\_start\_timer;

params-\&gt;stream\_executor-\&gt;stop\_timer = plugin\_stop\_timer;

params-\&gt;stream\_executor-\&gt;memcpy\_dtoh = plugin\_memcpy\_dtoh;

params-\&gt;stream\_executor-\&gt;memcpy\_htod = plugin\_memcpy\_htod;

params-\&gt;stream\_executor-\&gt;memcpy\_dtod = plugin\_memcpy\_dtod;

……

}

plugin authors need to populate all fields in SP\_StreamExecutor. For example, register allocate function with _plugin\_malloc,_ it synchronously allocates &#39;size&#39; bytes on the underlying platform and returns &#39;SP\_DeviceMemoryBase&#39; representing that allocation.

![](RackMultipart20210122-4-xmi1k9_html_e65921c8605509c2.gif)

if the backend doesn&#39;t support this functionality, plugin authors can provide a dummy function

- platform\_fns-\&gt;destroy\_stream\_executor: clean up fields inside SP\_StreamExecutor that were allocated by the plugin. `stream_executor` itself should not be deleted here.

![](RackMultipart20210122-4-xmi1k9_html_27fc23ba2cfc9017.gif)

- platform\_fns-\&gt;create\_timer\_fns: creating SP\_Timer. Allocates timer resources on the underlying platform and initialized its internals, setting &#39;timer&#39; output variable. You can provide a dummy function if you don&#39;t need this.
- platform\_fns-\&gt;destroy\_timer\_fns: destroy SP\_Timer and deallocates timer resources on the underlying platform. You can provide a dummy implementation if you don&#39;t need this.
- platform\_fns-\&gt;destroy\_platform: clean up fields insides SP\_Platform that were allocated by the plugin. `platform` itself should not be deleted here.
- platform\_fns-\&gt;destroy\_platform\_fns: clean up fields insides SP\_PlatformFns.

### Kernel/Op

Modular TensorFlow provides a set of C APIs as the ABI-stable API for implementing kernels and ops. The intention is that existing kernels should be able to be ported to the new APIs with a minimum of reimplementation effort. The ops C API can be found in[tensorflow/c/ops.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/ops.h)and kernels C API can be found in[tensorflow/c/kernels.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/kernels.h). [tensorflow/c/tf\_tensor.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_tensor.h), [tensorflow/c/tf\_status.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_status.h).

Plugin authors need to defineTF\_InitKernelfunction (include Ops/Kernels registration). When the plugin is loaded by TF at runtime, TF\_InitKernelmethod will be called and new Ops/Kernels will be registered to Core TensorFlow.

- **Ops registration**

This section introduces how to register a new op to Core TensorFlow. In the C++ API, ops are registered at static initialization time using the REGISTER\_OP macro. For example:

![](RackMultipart20210122-4-xmi1k9_html_a2509a308191d9ca.gif)

The equivalent C API will be a series of functions that operate on TF\_OpDefinitionBuilder \*, a pointer to an opaque struct (i.e. a struct whose content is not made known to the plugin authors). The functions include, but not limited to:

- TF\_OpDefinitionBuilder\* TF\_NewOpDefinitionBuilder(const char\* op\_name): constructs and returns a new op registration builder for an op with the given name.
- void TF\_OpDefinitionBuilderAddAttr(TF\_OpDefinitionBuilder\* builder, const char\* attr): adds the given attribute to the builder(equivalent to Attr above).
- void TF\_OpDefinitionBuilderAddInput(TF\_OpDefinitionBuilder\* builder, const char\* input): adds the given input to the builder(equivalent to Input above).

Additional functions are provided for setting other properties of the operation (e.g. TF\_OpDefinitionBuilderSetIsCommutative).

Registration is then actually performed using theTF\_RegisterOpDefinitionfunction. This function populates a TF\_Status indicating whether registration was successful and frees the resources associated with the op definition builder.

The C equivalent of the bitcast op registration example above is shown below:

![](RackMultipart20210122-4-xmi1k9_html_4b019c871e9485f3.gif)

- **Ops shape inference**

A significant feature of certain ops is their ability to infer their output shapes. TensorFlow will invoke the registered shape inference function (if one is provided) when it needs to know the op&#39;s output shape. The registration function declaration is shown below:

![](RackMultipart20210122-4-xmi1k9_html_e733d468cbff6edb.gif)

A series of functions prefixed with TF\_ShapeInferenceContext is provided for the following purposes:

- Examining operator input shapes (TF\_ShapeInferenceContextGetInput).
- Creating and deleting shape and dimension handles (TF\_{New,Delete}ShapeHandle, TF\_{New,Delete}DimensionHandle).
- Manipulating shape and dimension handles (TF\_ShapeInferenceContextWithRank, TF\_ShapeInferenceContextDim).

In general, C analogues to the C++ methods intensorflow::shape\_inference (see [tensorflow/core/framework/shape\_inference.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/shape_inference.h)) will be provided.

- **Kernels implementation and registration.**

In this section, you will learn how to implement kernels and register them to Core TensorFlow. Here we will use Conv2D as the example.

- **Kernel Implementation**

The main classes for C++ kernel implementations are OpKernelConstruction(provided by TensorFlow to kernel&#39;s constructor) and OpKernelContext (provided to kernel&#39;s compute method). The analogues in the C API are TF\_OpKernelConstruction and TF\_OpKernelContext. The aim of the C API is providing functions for working with these structs that match, as closely as possible, the C++ API.

See below for an example of Conv2D kernel with the C++ API:

![](RackMultipart20210122-4-xmi1k9_html_944ba3a9929c3d6d.gif)

Above code shows a prototype of Conv2D C++ kernel, basically we can find that it has a constructor, a compute function and a parameter struct. The C equivalent Conv2D op can be:

![](RackMultipart20210122-4-xmi1k9_html_be08a708c9405dd1.gif)

Usually, plugin authors need to provide three functions: a creation function, a compute function and a deletion function. Compute function is a must, creation function and deletion functions are optional but if a creation is provided that causes memory allocation, a deletion function that frees the memory should also be provided, otherwise a leak will occur.

- **Creation function(optional)**: responsible for creating a kernel, allocating private resource (such as memory), and storing attributions (if it has) retrieved from TF\_OpKernelConstruction to the kernel. Core TensorFlow will call this function when it needs to instantiate the kernel. The TF\_OpKernelConstruction pointer is owned by TensorFlow and will be deleted once the creation function returns.
- **Compute function** : responsible for retrieving inputs and compute stream and produce outputs. Core TensorFlow will call this function when need to perform a computation with this kernel.
- **Destroy function(optional)**: responsible for destroying the kernel and free the resource allocated in the creation function. When TensorFlow no longer needs the kernel, it will call this function if one is provided. This function will retrieve the pointer returned in creation function or nullptr if no creation function was provided.

Here we will show how to use kernel C APIs to implement these functions:

- **Creation function**

In the C++ API, kernel&#39;s attributions are retrieved through GetAttr method in OpKernelConstruction.

![](RackMultipart20210122-4-xmi1k9_html_651399922d7227ed.gif)

Kernel C API provides a set of TF\_OpKernelConstruction\_GetAttrXX API to retrieve attributions from TF\_OpKernelConstruction. These APIs can be separated into four categories according to the attribution&#39;s container:

1. Scalar

TF\_OpKernelConstruction\_GetAttr(Type, Float,Int32, Int64, Bool…) interprets the named kernel construction attribute as scalar value and places it into \*val, float for example:

![](RackMultipart20210122-4-xmi1k9_html_955a7887bcb38dc5.gif)

1. Vector

TF\_OpKernelConstruction\_GetAttr(Type, Float, Int32, Int64, Bool…)List interprets the named kernel construction as a (Type, Float, Int32, Int64, Bool) array and places it into \*vals. vals must point to an array of length at lease &#39;max\_values&#39; (ideally set to the list\_size from TF\_OpKernelConstruction\_GetAttrSize()).

![](RackMultipart20210122-4-xmi1k9_html_dc19cc2118be1185.gif)

1. String

TF\_OpKernelConstruction\_GetAttrStringinterprets the named kernel construction attribute as string and places it into \*val. vals must point to an array of length at least &#39;max\_length&#39; (ideally set to total\_size from TF\_OpKernelConstruction\_GetAttrSize()).

![](RackMultipart20210122-4-xmi1k9_html_671ea557ac8ae45a.gif)

1. Vector of strings

TF\_OpKernelConstruction\_GetAttrStringListinterprets the named kernel construction attribute as string array and fills in &#39;vals&#39; and &#39;length&#39;, each of which must point to an array of length at least &#39;max\_values&#39;. The elements of values will point to addresses in &#39;storage&#39; which must be at least &#39;storage\_size&#39; bytes in length. Ideally, max\_values would be set to list\_size and &#39;storage&#39; would be at least total\_size, obtained fromTF\_OpKernelConstruction\_GetAttrSize(). ![](RackMultipart20210122-4-xmi1k9_html_fc3be352afd4a622.gif)

With these C APIs, we can retrieve Conv2D kernel&#39; attributions from TF\_OpKernelConstruction, see below for an example of creating a Conv2D kernel with C API. In this example, we use a series of C API for retrieving std::vector\&lt;int32\&gt;, std::vector\&lt;int64\&gt; and std::string attributions from TF\_OpKernelConstruction. We also use a series of C APIs for error handling (TF\_NewStatus, TF\_GetCode, TF\_DeleteStatus).

| void\* Conv2D\_Create(Conv2D\* kernel, TF\_OpKernelConstruction\* ctx) {auto\* kernel = new Conv2DOp;TF\_Status\* s = TF\_NewStatus();// C++: context-\&gt;GetAttr(&quot;dilations&quot;, &amp;params\_.dilations);int32\_t list\_size = 0;int32\_t total\_size = 0;TF\_OpKernelConstruction\_GetAttrSize(ctx, &quot;dilations&quot;, &amp;list\_size, &amp;total\_size, s);if (TF\_GetCode(s) == TF\_OK) {kernel-\&gt;dilations\_.resize(list\_size);TF\_OpKernelConstruction\_GetAttrInt32List(ctx, &quot;dilations&quot;, kernel-\&gt;dilations.data(), list\_size, s);}// C++: context-\&gt;GetAttr(&quot;strides&quot;, &amp;params\_.strides);if (TF\_GetCode(s) == TF\_OK) {list\_size = total\_size = 0;TF\_OpKernelConstruction\_GetAttrSize(ctx, &quot;strides&quot;, &amp;list\_size, &amp;total\_size, s);if (TF\_GetCode(s) == TF\_OK) { kernel-\&gt;strides\_.resize(list\_size); TF\_OpKernelConstruction\_GetAttrInt32List(ctx, &quot;strides&quot;, kernel-\&gt;strides.data(), list\_size, s);}}// C++: context-\&gt;GetAttr(&quot;padding&quot;, &amp;params\_.padding)if (TF\_GetCode(s) == TF\_OK) {list\_size = total\_size = 0;TF\_OpKernelConstruction\_GetAttrSize(ctx, &quot;padding&quot;, &amp;list\_size, &amp;total\_size, s);if (TF\_GetCode(s) == TF\_OK) { std::vector\&lt;char\&gt; val(total\_size);
 TF\_OpKernelConstruction\_GetAttrString(ctx, &quot;padding&quot;, val.data(), total\_size, s); std::string padding\_str = std::string(val.data(), total\_size);if (padding\_str == &quot;VALID&quot;) { kernel-\&gt;padding\_ = Padding::VALID; } elif(padding\_str == &quot;SAME&quot;) { kernel-\&gt;padding\_ = Padding::SAME; } elif(padding\_str == &quot;EXPLICIT&quot;) { kernel-\&gt;padding\_ = Padding::EXPLICIT; } }}// C++: context-\&gt;HasAttr(&quot;explicit\_padding&quot;)if (TF\_GetCode(s) == TF\_OK) {if (TF\_OpKernelConstruction\_HasAttr(ctx, &quot;explicit\_paddings&quot;, s)) { list\_size = total\_size = 0; TF\_OpKernelConstruction\_GetAttrSize(ctx, &quot;explicit\_paddings&quot;, &amp;list\_size, &amp;total\_size, s); kernel-\&gt;explicit\_paddings\_.resize(list\_size); TF\_OpKernelConstruction\_GetAttrInt64List(ctx, &quot;explicit\_paddings&quot;, kernel-\&gt;explicit\_paddings\_.data(), list\_size, s);}}if (TF\_GetCode(s) != TF\_OK) {TF\_OpKenrelConstruction\_Failure(ctx, s);delete kernel;kernel = nullptr;}TF\_DeleteStatus(s);return kernel;}
 |
| --- |

- **Compute function**

Basically, compute functions are able to retrieve their input tensors and provide output tensors. In the C++ API, the tensorflow::OpKernelContext::input and setoutput family of functions provide this functionality. The equivalent C calls will be TF\_GetInput and TF\_SetOutput family of functions. These C functions operate on TF\_Tensor. Besides, Kernel C API provides TF\_GetStream() for retrieving a computation stream, which allows kernels submitted to the hardware.

In the C++ API, OpKernelContext provides a set of functions to retrieve input tensors, shapes, stream as well as allocate output tensors or forward input to output tensor. A simple Conv2D compute function with C++ API can be like:

![](RackMultipart20210122-4-xmi1k9_html_5e4238275dc72e30.gif)

The equivalent OpKernelContext C functions provided by Modular TensorFlow are:

- TF\_GetInput(): retrieves the ith input from ctx.

- TF\_NumInputs(): returns the number of inputs available in ctx.
- TF\_NumOutputs(): returns the number of outputs to be placed in \*ctx by the kernel.
- TF\_SetOutput(): Sets the ith output of ctx to tensor.
- TF\_AllocateOutput(): allocates Tensor for output at given index.
- TF\_ForwardInputOrAllocateOutput(): tries to forward one of the inputs given in input\_indices to output[output\_index].
- TF\_AllocateTmp(): Allocates a temporary Tensor of the specified type and shape.
- TF\_GetStream(): returns the SP\_Stream available in ctx.

[tensorflow/c/tf\_tensor.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/tf_tensor.h) also provides some C API for manipulate TF\_Tensor:

- TF\_NewTensor(): return a new tensor that holds the bytes data[0, len-1];
- TF\_DeleteTensor(): destroy a tensor.
- TF\_TensorType(): return the type of a tensor element.
- TF\_NumDims(): return the number of dimensions that the tensor has.
- TF\_Dim(): return the length of the tensor in the &quot;dim\_index&quot; dimension.
- TF\_TensorByteSize(): return the size of the underlying data in bytes.
- TF\_TensorData(): return a pointer to the underlying data buffer.
- TF\_TensorElementCount(): returns the number of elements in the tensor.
- TF\_TensorBitcastFrom(): copy the internal data representation of `from` to `to`. `new_dims` and`num_new_dims` specify the new shape of the `to` tensor, `type` specifies itsdata type.
- TF\_TensorIsAligned(): return bool if this tensor is aligned.

**It should be noted that:** when you call functions that deal with TF\_Tensor on TF\_OpKernelContext, such as :TF\_GetInput, TF\_AllocateOutput, TF\_ForwardInputOrAllocateOutput, TF\_AllocateTmp,you are creating a new TF\_Tensor indeed, so you need to call TF\_DeleteTensor() to delete these TF\_Tensor manually at the exist of compute function, or you will get mem leak since when creating TF\_Tensor based on tensorflow::Tensor in OpKernelContext, it will increase the ref count in the C++ Tensor and the tensor will not be freed if these TF\_Tensors are not deleted.

With these C APIs, we can retrieve the input tensors and computation stream, do the computation and then produce the output tensors. See below for an example of computing a Conv2D kernel, you may also notice that when compute is finished, we need to delete the input, filter, output tensors manually.

![](RackMultipart20210122-4-xmi1k9_html_8239e45209a10cbf.gif)

- **Destroy function**

When Tensorflow no longer needs the kernel, it will call the destructor function in the OpKernel to release the resources created in the constructor. In plugin, we need to implement and register a destroy function to release those resources.

![](RackMultipart20210122-4-xmi1k9_html_deef33a1f445bfbd.gif)

- **Kernel Registration**

After implementing a kernel, we need to register this kernel to the Core TensorFlow so that it can be dispatched at runtime. Kernel registration with the C++ API is accomplished with the REGISTER\_KERNEL\_BUILD macro. This macro expands to code that relies on static initialization to register the provided kernel with the global kernel registry. See below for an example of registering a kernel with the C++ API:

![](RackMultipart20210122-4-xmi1k9_html_52abae7153e3a007.gif)

The equivalent C API provides a series of functions that operate on TF\_KernelBuilder, an opaque struct obtained with the TF\_NewKernelBuilder call. The kernel builder is registered with TensorFlow using the TF\_RegisterKenrelBuilder function. See below for an example of registering the conv kernel using the C API:

template\&lt;typenameT\&gt;

void RegisterConv2DKernel() {

TF\_Status\* s = TF\_NewStatus();

auto\* builder = TF\_NewKernelBuilder(&quot;Conv2D&quot;, &quot;MY\_DEVICE&quot;, &amp;Conv2D\_Create, &amp;Conv2D\_Compute\&lt;T\&gt;, &amp;Conv2D\_Destroy);

TF\_KernelBuilder\_TypeConstraint(builder, &quot;T&quot;, static\_cast\&lt;TF\_DataType\&gt;(DataTypeToEnum::v()), s)

if (TF\_GetCode(s) != TF\_OK()) {/\* handle errors\*/}

TF\_RegisterKernelBuilder(&quot;Conv2D&quot;, builder, s);

if (TF\_GetCode(s) != TF\_OK()) {/\* handle errors\*/}

TF\_DeleteStatus(s);

}

void TF\_InitKernel() {

RegisterConv2DKenrel\&lt;float\&gt;();

}

The registration function prototypes are provided below. Kernel authors must provide a compute function. creation and destroy functions are optional, but if a creation function is provided that causes memory allocation, a destroy function that frees the memory should be provided, otherwise a leak will occur.

TF\_KernelBuilder\* TF\_NewKernelBuilder(

constchar\* op\_name, constchar\* device\_name,

void\* (\*create\_func)(TF\_OpKernelConstruction\*),

void (\*compute\_func)(void\*, TF\_OpKernelContext\*),

void (\*delete\_func)(void\*));

void TF\_RegisterKernelBuilder(constchar\* name, TF\_KernelBuilder\* builder,

TF\_Status\* status);

### Graph optimization

To be added😊

## Plugin build

After implementing the plugin, we need to build it as a dynamic library. Build system is decided by plugin authors, you can choose bazel, cmake or other build systems, it is out of scope in this tutorial. To make things simple, we just use the gcc command here.

When building the plugin, we have two dependencies here:

1. We need to include those C API header files provided by Core TensorFlow.
2. The built plugin library needs to add dependency to \_pywrap\_tensorflow\_internal.so, which is built by Core TensorFlow. \_pywrap\_tensorflow\_internal.so contains those C API implementations. If you don&#39;t add this dependency, it will report &quot;undefined symbol&quot; error when loading the plugin library.

A recommended build procedure is:

Step1: install TF with:

| python3 -m venv venvsource venv/bin/activatepip install tf-nightly |
| --- |

Step2: Then build plugin with:

| g++ -std=c++11 -shared plugin.cc -o plugin.so -fPIC -Ivenv/lib/python3.8/site-packages/tensorflow/include -Lvenv/lib/python3.8/site-packages/tensorflow/python -l:\_pywrap\_tensorflow\_internal.so -O2 |
| --- |

With this procedure, you can always build the plugin with installed TensorFlow &#39;s compatible the C API.

**It should be noted** that you should pick up a unique name for plugin&#39;s dynamic library, otherwise you may get conflict with(overwrite) other installed plugins.

## Plugin installation

After building the plugin, you may want to distribute it through python package. One additional thing you need to do is to make the plugin&#39;s dynamic library (libplugin.so for example) be installed to the specified path (site-packages/tensorflow/python/tensorflow-plugins/) when user installing the package. Core TensorFlow will automatically iterate all the installed dynamic libraries in this path and try to load them.

## Plugin Running

After installing the plugin to the specified path (site-packages/tensorflow/python/tensorflow-plugins/). we can run the TensorFlow with plugin now.

Front-end usage of the plugged device has no difference with first party devices. Suppose you have installed a plugin registers a new device with &quot;MY\_DEVICE&quot; device type, you can:

1. List device

You can use _tf.config.list\_physical\_device()_ to query whether the MY\_DEVICE device is present on the host machine. If it is not found, then plugin may not be loaded correctly.

| \&gt;\&gt;tf.list\_physical\_devices()[PhysicalDevice(name=&#39;/physical\_device:CPU:0&#39;, device\_type=&#39;CPU&#39;), PhysicalDevice(name=&#39;/physical\_device:MY\_DEVICE:0&#39;, device\_type=MY\_DEVICE)] |
| --- |

1. tf.device

you can use with tf.device(&quot;my\_device:0&quot;) to specify the MY\_DEVICE device to be used for ops created/executed in a particular context.

| with tf.device(&quot;my\_device:0&quot;):# ops created here have the device my\_device:0 |
| --- |

1. automatic device placement

if you don&#39;t specify the device to be user for ops created/executed in a particular context, the op will be auto placed into the MY\_DEVICE device if the op for the MY\_DEVICE device is registered. Plugged device currently has the highest priority.
