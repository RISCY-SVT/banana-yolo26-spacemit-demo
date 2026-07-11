#include <onnxruntime_c_api.h>
#include <spacemit_ort_plugin.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace {

constexpr std::uint8_t kXorValue = 0x5a;
std::atomic<std::uint64_t> g_dispatch_calls {0};

class U8XorDispatch final : public spacemit::plugin::SpineCustomDispatch {
  public:
    bool Support(spacemit::plugin::SpinePluginContext& context) override {
        return context.GetInputTensor(0).GetDataType() == spacemit::plugin::DataType::UINT8;
    }

    void ReDispatch(spacemit::plugin::SpinePluginContext& context) override {
        elements_ = context.GetInputTensor(0).GetElementCount();
    }

    void operator()(spacemit::plugin::SpinePluginContext& context) override {
        const auto& input = context.GetInputTensor(0);
        auto& output = context.GetOutputTensor(0);
        const auto* source = static_cast<const std::uint8_t*>(input.GetData());
        auto* destination = static_cast<std::uint8_t*>(output.GetMutableData());
        const std::size_t elements = elements_ == 0 ? input.GetElementCount() : elements_;
        const std::size_t threads = static_cast<std::size_t>(std::max<std::ptrdiff_t>(1, context.GetThreadCount()));
        const std::size_t thread = static_cast<std::size_t>(std::max<std::ptrdiff_t>(0, context.GetThreadIndex()));
        const std::size_t chunk = (elements + threads - 1) / threads;
        const std::size_t begin = std::min(elements, thread * chunk);
        const std::size_t end = std::min(elements, begin + chunk);
        for (std::size_t index = begin; index < end; ++index) {
            destination[index] = static_cast<std::uint8_t>(source[index] ^ kXorValue);
        }
        g_dispatch_calls.fetch_add(1, std::memory_order_relaxed);
    }

  private:
    std::size_t elements_ = 0;
};

class U8XorOperator final : public spacemit::plugin::SpineCustomOperator {
  public:
    void Compile(spacemit::plugin::SpinePluginContext&) override {}

    void ReCompile(spacemit::plugin::SpinePluginContext& context) override {
        context.SetOutputTensorInfo(
            0, spacemit::plugin::DataType::UINT8, context.GetInputTensor(0).GetShape());
    }

    void KernelDispatch(spacemit::plugin::SpinePluginContext& context) override {
        auto dispatch = std::make_unique<U8XorDispatch>();
        if (dispatch->Support(context)) {
            dispatch->ReDispatch(context);
            context.SetCustomDispatch(dispatch.release());
        }
    }

    std::pair<bool, std::int64_t> CheckCapability(
        const spacemit::plugin::SpinePluginNode& node,
        const spacemit::plugin::SpinePluginGraphViewer&) const override {
        return {node.GetOpType() == "Stage46U8Xor" && node.GetDomain() == "spacemit.custom", 100};
    }
};

const char* ORT_API_CALL get_name(const OrtCustomOp*) { return "Stage46U8Xor"; }
const char* ORT_API_CALL get_ep_type(const OrtCustomOp*) { return nullptr; }
ONNXTensorElementDataType ORT_API_CALL get_input_type(const OrtCustomOp*, std::size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}
std::size_t ORT_API_CALL get_input_count(const OrtCustomOp*) { return 1; }
ONNXTensorElementDataType ORT_API_CALL get_output_type(const OrtCustomOp*, std::size_t) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
}
std::size_t ORT_API_CALL get_output_count(const OrtCustomOp*) { return 1; }
void* ORT_API_CALL create_kernel(const OrtCustomOp*, const OrtApi*, const OrtKernelInfo*) { return nullptr; }
void ORT_API_CALL compute_kernel(void*, OrtKernelContext*) {}
void ORT_API_CALL destroy_kernel(void*) {}
OrtCustomOpInputOutputCharacteristic ORT_API_CALL get_characteristic(const OrtCustomOp*, std::size_t) {
    return INPUT_OUTPUT_REQUIRED;
}
OrtMemType ORT_API_CALL get_input_memory_type(const OrtCustomOp*, std::size_t) { return OrtMemTypeDefault; }
int ORT_API_CALL get_variadic_min_arity(const OrtCustomOp*) { return 1; }
int ORT_API_CALL get_variadic_homogeneity(const OrtCustomOp*) { return 1; }
int ORT_API_CALL get_start_version(const OrtCustomOp*) { return 1; }
int ORT_API_CALL get_end_version(const OrtCustomOp*) { return INT_MAX; }
std::size_t ORT_API_CALL get_map(int**, int**) { return 0; }
void ORT_API_CALL release_map(int*, int*) {}

OrtCustomOp g_schema_stub = {
    ORT_API_VERSION,
    create_kernel,
    get_name,
    get_ep_type,
    get_input_type,
    get_input_count,
    get_output_type,
    get_output_count,
    compute_kernel,
    destroy_kernel,
    get_characteristic,
    get_characteristic,
    get_input_memory_type,
    get_variadic_min_arity,
    get_variadic_homogeneity,
    get_variadic_min_arity,
    get_variadic_homogeneity,
    nullptr,
    nullptr,
    nullptr,
    get_start_version,
    get_end_version,
    get_map,
    release_map,
    get_map,
    release_map,
};

}  // namespace

SPACEMIT_PLUGIN_DECLARE_VERSION(
    "Stage46U8XorPlugin", "1.0.0", "2.0.5", "Stage46 exact uint8 plugin proof", "Banana YOLO26 R&D")

extern "C" __attribute__((visibility("default"))) void SpacemitPluginInit(
    spacemit::plugin::PluginRegistrar* registrar) {
    registrar->AddOperator<U8XorOperator>("Stage46U8Xor", "spacemit.custom");
}

extern "C" __attribute__((visibility("default"))) OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (api == nullptr) return nullptr;
    OrtCustomOpDomain* domain = nullptr;
    OrtStatus* status = api->CreateCustomOpDomain("spacemit.custom", &domain);
    if (status != nullptr) return status;
    status = api->CustomOpDomain_Add(domain, &g_schema_stub);
    if (status != nullptr) return status;
    return api->AddCustomOpDomain(options, domain);
}

extern "C" __attribute__((visibility("default"))) std::uint64_t Stage46PluginGetDispatchCalls() {
    return g_dispatch_calls.load(std::memory_order_relaxed);
}
