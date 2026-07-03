#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace y26_k1x {

constexpr std::array<char, 8> kModelMagicV0{'Y', '2', '6', 'I', '8', 'K', '1', 'X'};
constexpr std::uint32_t kModelFormatVersionV0 = 0;

enum class StatusCode {
    kOk = 0,
    kInvalidArgument,
    kInvalidModel,
    kUnsupported,
    kNotImplemented,
};

struct EngineOptions {
    bool enable_ime = false;
    bool require_cluster0 = true;
    int threads = 1;
};

struct Detection {
    float x0 = 0.0F;
    float y0 = 0.0F;
    float x1 = 0.0F;
    float y1 = 0.0F;
    float score = 0.0F;
    float class_id = 0.0F;
};

struct ModelHeaderV0 {
    std::array<char, 8> magic = kModelMagicV0;
    std::uint32_t version = kModelFormatVersionV0;
    std::uint32_t header_size = 0;
    std::uint32_t endianness = 0x01020304U;
    std::uint32_t alignment = 64;
    std::uint32_t tensor_count = 0;
    std::uint32_t op_count = 0;
    std::uint32_t scale_descriptor_count = 0;
    std::uint64_t tensor_table_offset = 0;
    std::uint64_t op_table_offset = 0;
    std::uint64_t weight_blob_offset = 0;
    std::uint64_t scale_blob_offset = 0;
    std::uint64_t string_table_offset = 0;
    std::uint64_t checksum = 0;
    std::uint32_t model_contract_id = 0;
    std::uint32_t quantization_profile_id = 0;
};

struct ScaleDescriptorV0 {
    std::uint32_t dtype = 0;
    std::uint32_t granularity = 0;
    std::int32_t axis = -1;
    std::uint32_t count = 0;
    std::uint32_t alignment = 4;
    std::uint64_t blob_offset = 0;
};

struct RequantParams {
    float effective_scale = 1.0F;
    std::int32_t output_zero_point = 0;
    std::int32_t qmin = -128;
    std::int32_t qmax = 127;
};

StatusCode validate_model_header(const ModelHeaderV0& header);
std::string_view status_message(StatusCode code);

std::int8_t requantize_s32_to_s8(std::int32_t accumulator, const RequantParams& params);

void pack_a_row_major_4x8(const std::int8_t* src, std::ptrdiff_t row_stride, std::span<std::int8_t, 32> dst);
void pack_b_transposed_4x8(const std::int8_t* src, std::ptrdiff_t row_stride, std::span<std::int8_t, 32> dst);

void vmadot_scalar_4x4x8(const std::int8_t* a_panel,
                         const std::int8_t* b_transposed_panel,
                         std::int32_t* c_tile,
                         std::ptrdiff_t c_stride,
                         bool accumulate);

class Engine {
public:
    explicit Engine(EngineOptions options);

    [[nodiscard]] StatusCode load_model(std::span<const std::byte> model_blob);
    [[nodiscard]] StatusCode infer(std::span<const std::int8_t> input_nchw,
                                   std::span<Detection> output_detections);

private:
    EngineOptions options_;
    bool model_loaded_ = false;
};

}  // namespace y26_k1x
