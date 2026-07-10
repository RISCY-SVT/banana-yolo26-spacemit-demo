#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace y26_stage42 {

enum class ElementType {
    UINT8,
    FLOAT32,
};

enum class OrtOptimizationLevel {
    DISABLE,
    BASIC,
    EXTENDED,
    ALL,
};

enum class OrtExecutionMode {
    SEQUENTIAL,
    PARALLEL,
};

enum class ReferencePolicy {
    MATCHED_RUNTIME,
    FIXED_HOST_ORACLE,
};

struct TensorView {
    ElementType type = ElementType::UINT8;
    std::vector<std::int64_t> shape;
    const void* data = nullptr;
    std::size_t byte_size = 0;
};

struct NumericSummary {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double sum = 0.0;
    std::size_t nonfinite_count = 0;
};

struct Comparison {
    bool structurally_valid = false;
    std::string structural_error;
    std::size_t element_count = 0;
    std::size_t mismatch_count = 0;
    double mismatch_ratio = 0.0;
    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    double rmse = 0.0;
    std::optional<std::size_t> first_mismatch_index;
    double p50_abs_diff = 0.0;
    double p90_abs_diff = 0.0;
    double p95_abs_diff = 0.0;
    double p99_abs_diff = 0.0;
    double p999_abs_diff = 0.0;
    std::map<int, std::size_t> signed_difference_histogram;
    NumericSummary lhs;
    NumericSummary rhs;
    bool byte_equal = false;
};

std::size_t element_size(ElementType type);
std::size_t checked_element_count(const std::vector<std::int64_t>& shape);
std::size_t checked_byte_count(const std::vector<std::int64_t>& shape, ElementType type);
NumericSummary summarize_tensor(const TensorView& tensor);
Comparison compare_tensors(const TensorView& lhs, const TensorView& rhs);

OrtOptimizationLevel parse_ort_optimization_level(const std::string& value);
OrtExecutionMode parse_ort_execution_mode(const std::string& value);
ReferencePolicy parse_reference_policy(const std::string& value);
std::string serialize_reference_policy(ReferencePolicy policy);

}  // namespace y26_stage42
