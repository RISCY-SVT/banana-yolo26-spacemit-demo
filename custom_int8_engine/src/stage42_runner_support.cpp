#include "y26_stage42_runner_support.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace y26_stage42 {
namespace {

double percentile(const std::vector<double>& sorted, double quantile) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double position = quantile * static_cast<double>(sorted.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) {
        return sorted[lower];
    }
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

double value_at(const TensorView& tensor, std::size_t index) {
    if (tensor.type == ElementType::UINT8) {
        return static_cast<const std::uint8_t*>(tensor.data)[index];
    }
    return static_cast<const float*>(tensor.data)[index];
}

bool float_values_equal(float lhs, float rhs) {
    if (std::isnan(lhs) || std::isnan(rhs)) {
        return std::isnan(lhs) && std::isnan(rhs);
    }
    if (std::isinf(lhs) || std::isinf(rhs)) {
        return lhs == rhs;
    }
    return lhs == rhs;
}

}  // namespace

std::size_t element_size(ElementType type) {
    switch (type) {
        case ElementType::UINT8:
            return sizeof(std::uint8_t);
        case ElementType::FLOAT32:
            return sizeof(float);
    }
    throw std::invalid_argument("unsupported element type");
}

std::size_t checked_element_count(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) {
        return 1;
    }
    std::size_t count = 1;
    for (std::int64_t dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("tensor shape contains unresolved or non-positive dimension");
        }
        const std::size_t value = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("tensor element count overflow");
        }
        count *= value;
    }
    return count;
}

std::size_t checked_byte_count(const std::vector<std::int64_t>& shape, ElementType type) {
    const std::size_t count = checked_element_count(shape);
    const std::size_t bytes_per_element = element_size(type);
    if (count > std::numeric_limits<std::size_t>::max() / bytes_per_element) {
        throw std::overflow_error("tensor byte count overflow");
    }
    return count * bytes_per_element;
}

NumericSummary summarize_tensor(const TensorView& tensor) {
    const std::size_t count = checked_element_count(tensor.shape);
    if (tensor.data == nullptr || tensor.byte_size != checked_byte_count(tensor.shape, tensor.type)) {
        throw std::invalid_argument("tensor data or byte size does not match shape/type");
    }

    NumericSummary summary;
    summary.min = std::numeric_limits<double>::infinity();
    summary.max = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < count; ++i) {
        const double value = value_at(tensor, i);
        if (!std::isfinite(value)) {
            ++summary.nonfinite_count;
            continue;
        }
        summary.min = std::min(summary.min, value);
        summary.max = std::max(summary.max, value);
        summary.sum += value;
    }
    const std::size_t finite_count = count - summary.nonfinite_count;
    summary.mean = finite_count == 0 ? 0.0 : summary.sum / static_cast<double>(finite_count);
    if (finite_count == 0) {
        summary.min = 0.0;
        summary.max = 0.0;
    }
    return summary;
}

Comparison compare_tensors(const TensorView& lhs, const TensorView& rhs) {
    Comparison result;
    if (lhs.type != rhs.type) {
        result.structural_error = "dtype mismatch";
        return result;
    }
    if (lhs.shape != rhs.shape) {
        result.structural_error = "shape mismatch";
        return result;
    }

    try {
        result.element_count = checked_element_count(lhs.shape);
        const std::size_t expected_bytes = checked_byte_count(lhs.shape, lhs.type);
        if (lhs.data == nullptr || rhs.data == nullptr) {
            result.structural_error = "null tensor data";
            return result;
        }
        if (lhs.byte_size != expected_bytes || rhs.byte_size != expected_bytes) {
            result.structural_error = "byte-size mismatch";
            return result;
        }
        result.lhs = summarize_tensor(lhs);
        result.rhs = summarize_tensor(rhs);
    } catch (const std::exception& exc) {
        result.structural_error = exc.what();
        return result;
    }

    result.structurally_valid = true;
    result.byte_equal = std::memcmp(lhs.data, rhs.data, lhs.byte_size) == 0;
    std::vector<double> abs_diffs;
    abs_diffs.reserve(result.element_count);
    long double abs_sum = 0.0L;
    long double squared_sum = 0.0L;

    for (std::size_t i = 0; i < result.element_count; ++i) {
        bool equal = false;
        double signed_diff = 0.0;
        double abs_diff = 0.0;
        if (lhs.type == ElementType::UINT8) {
            const int l = static_cast<const std::uint8_t*>(lhs.data)[i];
            const int r = static_cast<const std::uint8_t*>(rhs.data)[i];
            signed_diff = static_cast<double>(l - r);
            abs_diff = std::abs(signed_diff);
            equal = l == r;
            ++result.signed_difference_histogram[l - r];
        } else {
            const float l = static_cast<const float*>(lhs.data)[i];
            const float r = static_cast<const float*>(rhs.data)[i];
            equal = float_values_equal(l, r);
            if (equal) {
                abs_diff = 0.0;
            } else if (std::isfinite(l) && std::isfinite(r)) {
                signed_diff = static_cast<double>(l) - static_cast<double>(r);
                abs_diff = std::abs(signed_diff);
            } else {
                abs_diff = std::numeric_limits<double>::infinity();
            }
        }

        abs_diffs.push_back(abs_diff);
        if (!equal) {
            ++result.mismatch_count;
            if (!result.first_mismatch_index.has_value()) {
                result.first_mismatch_index = i;
            }
        }
        result.max_abs_diff = std::max(result.max_abs_diff, abs_diff);
        abs_sum += abs_diff;
        squared_sum += abs_diff * abs_diff;
    }

    const long double count = static_cast<long double>(result.element_count);
    result.mismatch_ratio = static_cast<double>(result.mismatch_count) / static_cast<double>(result.element_count);
    result.mean_abs_diff = static_cast<double>(abs_sum / count);
    result.rmse = std::sqrt(static_cast<double>(squared_sum / count));
    std::sort(abs_diffs.begin(), abs_diffs.end());
    result.p50_abs_diff = percentile(abs_diffs, 0.50);
    result.p90_abs_diff = percentile(abs_diffs, 0.90);
    result.p95_abs_diff = percentile(abs_diffs, 0.95);
    result.p99_abs_diff = percentile(abs_diffs, 0.99);
    result.p999_abs_diff = percentile(abs_diffs, 0.999);
    return result;
}

OrtOptimizationLevel parse_ort_optimization_level(const std::string& value) {
    if (value == "disable") {
        return OrtOptimizationLevel::DISABLE;
    }
    if (value == "basic") {
        return OrtOptimizationLevel::BASIC;
    }
    if (value == "extended") {
        return OrtOptimizationLevel::EXTENDED;
    }
    if (value == "all") {
        return OrtOptimizationLevel::ALL;
    }
    throw std::invalid_argument("invalid ORT optimization level: " + value);
}

OrtExecutionMode parse_ort_execution_mode(const std::string& value) {
    if (value == "sequential") {
        return OrtExecutionMode::SEQUENTIAL;
    }
    if (value == "parallel") {
        return OrtExecutionMode::PARALLEL;
    }
    throw std::invalid_argument("invalid ORT execution mode: " + value);
}

ReferencePolicy parse_reference_policy(const std::string& value) {
    if (value == "matched-runtime") {
        return ReferencePolicy::MATCHED_RUNTIME;
    }
    if (value == "fixed-host-oracle") {
        return ReferencePolicy::FIXED_HOST_ORACLE;
    }
    throw std::invalid_argument("invalid reference policy: " + value);
}

std::string serialize_reference_policy(ReferencePolicy policy) {
    switch (policy) {
        case ReferencePolicy::MATCHED_RUNTIME:
            return "matched-runtime";
        case ReferencePolicy::FIXED_HOST_ORACLE:
            return "fixed-host-oracle";
    }
    throw std::invalid_argument("unsupported reference policy");
}

}  // namespace y26_stage42
