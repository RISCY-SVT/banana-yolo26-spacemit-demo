#include "y26_stage42_runner_support.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <typename Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    int failures = 0;
    failures += !throws([] { (void)y26_stage42::checked_element_count({1, -1, 80, 80}); });
    failures += !throws([] {
        (void)y26_stage42::checked_element_count(
            {std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::int64_t>::max()});
    });

    const std::vector<std::uint8_t> lhs = {0, 1, 2, 255};
    const std::vector<std::uint8_t> rhs = {0, 2, 2, 254};
    const y26_stage42::TensorView lhs_view{
        y26_stage42::ElementType::UINT8, {1, 4}, lhs.data(), lhs.size()};
    const y26_stage42::TensorView rhs_view{
        y26_stage42::ElementType::UINT8, {1, 4}, rhs.data(), rhs.size()};
    const y26_stage42::Comparison integer_cmp = y26_stage42::compare_tensors(lhs_view, rhs_view);
    failures += !integer_cmp.structurally_valid;
    failures += integer_cmp.mismatch_count != 2;
    failures += integer_cmp.max_abs_diff != 1.0;
    failures += !integer_cmp.first_mismatch_index.has_value() || *integer_cmp.first_mismatch_index != 1;

    const y26_stage42::TensorView bad_shape{
        y26_stage42::ElementType::UINT8, {2, 2}, rhs.data(), rhs.size()};
    const y26_stage42::Comparison structural = y26_stage42::compare_tensors(lhs_view, bad_shape);
    failures += structural.structurally_valid;
    failures += structural.structural_error != "shape mismatch";

    const y26_stage42::TensorView bad_bytes{
        y26_stage42::ElementType::UINT8, {1, 4}, rhs.data(), rhs.size() - 1};
    const y26_stage42::Comparison byte_size = y26_stage42::compare_tensors(lhs_view, bad_bytes);
    failures += byte_size.structurally_valid;
    failures += byte_size.structural_error != "byte-size mismatch";

    const y26_stage42::TensorView bad_type{
        y26_stage42::ElementType::FLOAT32, {1, 4}, rhs.data(), 4 * sizeof(float)};
    const y26_stage42::Comparison type = y26_stage42::compare_tensors(lhs_view, bad_type);
    failures += type.structurally_valid;
    failures += type.structural_error != "dtype mismatch";

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> float_lhs = {nan, inf, -inf, 1.0F};
    const std::vector<float> float_rhs = {nan, inf, inf, 1.0F};
    const y26_stage42::TensorView float_lhs_view{
        y26_stage42::ElementType::FLOAT32, {4}, float_lhs.data(), float_lhs.size() * sizeof(float)};
    const y26_stage42::TensorView float_rhs_view{
        y26_stage42::ElementType::FLOAT32, {4}, float_rhs.data(), float_rhs.size() * sizeof(float)};
    const y26_stage42::Comparison float_cmp = y26_stage42::compare_tensors(float_lhs_view, float_rhs_view);
    failures += float_cmp.mismatch_count != 1;
    failures += float_cmp.lhs.nonfinite_count != 3 || float_cmp.rhs.nonfinite_count != 3;

    failures += y26_stage42::parse_ort_optimization_level("disable") !=
                y26_stage42::OrtOptimizationLevel::DISABLE;
    failures += y26_stage42::parse_ort_optimization_level("basic") !=
                y26_stage42::OrtOptimizationLevel::BASIC;
    failures += y26_stage42::parse_ort_optimization_level("extended") !=
                y26_stage42::OrtOptimizationLevel::EXTENDED;
    failures += y26_stage42::parse_ort_optimization_level("all") !=
                y26_stage42::OrtOptimizationLevel::ALL;
    failures += y26_stage42::parse_ort_execution_mode("sequential") !=
                y26_stage42::OrtExecutionMode::SEQUENTIAL;
    failures += y26_stage42::parse_ort_execution_mode("parallel") !=
                y26_stage42::OrtExecutionMode::PARALLEL;
    failures += y26_stage42::serialize_reference_policy(
                    y26_stage42::parse_reference_policy("fixed-host-oracle")) != "fixed-host-oracle";
    failures += y26_stage42::serialize_reference_policy(
                    y26_stage42::parse_reference_policy("matched-runtime")) != "matched-runtime";
    failures += !throws([] { (void)y26_stage42::parse_ort_optimization_level("invalid"); });
    failures += !throws([] { (void)y26_stage42::parse_ort_execution_mode("invalid"); });

    std::cout << "stage42_runner_support failures=" << failures << "\n";
    return failures == 0 ? 0 : 1;
}
