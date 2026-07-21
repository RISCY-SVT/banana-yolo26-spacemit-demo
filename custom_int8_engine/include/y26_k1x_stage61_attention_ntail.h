#pragma once

#include <cstddef>
#include <cstdint>

namespace y26::stage61 {

enum class Ntail13Strategy {
    n8_n8,
    padded_n16,
};

struct NtailRouteCount {
    int n4 = 0;
    int n8 = 0;
    int n16 = 0;
    int padded_dead_columns = 0;
};

NtailRouteCount ntail_route_count(int live_columns,
                                  Ntail13Strategy strategy) noexcept;

// The packed B panel always has an N16 stride. C uses the existing
// output-group-major M12 layout and only covers ceil(live_columns / 4) groups.
bool run_m12n_tail(const std::int8_t* packed_a,
                   const std::int8_t* packed_b,
                   int k_tiles,
                   int live_columns,
                   Ntail13Strategy strategy,
                   std::int32_t* output,
                   std::size_t output_elements) noexcept;

}  // namespace y26::stage61
