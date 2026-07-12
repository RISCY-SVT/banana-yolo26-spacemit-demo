#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

struct EventSpec {
    std::string_view name;
    std::uint32_t type;
    std::uint64_t config;
};

#if defined(__linux__)
int perf_event_open(perf_event_attr* attributes) {
    return static_cast<int>(syscall(__NR_perf_event_open, attributes, 0, -1, -1, 0));
}

std::uint64_t diagnostic_work() {
    std::uint64_t value = 0x123456789abcdef0ULL;
    for (std::uint64_t i = 0; i < 20'000'000ULL; ++i) {
        value ^= value << 7;
        value ^= value >> 9;
        value += i * 0x9e3779b97f4a7c15ULL;
    }
    return value;
}
#endif

}  // namespace

int main() {
    std::cout << "event\tstatus\terrno\terror\tcount\n";
#if !defined(__linux__)
    std::cout << "all\tunavailable\t0\tnon-linux\t0\n";
    return 0;
#else
    constexpr EventSpec events[] = {
        {"task_clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
        {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {"cache_references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        {"cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {"branches", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        {"branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        {"context_switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES},
    };

    std::uint64_t sink = 0;
    for (const EventSpec& event : events) {
        perf_event_attr attributes{};
        attributes.type = event.type;
        attributes.size = sizeof(attributes);
        attributes.config = event.config;
        attributes.disabled = 1;
        attributes.exclude_kernel = 0;
        attributes.exclude_hv = 1;

        errno = 0;
        const int descriptor = perf_event_open(&attributes);
        if (descriptor < 0) {
            const int error = errno;
            std::cout << event.name << "\tunavailable\t" << error << '\t' << std::strerror(error) << "\t0\n";
            continue;
        }

        if (ioctl(descriptor, PERF_EVENT_IOC_RESET, 0) != 0 ||
            ioctl(descriptor, PERF_EVENT_IOC_ENABLE, 0) != 0) {
            const int error = errno;
            std::cout << event.name << "\tunavailable\t" << error << '\t' << std::strerror(error) << "\t0\n";
            close(descriptor);
            continue;
        }

        sink ^= diagnostic_work();
        const int disable_status = ioctl(descriptor, PERF_EVENT_IOC_DISABLE, 0);
        std::uint64_t count = 0;
        const ssize_t bytes = read(descriptor, &count, sizeof(count));
        const int read_error = errno;
        close(descriptor);
        if (disable_status != 0 || bytes != static_cast<ssize_t>(sizeof(count))) {
            std::cout << event.name << "\tunavailable\t" << read_error << '\t'
                      << std::strerror(read_error) << "\t0\n";
            continue;
        }
        std::cout << event.name << "\tavailable\t0\tok\t" << count << '\n';
    }
    std::cerr << "diagnostic_sink=" << sink << '\n';
    return 0;
#endif
}
