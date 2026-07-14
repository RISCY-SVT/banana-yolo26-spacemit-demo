#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

struct EventSpec {
    std::string_view name;
    std::uint32_t type;
    std::uint64_t config;
};

struct ReadValue {
    std::uint64_t value;
    std::uint64_t time_enabled;
    std::uint64_t time_running;
};

#if defined(__linux__)
int open_cpu_event(const EventSpec& event, int cpu) {
    perf_event_attr attributes{};
    attributes.type = event.type;
    attributes.size = sizeof(attributes);
    attributes.config = event.config;
    attributes.disabled = 1;
    attributes.exclude_hv = 1;
    attributes.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return static_cast<int>(syscall(__NR_perf_event_open, &attributes, -1, cpu, -1, 0));
}
#endif

const EventSpec* find_event(std::string_view name) {
#if defined(__linux__)
    static constexpr EventSpec events[] = {
        {"task_clock", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
        {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {"cache_references", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        {"cache_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {"branches", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        {"branch_misses", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        {"context_switches", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES},
        {"cpu_migrations", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS},
    };
    for (const EventSpec& event : events) {
        if (event.name == name) {
            return &event;
        }
    }
#else
    (void)name;
#endif
    return nullptr;
}

bool parse_cpu_list(std::string_view text, std::vector<int>* cpus) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string item(text.substr(begin, end == std::string_view::npos ? text.size() - begin : end - begin));
        try {
            const int cpu = std::stoi(item);
            if (cpu < 0) {
                return false;
            }
            cpus->push_back(cpu);
        } catch (...) {
            return false;
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    return !cpus->empty();
}

}  // namespace

int main(int argc, char** argv) {
#if !defined(__linux__)
    (void)argc;
    (void)argv;
    std::cerr << "stage49_perf_exec requires Linux\n";
    return 2;
#else
    std::string event_name;
    std::vector<int> cpus;
    int command_index = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--event" && i + 1 < argc) {
            event_name = argv[++i];
        } else if (argument == "--cpus" && i + 1 < argc) {
            if (!parse_cpu_list(argv[++i], &cpus)) {
                std::cerr << "invalid --cpus list\n";
                return 2;
            }
        } else if (argument == "--") {
            command_index = i + 1;
            break;
        } else {
            std::cerr << "usage: stage49_perf_exec --event NAME --cpus 0,1,2,3 -- command [args...]\n";
            return 2;
        }
    }
    if (command_index < 0 || command_index >= argc || event_name.empty() || cpus.empty()) {
        std::cerr << "missing event, CPU list, or command\n";
        return 2;
    }

    const EventSpec* event = find_event(event_name);
    if (event == nullptr) {
        std::cerr << "unsupported event name: " << event_name << '\n';
        return 2;
    }

    std::vector<int> descriptors;
    descriptors.reserve(cpus.size());
    std::cout << "event\tcpu\tstatus\terrno\terror\tcount\ttime_enabled\ttime_running\tscale\n";
    for (const int cpu : cpus) {
        errno = 0;
        const int descriptor = open_cpu_event(*event, cpu);
        if (descriptor < 0) {
            const int error = errno;
            std::cout << event->name << '\t' << cpu << "\tunavailable\t" << error << '\t'
                      << std::strerror(error) << "\t0\t0\t0\t0\n";
            for (const int open_descriptor : descriptors) {
                close(open_descriptor);
            }
            return 3;
        }
        descriptors.push_back(descriptor);
    }

    for (const int descriptor : descriptors) {
        if (ioctl(descriptor, PERF_EVENT_IOC_RESET, 0) != 0 ||
            ioctl(descriptor, PERF_EVENT_IOC_ENABLE, 0) != 0) {
            std::cerr << "failed to start counters: " << std::strerror(errno) << '\n';
            for (const int open_descriptor : descriptors) {
                close(open_descriptor);
            }
            return 3;
        }
    }

    const pid_t child = fork();
    if (child < 0) {
        std::cerr << "fork failed: " << std::strerror(errno) << '\n';
        return 4;
    }
    if (child == 0) {
        execvp(argv[command_index], &argv[command_index]);
        std::cerr << "exec failed: " << std::strerror(errno) << '\n';
        _exit(127);
    }

    int child_status = 0;
    if (waitpid(child, &child_status, 0) < 0) {
        std::cerr << "waitpid failed: " << std::strerror(errno) << '\n';
        child_status = 127 << 8;
    }
    for (const int descriptor : descriptors) {
        ioctl(descriptor, PERF_EVENT_IOC_DISABLE, 0);
    }

    bool read_ok = true;
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        ReadValue value{};
        errno = 0;
        const ssize_t bytes = read(descriptors[i], &value, sizeof(value));
        const int error = errno;
        close(descriptors[i]);
        if (bytes != static_cast<ssize_t>(sizeof(value))) {
            read_ok = false;
            std::cout << event->name << '\t' << cpus[i] << "\tunavailable\t" << error << '\t'
                      << std::strerror(error) << "\t0\t0\t0\t0\n";
            continue;
        }
        const double scale = value.time_running == 0
                                 ? 0.0
                                 : static_cast<double>(value.time_enabled) / static_cast<double>(value.time_running);
        std::cout << event->name << '\t' << cpus[i] << "\tavailable\t0\tok\t" << value.value << '\t'
                  << value.time_enabled << '\t' << value.time_running << '\t' << scale << '\n';
    }
    std::cout << "child_exit_code\t";
    if (WIFEXITED(child_status)) {
        std::cout << WEXITSTATUS(child_status) << '\n';
    } else if (WIFSIGNALED(child_status)) {
        std::cout << 128 + WTERMSIG(child_status) << '\n';
    } else {
        std::cout << "unknown\n";
    }
    return read_ok && WIFEXITED(child_status) ? WEXITSTATUS(child_status) : 5;
#endif
}
