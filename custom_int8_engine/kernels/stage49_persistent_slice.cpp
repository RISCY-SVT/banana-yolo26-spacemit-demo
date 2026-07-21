#include "y26_k1x_stage49_slice.h"

#include "y26_k1x_conv_kernels.h"
#include "y26_k1x_int8_v1.h"
#include "y26_k1x_package.h"
#include "y26_k1x_stage51_q62.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#if defined(__linux__)
#include <linux/perf_event.h>
#include <pthread.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
extern "C" void y26_stage48_kernel_m4n16(const std::int8_t*, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage48_kernel_m8n16(const std::int8_t*, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage48_kernel_m12n16(const std::int8_t*, const std::int8_t*, int, std::int32_t*);
extern "C" void y26_stage48_load_vlse64_4(const std::int8_t*, std::int8_t*);
extern "C" void y26_stage48_load_vlseg2e64_4(const std::int8_t*, std::int8_t*);
#endif

namespace y26::stage49 {
namespace {

using Clock = std::chrono::steady_clock;
using Row = std::unordered_map<std::string, std::string>;
constexpr int kMaximumWorkers = 4;
constexpr int kNBlock = 16;

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

int rows_for(KernelShape shape) noexcept { return static_cast<int>(shape); }

bool pin_current_thread(int cpu) noexcept {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

int current_cpu() noexcept {
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> values;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        values.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return values;
}

std::vector<Row> read_tsv(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::string line;
    if (!stream || !std::getline(stream, line)) throw std::runtime_error("cannot read TSV: " + path.string());
    const auto header = split_tsv(line);
    if (header.empty()) throw std::runtime_error("empty TSV header: " + path.string());
    std::vector<Row> rows;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        const auto values = split_tsv(line);
        if (values.size() != header.size()) throw std::runtime_error("malformed TSV: " + path.string());
        Row& row = rows.emplace_back();
        for (std::size_t index = 0; index < header.size(); ++index) row.emplace(header[index], values[index]);
    }
    return rows;
}

const std::string& field(const Row& row, const char* name) {
    const auto found = row.find(name);
    if (found == row.end()) throw std::runtime_error(std::string("missing TSV field: ") + name);
    return found->second;
}

std::int64_t parse_i64(std::string_view text, const char* name) {
    std::int64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) {
        throw std::runtime_error(std::string("invalid integer field: ") + name);
    }
    return value;
}

int integer(const Row& row, const char* name) {
    const auto value = parse_i64(field(row, name), name);
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("integer field out of range: ") + name);
    }
    return static_cast<int>(value);
}

int optional_integer(const Row& row, const char* name, int fallback) {
    return row.contains(name) ? integer(row, name) : fallback;
}

std::size_t size_value(const Row& row, const char* name) {
    const auto value = parse_i64(field(row, name), name);
    if (value < 0 || static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string("size field out of range: ") + name);
    }
    return static_cast<std::size_t>(value);
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t expected_count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open binary asset: " + path.string());
    const std::streamsize bytes = stream.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) != expected_count * sizeof(T)) {
        throw std::runtime_error("binary asset size mismatch: " + path.string());
    }
    stream.seekg(0);
    std::vector<T> values(expected_count);
    if (bytes != 0 && !stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("binary asset read failed: " + path.string());
    }
    return values;
}

struct Tensor {
    int id = -1;
    std::string key;
    std::string name;
    int h = 0;
    int w = 0;
    int c = 0;
    int zero_point = 0;
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

struct Segment {
    int channel_begin = 0;
    int channel_count = 0;
    int tensor_id = -1;
    std::array<std::int8_t, 256> lut {};
};

struct ConvAsset {
    int input_h = 0;
    int input_w = 0;
    int input_c = 0;
    int output_h = 0;
    int output_w = 0;
    int output_c = 0;
    int kernel_h = 0;
    int kernel_w = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;
    int input_zero_point = 0;
    int conv_output_zero_point = 0;
    int k = 0;
    int k_tiles = 0;
    int n_blocks = 0;
    std::uint64_t accumulator_bound = 0;
    std::vector<std::int8_t> packed_weights;
    std::vector<std::int32_t> weight_sums;
    std::vector<std::int32_t> bias;
    std::vector<std::int64_t> multiplier;
    std::vector<std::int64_t> multiplier_m63;
    std::vector<std::int32_t> shift;
    std::vector<std::int64_t> corrected_bias;
    std::array<Segment, 2> segments;
    int segment_count = 0;
    bool vsmul_q62_compatible = false;
};

struct Operation {
    int index = -1;
    std::string kind;
    std::string name;
    std::array<int, 4> inputs {-1, -1, -1, -1};
    std::array<int, 2> outputs {-1, -1};
    ConvAsset conv;
    int pool_kernel_h = 0;
    int pool_kernel_w = 0;
    int pool_stride_h = 0;
    int pool_stride_w = 0;
    int pool_pad_h = 0;
    int pool_pad_w = 0;
    std::array<std::int8_t, 256> lut {};
    std::vector<std::int8_t> add_lut;
    std::array<std::array<std::int8_t, 256>, 4> concat_lut {};
};

struct WorkerScratch {
    std::vector<std::int8_t> a_panel;
    std::array<std::int32_t, 12 * kNBlock> c_tile {};
    int observed_cpu = -1;
    int tid = -1;
    bool affinity_set = false;
    struct Counter {
        int fd = -1;
        int error_number = 0;
        std::uint64_t id = 0;
        std::string event;
        std::string status = "disabled";
        std::uint64_t value = 0;
        std::uint64_t time_enabled = 0;
        std::uint64_t time_running = 0;
    };
    std::string counter_group;
    std::vector<Counter> counters;
    std::uint64_t counter_iterations = 0;
    std::uint64_t counter_last_time_enabled = 0;
    std::uint64_t counter_last_time_running = 0;
};

struct PerfEventSpec {
    std::uint32_t type = 0;
    std::uint64_t config = 0;
};

struct PerfGroupRead {
    std::uint64_t nr = 0;
    std::uint64_t time_enabled = 0;
    std::uint64_t time_running = 0;
    struct Value {
        std::uint64_t value = 0;
        std::uint64_t id = 0;
    } values[8] {};
};

bool perf_event_spec(std::string_view name, PerfEventSpec* event) noexcept {
#if defined(__linux__)
    if (name == "task_clock") *event = {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK};
    else if (name == "cycles") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES};
    else if (name == "instructions") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS};
    else if (name == "cache_references") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES};
    else if (name == "cache_misses") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES};
    else if (name == "branches") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS};
    else if (name == "branch_misses") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES};
    else if (name == "bus_cycles") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES};
    else if (name == "frontend_stalls") {
        *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND};
    }
    else if (name == "backend_stalls") {
        *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND};
    }
    else if (name == "ref_cycles") *event = {PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES};
    else if (name == "l1d_read_access") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_L1D |
                      (PERF_COUNT_HW_CACHE_OP_READ << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16U)};
    }
    else if (name == "l1d_read_miss") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_L1D |
                      (PERF_COUNT_HW_CACHE_OP_READ << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16U)};
    }
    else if (name == "l1d_write_access") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_L1D |
                      (PERF_COUNT_HW_CACHE_OP_WRITE << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16U)};
    }
    else if (name == "l1d_write_miss") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_L1D |
                      (PERF_COUNT_HW_CACHE_OP_WRITE << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16U)};
    }
    else if (name == "l1i_read_access") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_L1I |
                      (PERF_COUNT_HW_CACHE_OP_READ << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16U)};
    }
    else if (name == "l1i_read_miss") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_L1I |
                      (PERF_COUNT_HW_CACHE_OP_READ << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16U)};
    }
    else if (name == "dtlb_read_access") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_DTLB |
                      (PERF_COUNT_HW_CACHE_OP_READ << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16U)};
    }
    else if (name == "dtlb_read_miss") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_DTLB |
                      (PERF_COUNT_HW_CACHE_OP_READ << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16U)};
    }
    else if (name == "dtlb_write_access") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_DTLB |
                      (PERF_COUNT_HW_CACHE_OP_WRITE << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16U)};
    }
    else if (name == "dtlb_write_miss") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_DTLB |
                      (PERF_COUNT_HW_CACHE_OP_WRITE << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16U)};
    }
    else if (name == "itlb_read_miss") {
        *event = {PERF_TYPE_HW_CACHE,
                  PERF_COUNT_HW_CACHE_ITLB |
                      (PERF_COUNT_HW_CACHE_OP_READ << 8U) |
                      (PERF_COUNT_HW_CACHE_RESULT_MISS << 16U)};
    }
    else if (name == "context_switches") *event = {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES};
    else if (name.starts_with("raw:")) {
        std::string_view raw = name.substr(4);
        if (raw.starts_with("0x") || raw.starts_with("0X")) raw.remove_prefix(2);
        std::uint64_t config = 0;
        const auto parsed = std::from_chars(raw.data(), raw.data() + raw.size(), config, 16);
        if (raw.empty() || parsed.ec != std::errc {} || parsed.ptr != raw.data() + raw.size()) {
            return false;
        }
        *event = {PERF_TYPE_RAW, config};
    }
    else return false;
    return true;
#else
    (void)name;
    (void)event;
    return false;
#endif
}

int open_thread_perf_event(std::string_view name, int group_fd, bool leader) noexcept {
#if defined(__linux__)
    PerfEventSpec event;
    if (!perf_event_spec(name, &event)) {
        errno = EINVAL;
        return -1;
    }
    perf_event_attr attributes {};
    attributes.type = event.type;
    attributes.size = sizeof(attributes);
    attributes.config = event.config;
    attributes.disabled = leader ? 1U : 0U;
    attributes.exclude_kernel = 1;
    attributes.exclude_hv = 1;
    attributes.read_format = leader
        ? PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
              PERF_FORMAT_TOTAL_TIME_RUNNING | PERF_FORMAT_ID
        : 0;
    return static_cast<int>(syscall(__NR_perf_event_open, &attributes, 0, -1,
                                    group_fd, PERF_FLAG_FD_CLOEXEC));
#else
    (void)name;
    (void)group_fd;
    (void)leader;
    errno = ENOSYS;
    return -1;
#endif
}

std::vector<std::string> split_counter_group(std::string_view group) {
    std::vector<std::string> events;
    std::size_t begin = 0;
    while (begin < group.size()) {
        const std::size_t comma = group.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? group.size() : comma;
        if (end > begin) events.emplace_back(group.substr(begin, end - begin));
        if (comma == std::string_view::npos) break;
        begin = comma + 1;
    }
    return events;
}

void close_counter_group(WorkerScratch& scratch) noexcept {
#if defined(__linux__)
    for (WorkerScratch::Counter& counter : scratch.counters) {
        if (counter.fd >= 0) close(counter.fd);
        counter.fd = -1;
    }
#endif
    scratch.counters.clear();
    scratch.counter_group.clear();
    scratch.counter_iterations = 0;
    scratch.counter_last_time_enabled = 0;
    scratch.counter_last_time_running = 0;
}

void open_counter_group(WorkerScratch& scratch, std::string_view group) {
    close_counter_group(scratch);
    scratch.counter_group.assign(group);
    const std::vector<std::string> events = split_counter_group(group);
    if (events.empty()) return;

    scratch.counters.reserve(events.size());
    int leader_fd = -1;
    int group_error = 0;
    for (std::size_t index = 0; index < events.size(); ++index) {
        WorkerScratch::Counter& counter = scratch.counters.emplace_back();
        counter.event = events[index];
        if (group_error != 0) {
            counter.error_number = group_error;
            counter.status = "group-unavailable";
            continue;
        }
        counter.fd = open_thread_perf_event(counter.event, leader_fd, index == 0);
        if (counter.fd < 0) {
            group_error = errno;
            counter.error_number = group_error;
            counter.status = "unavailable";
            continue;
        }
        if (index == 0) leader_fd = counter.fd;
#if defined(__linux__)
        if (ioctl(counter.fd, PERF_EVENT_IOC_ID, &counter.id) != 0) {
            group_error = errno;
            counter.error_number = group_error;
            counter.status = "unavailable";
            continue;
        }
#endif
        counter.status = "available";
    }
    if (group_error != 0) {
        for (WorkerScratch::Counter& counter : scratch.counters) {
#if defined(__linux__)
            if (counter.fd >= 0) close(counter.fd);
#endif
            counter.fd = -1;
            if (counter.error_number == 0) counter.error_number = group_error;
            counter.status = "group-unavailable";
        }
    }
}

void reset_counter_values(WorkerScratch& scratch) noexcept {
    scratch.counter_iterations = 0;
    for (WorkerScratch::Counter& counter : scratch.counters) {
        counter.value = 0;
        counter.time_enabled = 0;
        counter.time_running = 0;
        if (counter.fd >= 0) {
            counter.error_number = 0;
            counter.status = "available";
        }
    }
}

class WorkerPool {
public:
    using Job = void (*)(void*, int, WorkerScratch&);

    WorkerPool(int requested, std::size_t panel_bytes, int cpu_base,
               WorkerWakePolicy wake_policy)
        : count_(std::clamp(requested, 1, kMaximumWorkers)), cpu_base_(cpu_base) {
        if (wake_policy == WorkerWakePolicy::frame_gated_spin) {
            spin_policy_ = 1;
            frame_gated_spin_ = true;
        }
#if !defined(Y26_K1X_FROZEN_RELEASE_PROFILE)
        if (const char* policy = std::getenv("Y26_STAGE54_SPIN_POLICY"); policy != nullptr) {
            const std::string_view value(policy);
            if (value == "pause") spin_policy_ = 2;
            else if (value == "adaptive") spin_policy_ = 3;
            else if (value == "raw") spin_policy_ = 1;
        } else if (const char* value = std::getenv("Y26_STAGE53_SPIN_POOL");
                   value != nullptr && std::string_view(value) == "1") {
            spin_policy_ = 1;
        }
        if (const char* value = std::getenv("Y26_STAGE55_FRAME_GATED_SPIN");
            value != nullptr && std::string_view(value) == "1") {
            frame_gated_spin_ = true;
        }
#endif
        spin_mode_ = spin_policy_ != 0;
        scratch_.resize(static_cast<std::size_t>(count_));
        for (WorkerScratch& scratch : scratch_) scratch.a_panel.resize(panel_bytes);
        threads_.reserve(static_cast<std::size_t>(count_));
        for (int index = 0; index < count_; ++index) {
            threads_.emplace_back([this, index]() { worker_loop(index); });
        }
        std::unique_lock lock(mutex_);
        ready_cv_.wait(lock, [this]() { return ready_ == count_; });
    }

    ~WorkerPool() {
        if (spin_mode_) {
            spin_stopping_.store(true, std::memory_order_relaxed);
            spin_generation_.fetch_add(1, std::memory_order_release);
            active_window_cv_.notify_all();
            for (std::thread& thread : threads_) if (thread.joinable()) thread.join();
            for (WorkerScratch& scratch : scratch_) close_counter_group(scratch);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& thread : threads_) if (thread.joinable()) thread.join();
        for (WorkerScratch& scratch : scratch_) close_counter_group(scratch);
    }

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    int capacity() const noexcept { return count_; }

    void begin_active_window() noexcept {
        if (!spin_mode_ || !frame_gated_spin_) return;
        std::unique_lock lock(mutex_);
        active_window_.store(true, std::memory_order_release);
        active_window_cv_.notify_all();
        active_window_cv_.wait(lock, [this]() {
            return active_window_parked_ == 0 ||
                spin_stopping_.load(std::memory_order_relaxed);
        });
    }

    void end_active_window() noexcept {
        if (!spin_mode_ || !frame_gated_spin_) return;
        std::unique_lock lock(mutex_);
        active_window_.store(false, std::memory_order_release);
        active_window_cv_.wait(lock, [this]() {
            return active_window_parked_ == count_ ||
                spin_stopping_.load(std::memory_order_relaxed);
        });
    }

    bool affinity_ok() const noexcept {
        for (const WorkerScratch& scratch : scratch_) {
            if (!scratch.affinity_set || scratch.observed_cpu < cpu_base_ ||
                scratch.observed_cpu >= cpu_base_ + count_) {
                return false;
            }
        }
        return true;
    }

    void begin_counter_collection(std::string_view event) {
        {
            std::lock_guard lock(mutex_);
            counter_event_.assign(event);
            for (WorkerScratch& scratch : scratch_) reset_counter_values(scratch);
        }
        if (!event.empty()) {
            dispatch(count_, [](void*, int, WorkerScratch&) {}, nullptr,
                     SchedulerStrategy::all_workers_complete);
            for (WorkerScratch& scratch : scratch_) reset_counter_values(scratch);
        }
    }

    std::vector<WorkerCounter> worker_counters() const {
        std::vector<WorkerCounter> counters;
        counters.reserve(scratch_.size());
        for (std::size_t index = 0; index < scratch_.size(); ++index) {
            const WorkerScratch& scratch = scratch_[index];
            for (const WorkerScratch::Counter& counter : scratch.counters) {
                counters.push_back({static_cast<int>(index), scratch.tid, scratch.observed_cpu,
                                    counter.event, counter.status, counter.error_number,
                                    counter.id, scratch.counter_iterations, counter.value, counter.time_enabled,
                                    counter.time_running});
            }
        }
        return counters;
    }

    void dispatch(int active, Job job, void* context, SchedulerStrategy scheduler) {
        if (spin_mode_) {
            active_ = std::clamp(active, 1, count_);
            active_workers_only_ = scheduler == SchedulerStrategy::active_workers_complete;
            completion_target_ = active_workers_only_ ? active_ : count_;
            job_ = job;
            context_ = context;
            spin_completed_.store(0, std::memory_order_relaxed);
            spin_generation_.fetch_add(1, std::memory_order_release);
            std::size_t wait_iterations = 0;
            while (spin_completed_.load(std::memory_order_acquire) != completion_target_) {
                spin_wait(wait_iterations);
            }
            return;
        }
        {
            std::lock_guard lock(mutex_);
            active_ = std::clamp(active, 1, count_);
            active_workers_only_ = scheduler == SchedulerStrategy::active_workers_complete;
            completion_target_ = active_workers_only_ ? active_ : count_;
            job_ = job;
            context_ = context;
            completed_ = 0;
            ++generation_;
        }
        start_cv_.notify_all();
        std::unique_lock lock(mutex_);
        done_cv_.wait(lock, [this]() { return completed_ == completion_target_; });
    }

private:
    static void pause_hint() noexcept {
#if defined(__riscv)
        __asm__ volatile(
            ".option push\n\t"
            ".option arch,+zihintpause\n\t"
            "pause\n\t"
            ".option pop\n\t"
            ::: "memory");
#else
        __asm__ volatile("" ::: "memory");
#endif
    }

    void spin_wait(std::size_t& iterations) const noexcept {
        ++iterations;
        if (spin_policy_ == 2) {
            pause_hint();
            return;
        }
        if (spin_policy_ == 3) {
            if (iterations < 256) {
                __asm__ volatile("" ::: "memory");
            } else if (iterations < 4096) {
                pause_hint();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                iterations = 256;
            }
            return;
        }
        __asm__ volatile("" ::: "memory");
    }

    void worker_loop(int index) {
        WorkerScratch& scratch = scratch_[static_cast<std::size_t>(index)];
        scratch.affinity_set = pin_current_thread(cpu_base_ + index);
        scratch.observed_cpu = current_cpu();
#if defined(__linux__)
        scratch.tid = static_cast<int>(syscall(SYS_gettid));
#endif
        {
            std::lock_guard lock(mutex_);
            ++ready_;
        }
        ready_cv_.notify_one();
        std::uint64_t local_generation = 0;
        for (;;) {
            Job job = nullptr;
            void* context = nullptr;
            int active = 0;
            bool active_workers_only = false;
            std::string counter_event;
            if (spin_mode_) {
                if (frame_gated_spin_ && !active_window_.load(std::memory_order_acquire)) {
                    std::unique_lock lock(mutex_);
                    // Recheck under the lifecycle mutex. A new active window may
                    // have started after the fast-path atomic load.
                    if (!active_window_.load(std::memory_order_acquire)) {
                        ++active_window_parked_;
                        active_window_cv_.notify_all();
                        active_window_cv_.wait(lock, [&]() {
                            return spin_stopping_.load(std::memory_order_relaxed) ||
                                active_window_.load(std::memory_order_acquire);
                        });
                        --active_window_parked_;
                        active_window_cv_.notify_all();
                    }
                    if (spin_stopping_.load(std::memory_order_relaxed)) return;
                }
                std::uint64_t generation = spin_generation_.load(std::memory_order_acquire);
                std::size_t wait_iterations = 0;
                while (generation == local_generation) {
                    if (spin_stopping_.load(std::memory_order_relaxed)) return;
                    if (frame_gated_spin_ &&
                        !active_window_.load(std::memory_order_acquire)) {
                        break;
                    }
                    spin_wait(wait_iterations);
                    generation = spin_generation_.load(std::memory_order_acquire);
                }
                // A new active window can begin after the park check but before this
                // worker observes a new job generation. Do not replay the previous job.
                if (generation == local_generation) continue;
                if (frame_gated_spin_ &&
                    !active_window_.load(std::memory_order_acquire)) {
                    continue;
                }
                local_generation = generation;
                if (spin_stopping_.load(std::memory_order_relaxed)) return;
                job = job_;
                context = context_;
                active = active_;
                active_workers_only = active_workers_only_;
                counter_event = counter_event_;
            } else {
                std::unique_lock lock(mutex_);
                start_cv_.wait(lock, [&]() { return generation_ != local_generation; });
                local_generation = generation_;
                if (stopping_) return;
                job = job_;
                context = context_;
                active = active_;
                active_workers_only = active_workers_only_;
                counter_event = counter_event_;
            }
            if (index < active && job != nullptr) {
                scratch.observed_cpu = current_cpu();
                const bool count = !counter_event.empty();
                if (count && scratch.counter_group != counter_event) {
                    open_counter_group(scratch, counter_event);
                }
                bool counter_started = false;
#if defined(__linux__)
                const int leader_fd = !scratch.counters.empty() ? scratch.counters.front().fd : -1;
                if (count && leader_fd >= 0) {
                    if (ioctl(leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == 0 &&
                        ioctl(leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == 0) {
                        counter_started = true;
                    } else {
                        for (WorkerScratch::Counter& counter : scratch.counters) {
                            counter.error_number = errno;
                            counter.status = "unavailable";
                        }
                    }
                }
#endif
                job(context, index, scratch);
#if defined(__linux__)
                if (counter_started) {
                    const int leader_fd = scratch.counters.front().fd;
                    PerfGroupRead values;
                    const std::size_t expected_size = 3U * sizeof(std::uint64_t) +
                        2U * sizeof(std::uint64_t) * scratch.counters.size();
                    const ssize_t bytes = ioctl(leader_fd, PERF_EVENT_IOC_DISABLE,
                                                PERF_IOC_FLAG_GROUP) == 0
                        ? read(leader_fd, &values, sizeof(values))
                        : -1;
                    if (bytes < 0 || static_cast<std::size_t>(bytes) < expected_size ||
                        values.nr != scratch.counters.size()) {
                        const int failure = errno == 0 ? EIO : errno;
                        for (WorkerScratch::Counter& counter : scratch.counters) {
                            counter.error_number = failure;
                            counter.status = "unavailable";
                        }
                    } else {
                        const std::uint64_t enabled_delta = values.time_enabled >= scratch.counter_last_time_enabled
                            ? values.time_enabled - scratch.counter_last_time_enabled
                            : values.time_enabled;
                        const std::uint64_t running_delta = values.time_running >= scratch.counter_last_time_running
                            ? values.time_running - scratch.counter_last_time_running
                            : values.time_running;
                        scratch.counter_last_time_enabled = values.time_enabled;
                        scratch.counter_last_time_running = values.time_running;
                        ++scratch.counter_iterations;
                        for (std::size_t value_index = 0; value_index < values.nr; ++value_index) {
                            const PerfGroupRead::Value& value = values.values[value_index];
                            const auto found = std::find_if(
                                scratch.counters.begin(), scratch.counters.end(),
                                [&](const WorkerScratch::Counter& counter) {
                                    return counter.id == value.id;
                                });
                            if (found == scratch.counters.end()) continue;
                            found->value += value.value;
                            found->time_enabled += enabled_delta;
                            found->time_running += running_delta;
                            found->status = running_delta == 0
                                ? "unmapped-or-unsupported" : "available";
                        }
                    }
                }
#endif
            }
            if (!active_workers_only || index < active) {
                if (spin_mode_) {
                    spin_completed_.fetch_add(1, std::memory_order_release);
                    continue;
                }
                {
                    std::lock_guard lock(mutex_);
                    ++completed_;
                }
                done_cv_.notify_one();
            }
        }
    }

    int count_ = 0;
    int cpu_base_ = 0;
    std::vector<std::thread> threads_;
    std::vector<WorkerScratch> scratch_;
    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::condition_variable ready_cv_;
    std::condition_variable active_window_cv_;
    std::uint64_t generation_ = 0;
    int active_ = 0;
    int completed_ = 0;
    int completion_target_ = 0;
    int ready_ = 0;
    Job job_ = nullptr;
    void* context_ = nullptr;
    bool active_workers_only_ = false;
    bool stopping_ = false;
    int spin_policy_ = 0;
    bool spin_mode_ = false;
    bool frame_gated_spin_ = false;
    int active_window_parked_ = 0;
    std::atomic<std::uint64_t> spin_generation_ {0};
    std::atomic<int> spin_completed_ {0};
    std::atomic<bool> spin_stopping_ {false};
    std::atomic<bool> active_window_ {false};
    std::string counter_event_;
};

void scalar_block(const std::int8_t* a, const std::int8_t* b,
                  int k_tiles, int rows, std::int32_t* c) {
    std::fill(c, c + static_cast<std::size_t>(rows) * kNBlock, 0);
    const int row_groups = rows / 4;
    for (int tile = 0; tile < k_tiles; ++tile) {
        const std::int8_t* at = a + static_cast<std::size_t>(tile) * rows * 8;
        const std::int8_t* bt = b + static_cast<std::size_t>(tile) * kNBlock * 8;
        for (int output_group = 0; output_group < 4; ++output_group) {
            for (int row_group = 0; row_group < row_groups; ++row_group) {
                std::int32_t* result = c + static_cast<std::size_t>(output_group * row_groups + row_group) * 16;
                for (int row = 0; row < 4; ++row) {
                    for (int output = 0; output < 4; ++output) {
                        std::int32_t sum = result[row * 4 + output];
                        for (int lane = 0; lane < 8; ++lane) {
                            sum += static_cast<std::int32_t>(at[(row_group * 4 + row) * 8 + lane]) *
                                   static_cast<std::int32_t>(bt[(output_group * 4 + output) * 8 + lane]);
                        }
                        result[row * 4 + output] = sum;
                    }
                }
            }
        }
    }
}

void run_block(ComputeRoute route, KernelShape shape, const std::int8_t* a,
               const std::int8_t* b, int k_tiles, std::int32_t* c) {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (route == ComputeRoute::ime) {
        if (shape == KernelShape::m4n16) y26_stage48_kernel_m4n16(a, b, k_tiles, c);
        else if (shape == KernelShape::m8n16) y26_stage48_kernel_m8n16(a, b, k_tiles, c);
        else y26_stage48_kernel_m12n16(a, b, k_tiles, c);
        return;
    }
#else
    (void)route;
#endif
    scalar_block(a, b, k_tiles, rows_for(shape), c);
}

std::size_t nchwc8_offset(int channel, int y, int x, int h, int w) noexcept {
    return (((static_cast<std::size_t>(channel / 8) * h + y) * w + x) * 8U) + channel % 8;
}

void copy_c8(const std::int8_t* source, std::int8_t* destination) noexcept {
    std::uint64_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    std::memcpy(destination, &value, sizeof(value));
}

#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
extern "C" __attribute__((noinline)) void y26_stage49_load_vlseg2_pair_4(
    const std::int8_t* source, std::int8_t* even, std::int8_t* odd) {
    __asm__ volatile(
        "vsetivli t0, 4, e64, m1, ta, ma\n\t"
        "vlseg2e64.v v0, (%[source])\n\t"
        "vse64.v v0, (%[even])\n\t"
        "vse64.v v1, (%[odd])\n\t"
        :
        : [source] "r"(source), [even] "r"(even), [odd] "r"(odd)
        : "memory", "t0", "v0", "v1");
}

extern "C" __attribute__((noinline)) void y26_stage49_load_contiguous_c8x4(
    const std::int8_t* source, std::int8_t* destination) {
    __asm__ volatile(
        "li t0, 32\n\t"
        "vsetvli t0, t0, e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[source])\n\t"
        "vse8.v v0, (%[destination])\n\t"
        :
        : [source] "r"(source), [destination] "r"(destination)
        : "memory", "t0", "v0");
}
#endif

void load_stride_group(const std::int8_t* source, int stride,
                       LoadStrategy strategy, std::int8_t* destination) {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    if (stride == 1 && strategy != LoadStrategy::four_u64) {
        y26_stage49_load_contiguous_c8x4(source, destination);
        return;
    }
    if (stride == 2 && strategy == LoadStrategy::vlse64) {
        y26_stage48_load_vlse64_4(source, destination);
        return;
    }
    if (stride == 2 && strategy == LoadStrategy::vlseg2_even) {
        y26_stage48_load_vlseg2e64_4(source, destination);
        return;
    }
#else
    (void)strategy;
#endif
    for (int row = 0; row < 4; ++row) copy_c8(source + static_cast<std::size_t>(row * stride) * 8U,
                                               destination + row * 8);
}

void pack_a_scalar_positions(const ConvAsset& conv, const std::int8_t* input,
                             int m_begin, int rows, int valid_rows,
                             std::int8_t* panel) {
    const std::int8_t padding = int8_v1::signed_storage(static_cast<std::uint8_t>(conv.input_zero_point));
    const std::uint64_t padding_word = static_cast<std::uint64_t>(static_cast<std::uint8_t>(padding)) *
                                       0x0101010101010101ULL;
    const int channel_blocks = conv.input_c / 8;
    for (int tile = 0; tile < conv.k_tiles; ++tile) {
        const int channel_block = tile % channel_blocks;
        const int kernel_position = tile / channel_blocks;
        const int kernel_y = kernel_position / conv.kernel_w;
        const int kernel_x = kernel_position % conv.kernel_w;
        for (int row = 0; row < rows; ++row) {
            std::int8_t* destination = panel + (static_cast<std::size_t>(tile) * rows + row) * 8U;
            const int flat = m_begin + row;
            if (row >= valid_rows || flat >= conv.output_h * conv.output_w) {
                std::memcpy(destination, &padding_word, 8);
                continue;
            }
            const int output_y = flat / conv.output_w;
            const int output_x = flat % conv.output_w;
            const int input_y = output_y * conv.stride_h + kernel_y - conv.pad_h;
            const int input_x = output_x * conv.stride_w + kernel_x - conv.pad_w;
            if (input_y < 0 || input_y >= conv.input_h || input_x < 0 || input_x >= conv.input_w) {
                std::memcpy(destination, &padding_word, 8);
            } else {
                const std::size_t offset = (((static_cast<std::size_t>(channel_block) * conv.input_h + input_y) *
                                              conv.input_w + input_x) * 8U);
                copy_c8(input + offset, destination);
            }
        }
    }
}

bool complete_group(const ConvAsset& conv, int m_begin, int group, int valid_rows,
                    int kernel_y, int kernel_x, int* input_y, int* input_x) noexcept {
    const int flat = m_begin + group;
    const int output_y = flat / conv.output_w;
    const int output_x = flat % conv.output_w;
    *input_y = output_y * conv.stride_h + kernel_y - conv.pad_h;
    *input_x = output_x * conv.stride_w + kernel_x - conv.pad_w;
    return group + 4 <= valid_rows && flat + 3 < conv.output_h * conv.output_w &&
           output_x + 3 < conv.output_w && *input_y >= 0 && *input_y < conv.input_h &&
           *input_x >= 0 && *input_x + 3 * conv.stride_w < conv.input_w;
}

void pack_a_generic(const ConvAsset& conv, const std::int8_t* input,
                    int m_begin, int rows, int valid_rows,
                    LoadStrategy strategy, std::int8_t* panel) {
    if (strategy == LoadStrategy::four_u64) {
        pack_a_scalar_positions(conv, input, m_begin, rows, valid_rows, panel);
        return;
    }
    if (conv.stride_w == 2 && conv.kernel_w == 3 &&
        (strategy == LoadStrategy::vlseg2_pair_vlse || strategy == LoadStrategy::vlseg2_pair_shift)) {
        const int channel_blocks = conv.input_c / 8;
        for (int kernel_y = 0; kernel_y < conv.kernel_h; ++kernel_y) {
            for (int channel_block = 0; channel_block < channel_blocks; ++channel_block) {
                const int tile0 = (kernel_y * conv.kernel_w) * channel_blocks + channel_block;
                const int tile1 = tile0 + channel_blocks;
                const int tile2 = tile1 + channel_blocks;
                for (int group = 0; group < rows; group += 4) {
                    int input_y = 0;
                    int input_x = 0;
                    const bool complete = complete_group(conv, m_begin, group, valid_rows,
                                                         kernel_y, 0, &input_y, &input_x) &&
                                          input_x + 8 < conv.input_w;
                    if (!complete) {
                        pack_a_scalar_positions(conv, input, m_begin, rows, valid_rows, panel);
                        return;
                    }
                    const std::int8_t* source = input + (((static_cast<std::size_t>(channel_block) * conv.input_h +
                        input_y) * conv.input_w + input_x) * 8U);
                    std::int8_t* destination0 = panel + (static_cast<std::size_t>(tile0) * rows + group) * 8U;
                    std::int8_t* destination1 = panel + (static_cast<std::size_t>(tile1) * rows + group) * 8U;
                    std::int8_t* destination2 = panel + (static_cast<std::size_t>(tile2) * rows + group) * 8U;
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
                    y26_stage49_load_vlseg2_pair_4(source, destination0, destination1);
#else
                    for (int row = 0; row < 4; ++row) {
                        copy_c8(source + row * 16, destination0 + row * 8);
                        copy_c8(source + row * 16 + 8, destination1 + row * 8);
                    }
#endif
                    if (strategy == LoadStrategy::vlseg2_pair_vlse) {
                        load_stride_group(source + 16, 2, LoadStrategy::vlse64, destination2);
                    } else {
                        std::memcpy(destination2, destination0 + 8, 24);
                        copy_c8(source + 64, destination2 + 24);
                    }
                }
            }
        }
        return;
    }

    const int channel_blocks = conv.input_c / 8;
    for (int tile = 0; tile < conv.k_tiles; ++tile) {
        const int channel_block = tile % channel_blocks;
        const int kernel_position = tile / channel_blocks;
        const int kernel_y = kernel_position / conv.kernel_w;
        const int kernel_x = kernel_position % conv.kernel_w;
        for (int group = 0; group < rows; group += 4) {
            int input_y = 0;
            int input_x = 0;
            const bool complete = complete_group(conv, m_begin, group, valid_rows,
                                                 kernel_y, kernel_x, &input_y, &input_x);
            const bool segment_safe = complete && input_x + 3 * conv.stride_w + 1 < conv.input_w;
            std::int8_t* destination = panel + (static_cast<std::size_t>(tile) * rows + group) * 8U;
            if (complete && (strategy != LoadStrategy::vlseg2_even || segment_safe)) {
                const std::int8_t* source = input + (((static_cast<std::size_t>(channel_block) * conv.input_h +
                    input_y) * conv.input_w + input_x) * 8U);
                load_stride_group(source, conv.stride_w, strategy, destination);
            } else {
                pack_a_scalar_positions(conv, input, m_begin, rows, valid_rows, panel);
                return;
            }
        }
    }
}

__attribute__((always_inline)) inline std::uint8_t requant_prevalidated(
    std::int64_t accumulator, std::int64_t multiplier, int right_shift, int output_zero_point) noexcept {
    __extension__ using Signed128 = __int128;
    __extension__ using Unsigned128 = unsigned __int128;
    const Signed128 product = static_cast<Signed128>(accumulator) * static_cast<Signed128>(multiplier);
    const bool negative = product < 0;
    const Unsigned128 bits = static_cast<Unsigned128>(product);
    const Unsigned128 absolute = negative ? (~bits) + 1U : bits;
    Unsigned128 quotient = absolute;
    if (right_shift != 0) {
        quotient = absolute >> right_shift;
        const Unsigned128 mask = (static_cast<Unsigned128>(1) << right_shift) - 1U;
        const Unsigned128 remainder = absolute & mask;
        const Unsigned128 half = static_cast<Unsigned128>(1) << (right_shift - 1);
        if (remainder > half || (remainder == half && (quotient & 1U) != 0)) ++quotient;
    }
    const Signed128 rounded = negative ? -static_cast<Signed128>(quotient) : static_cast<Signed128>(quotient);
    const Signed128 shifted = rounded + output_zero_point;
    if (shifted <= 0) return 0;
    if (shifted >= 255) return 255;
    return static_cast<std::uint8_t>(shifted);
}

__attribute__((always_inline)) inline std::int64_t round_q62_prevalidated(
    std::int64_t accumulator, std::int64_t multiplier, int right_shift) noexcept {
    __extension__ using Signed128 = __int128;
    __extension__ using Unsigned128 = unsigned __int128;
    const Signed128 product = static_cast<Signed128>(accumulator) * static_cast<Signed128>(multiplier);
    const bool negative = product < 0;
    const Unsigned128 bits = static_cast<Unsigned128>(product);
    const Unsigned128 absolute = negative ? (~bits) + 1U : bits;
    Unsigned128 quotient = absolute;
    if (right_shift != 0) {
        quotient = absolute >> right_shift;
        const Unsigned128 mask = (static_cast<Unsigned128>(1) << right_shift) - 1U;
        const Unsigned128 remainder = absolute & mask;
        const Unsigned128 half = static_cast<Unsigned128>(1) << (right_shift - 1);
        if (remainder > half || (remainder == half && (quotient & 1U) != 0)) ++quotient;
    }
    const auto rounded = static_cast<std::int64_t>(quotient);
    return negative ? -rounded : rounded;
}

__attribute__((always_inline)) inline std::uint8_t clamp_code(
    std::int64_t rounded, int output_zero_point) noexcept {
    const std::int64_t shifted = rounded + output_zero_point;
    if (shifted <= 0) return 0;
    if (shifted >= 255) return 255;
    return static_cast<std::uint8_t>(shifted);
}

struct EpilogueProfile {
    double extract_us = 0.0;
    double bias_us = 0.0;
    double multiply_rne_us = 0.0;
    double clamp_us = 0.0;
    double lut_us = 0.0;
    double store_us = 0.0;
};

struct ConvRunContext {
    const ConvAsset* conv = nullptr;
    WorkerPool* pool = nullptr;
    const std::int8_t* input = nullptr;
    std::array<std::int8_t*, 2> outputs {nullptr, nullptr};
    RunOptions options;
    std::array<int, kMaximumWorkers> status {};
    std::array<double, kMaximumWorkers> total_us {};
    std::array<double, kMaximumWorkers> delivery_us {};
    std::array<double, kMaximumWorkers> kernel_us {};
    std::array<double, kMaximumWorkers> epilogue_us {};
    std::array<EpilogueProfile, kMaximumWorkers> epilogue_profile {};
};

const Segment* find_segment(const ConvAsset& conv, int output_channel) noexcept {
    for (int index = 0; index < conv.segment_count; ++index) {
        const Segment& segment = conv.segments[static_cast<std::size_t>(index)];
        if (output_channel >= segment.channel_begin &&
            output_channel < segment.channel_begin + segment.channel_count) return &segment;
    }
    return nullptr;
}

void store_tile(const ConvAsset& conv, const std::array<std::int8_t*, 2>& outputs,
                const std::int32_t* accumulators, int rows, int valid_rows,
                int m_begin, int n_begin, EpilogueStrategy epilogue,
                EpilogueProfile* profile) {
    const int row_groups = rows / 4;
    const std::int64_t correction = 128 - static_cast<std::int64_t>(conv.input_zero_point);
    if (epilogue == EpilogueStrategy::inline_scalar) {
        const int output_m = conv.output_h * conv.output_w;
        for (int output_lane = 0; output_lane < kNBlock; ++output_lane) {
            const int output_channel = n_begin + output_lane;
            if (output_channel >= conv.output_c) continue;
            const Segment* segment = find_segment(conv, output_channel);
            if (segment == nullptr) continue;
            const int segment_channel = output_channel - segment->channel_begin;
            const int segment_index = segment == &conv.segments[0] ? 0 : 1;
            std::int8_t* output = outputs[static_cast<std::size_t>(segment_index)];
            const std::size_t channel_base = static_cast<std::size_t>(segment_channel / 8) * output_m * 8U;
            const std::size_t channel_inner = static_cast<std::size_t>(segment_channel % 8);
            const std::int64_t corrected_bias = conv.corrected_bias[static_cast<std::size_t>(output_channel)];
            const std::int64_t multiplier = conv.multiplier[static_cast<std::size_t>(output_channel)];
            const int shift = conv.shift[static_cast<std::size_t>(output_channel)];
            const int output_group = output_lane / 4;
            const int output_inner = output_lane % 4;
            for (int row = 0; row < valid_rows; ++row) {
                if (profile == nullptr) {
                    const int row_group = row / 4;
                    const int row_inner = row % 4;
                    const std::int32_t raw = accumulators[
                        (output_group * row_groups + row_group) * 16 + row_inner * 4 + output_inner];
                    const std::uint8_t conv_code = requant_prevalidated(
                        static_cast<std::int64_t>(raw) + corrected_bias, multiplier, shift,
                        conv.conv_output_zero_point);
                    output[channel_base + static_cast<std::size_t>(m_begin + row) * 8U + channel_inner] =
                        segment->lut[conv_code];
                    continue;
                }
                auto begin = Clock::now();
                const int row_group = row / 4;
                const int row_inner = row % 4;
                const std::int32_t raw = accumulators[
                    (output_group * row_groups + row_group) * 16 + row_inner * 4 + output_inner];
                auto end = Clock::now();
                profile->extract_us += elapsed_us(begin, end);
                begin = end;
                const std::int64_t corrected = static_cast<std::int64_t>(raw) + corrected_bias;
                end = Clock::now();
                profile->bias_us += elapsed_us(begin, end);
                begin = end;
                const std::int64_t rounded = round_q62_prevalidated(corrected, multiplier, shift);
                end = Clock::now();
                profile->multiply_rne_us += elapsed_us(begin, end);
                begin = end;
                const std::uint8_t conv_code = clamp_code(rounded, conv.conv_output_zero_point);
                end = Clock::now();
                profile->clamp_us += elapsed_us(begin, end);
                begin = end;
                const std::int8_t activated = segment->lut[conv_code];
                end = Clock::now();
                profile->lut_us += elapsed_us(begin, end);
                begin = end;
                output[channel_base + static_cast<std::size_t>(m_begin + row) * 8U + channel_inner] = activated;
                profile->store_us += elapsed_us(begin, Clock::now());
            }
        }
        return;
    }
    if (epilogue == EpilogueStrategy::rvv_q62) {
        const int output_m = conv.output_h * conv.output_w;
        for (int output_group = 0; output_group < 4; ++output_group) {
            const int output_channel = n_begin + output_group * 4;
            const Segment* segment = find_segment(conv, output_channel);
            if (segment == nullptr) continue;
            const int segment_channel = output_channel - segment->channel_begin;
            const int segment_index = segment == &conv.segments[0] ? 0 : 1;
            std::int8_t* output = outputs[static_cast<std::size_t>(segment_index)];
            const std::size_t channel_base = static_cast<std::size_t>(segment_channel / 8) * output_m * 8U;
            const std::size_t channel_inner = static_cast<std::size_t>(segment_channel % 8);
            alignas(32) std::array<std::int64_t, 4> corrected {};
            alignas(32) std::array<std::int64_t, 4> multiplier_m63 {};
            alignas(32) std::array<std::int64_t, 4> rounded {};
            for (int lane = 0; lane < 4; ++lane) {
                multiplier_m63[static_cast<std::size_t>(lane)] =
                    conv.multiplier_m63[static_cast<std::size_t>(output_channel + lane)];
            }
            for (int row = 0; row < valid_rows; ++row) {
                const auto extract_begin = profile != nullptr ? Clock::now() : Clock::time_point {};
                const int row_group = row / 4;
                const int row_inner = row % 4;
                const std::int32_t* raw = accumulators +
                    (output_group * row_groups + row_group) * 16 + row_inner * 4;
                const auto extract_end = profile != nullptr ? Clock::now() : Clock::time_point {};
                for (int lane = 0; lane < 4; ++lane) {
                    corrected[static_cast<std::size_t>(lane)] = static_cast<std::int64_t>(raw[lane]) +
                        conv.corrected_bias[static_cast<std::size_t>(output_channel + lane)];
                }
                const auto bias_end = profile != nullptr ? Clock::now() : Clock::time_point {};
                stage51::q62_vsmul_m63_i64x4(corrected.data(), multiplier_m63.data(), rounded.data());
                const auto multiply_end = profile != nullptr ? Clock::now() : Clock::time_point {};
                std::array<std::uint8_t, 4> codes {};
                for (int lane = 0; lane < 4; ++lane) {
                    codes[static_cast<std::size_t>(lane)] = clamp_code(
                        rounded[static_cast<std::size_t>(lane)], conv.conv_output_zero_point);
                }
                const auto clamp_end = profile != nullptr ? Clock::now() : Clock::time_point {};
                std::array<std::int8_t, 4> activated {};
                for (int lane = 0; lane < 4; ++lane) {
                    activated[static_cast<std::size_t>(lane)] = segment->lut[codes[static_cast<std::size_t>(lane)]];
                }
                const auto lut_end = profile != nullptr ? Clock::now() : Clock::time_point {};
                std::memcpy(output + channel_base + static_cast<std::size_t>(m_begin + row) * 8U + channel_inner,
                            activated.data(), activated.size());
                if (profile != nullptr) {
                    const auto store_end = Clock::now();
                    profile->extract_us += elapsed_us(extract_begin, extract_end);
                    profile->bias_us += elapsed_us(extract_end, bias_end);
                    profile->multiply_rne_us += elapsed_us(bias_end, multiply_end);
                    profile->clamp_us += elapsed_us(multiply_end, clamp_end);
                    profile->lut_us += elapsed_us(clamp_end, lut_end);
                    profile->store_us += elapsed_us(lut_end, store_end);
                }
            }
        }
        return;
    }
    for (int row = 0; row < valid_rows; ++row) {
        const int flat = m_begin + row;
        const int output_y = flat / conv.output_w;
        const int output_x = flat % conv.output_w;
        for (int output_lane = 0; output_lane < kNBlock; ++output_lane) {
            const int output_channel = n_begin + output_lane;
            if (output_channel >= conv.output_c) continue;
            const Segment* segment = find_segment(conv, output_channel);
            if (segment == nullptr) continue;
            const int output_group = output_lane / 4;
            const int output_inner = output_lane % 4;
            const int row_group = row / 4;
            const int row_inner = row % 4;
            const std::int32_t raw = accumulators[
                (output_group * row_groups + row_group) * 16 + row_inner * 4 + output_inner];
            std::uint8_t conv_code = 0;
            const std::int64_t corrected = static_cast<std::int64_t>(raw) +
                correction * conv.weight_sums[static_cast<std::size_t>(output_channel)] +
                conv.bias[static_cast<std::size_t>(output_channel)];
            const int8_v1::RequantAsset asset {
                conv.multiplier[static_cast<std::size_t>(output_channel)],
                conv.shift[static_cast<std::size_t>(output_channel)], conv.conv_output_zero_point, 0, 255};
            if (!int8_v1::requantize_u8(corrected, asset, &conv_code)) continue;
            const int segment_channel = output_channel - segment->channel_begin;
            const std::size_t offset = nchwc8_offset(segment_channel, output_y, output_x,
                                                     conv.output_h, conv.output_w);
            const int segment_index = segment == &conv.segments[0] ? 0 : 1;
            outputs[static_cast<std::size_t>(segment_index)][offset] = segment->lut[conv_code];
        }
    }
}

void run_conv_worker(void* opaque, int worker_index, WorkerScratch& scratch) {
    auto& context = *static_cast<ConvRunContext*>(opaque);
    const ConvAsset& conv = *context.conv;
    const int rows = rows_for(context.options.kernel);
    const int total_m = conv.output_h * conv.output_w;
    const int spatial_tiles = (total_m + rows - 1) / rows;
    int tile_begin = 0;
    int tile_end = spatial_tiles;
    int n_begin = 0;
    int n_end = conv.n_blocks;
    if (context.options.partition == PartitionPolicy::spatial) {
        tile_begin = spatial_tiles * worker_index / context.options.workers;
        tile_end = spatial_tiles * (worker_index + 1) / context.options.workers;
    } else {
        n_begin = conv.n_blocks * worker_index / context.options.workers;
        n_end = conv.n_blocks * (worker_index + 1) / context.options.workers;
    }
    stage51::VectorFixedPointState vector_state;
    if (context.options.epilogue == EpilogueStrategy::rvv_q62 &&
        !stage51::begin_q62_vector_rne(&vector_state)) {
        context.status[static_cast<std::size_t>(worker_index)] = Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
        return;
    }
    const auto worker_begin = Clock::now();
    for (int tile = tile_begin; tile < tile_end; ++tile) {
        const int m_begin = tile * rows;
        const int valid_rows = std::min(rows, total_m - m_begin);
        const auto delivery_begin = context.options.profile_phases ? Clock::now() : worker_begin;
        pack_a_generic(conv, context.input, m_begin, rows, valid_rows,
                       context.options.load, scratch.a_panel.data());
        if (context.options.profile_phases) {
            context.delivery_us[static_cast<std::size_t>(worker_index)] += elapsed_us(delivery_begin, Clock::now());
        }
        for (int n_block = n_begin; n_block < n_end; ++n_block) {
            const auto kernel_begin = context.options.profile_phases ? Clock::now() : worker_begin;
            const std::int8_t* weights = conv.packed_weights.data() +
                static_cast<std::size_t>(n_block) * conv.k_tiles * kNBlock * 8U;
            run_block(context.options.route, context.options.kernel, scratch.a_panel.data(), weights,
                      conv.k_tiles, scratch.c_tile.data());
            if (context.options.profile_phases) {
                context.kernel_us[static_cast<std::size_t>(worker_index)] += elapsed_us(kernel_begin, Clock::now());
            }
            const auto epilogue_begin = context.options.profile_phases ? Clock::now() : worker_begin;
            store_tile(conv, context.outputs, scratch.c_tile.data(), rows, valid_rows,
                       m_begin, n_block * kNBlock, context.options.epilogue,
                       context.options.profile_phases
                           ? &context.epilogue_profile[static_cast<std::size_t>(worker_index)] : nullptr);
            if (context.options.profile_phases) {
                context.epilogue_us[static_cast<std::size_t>(worker_index)] += elapsed_us(epilogue_begin, Clock::now());
            }
        }
    }
    context.total_us[static_cast<std::size_t>(worker_index)] = elapsed_us(worker_begin, Clock::now());
    bool vector_ok = true;
    if (context.options.epilogue == EpilogueStrategy::rvv_q62) {
        const auto result = stage51::end_q62_vector_rne(&vector_state);
        vector_ok = result.restored && !result.saturated;
    }
    context.status[static_cast<std::size_t>(worker_index)] =
        current_cpu() >= 0 && current_cpu() <= 3 && vector_ok
            ? Y26_CONV_STATUS_SUCCESS : Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
}

int run_conv(WorkerPool& pool, const ConvAsset& conv, const std::int8_t* input,
             const std::array<std::int8_t*, 2>& outputs, const RunOptions& options,
             OperationTiming* timing, double* min_worker, double* max_worker) {
    if (input == nullptr || outputs[0] == nullptr || options.workers < 1 ||
        options.workers > pool.capacity() ||
        (options.epilogue == EpilogueStrategy::rvv_q62 && !conv.vsmul_q62_compatible)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    for (int segment = 0; segment < conv.segment_count; ++segment) {
        if (outputs[static_cast<std::size_t>(segment)] == nullptr) return Y26_CONV_STATUS_INVALID_ARGUMENT;
        if (int8_v1::ranges_overlap(input, static_cast<std::size_t>(conv.input_h) * conv.input_w * conv.input_c,
                                    outputs[static_cast<std::size_t>(segment)],
                                    static_cast<std::size_t>(conv.output_h) * conv.output_w *
                                        conv.segments[static_cast<std::size_t>(segment)].channel_count)) {
            return Y26_CONV_STATUS_INVALID_ARGUMENT;
        }
    }
    ConvRunContext context;
    context.conv = &conv;
    context.pool = &pool;
    context.input = input;
    context.outputs = outputs;
    context.options = options;
    const auto begin = Clock::now();
    pool.dispatch(options.workers, run_conv_worker, &context, options.scheduler);
    const double wall = elapsed_us(begin, Clock::now());
    for (int worker = 0; worker < options.workers; ++worker) {
        if (context.status[static_cast<std::size_t>(worker)] != Y26_CONV_STATUS_SUCCESS) {
            return context.status[static_cast<std::size_t>(worker)];
        }
    }
    if (timing != nullptr) {
        timing->wall_us = wall;
        for (int worker = 0; worker < options.workers; ++worker) {
            timing->delivery_worker_sum_us += context.delivery_us[static_cast<std::size_t>(worker)];
            timing->vmadot_worker_sum_us += context.kernel_us[static_cast<std::size_t>(worker)];
            timing->epilogue_worker_sum_us += context.epilogue_us[static_cast<std::size_t>(worker)];
            const auto& phases = context.epilogue_profile[static_cast<std::size_t>(worker)];
            timing->epilogue_extract_worker_sum_us += phases.extract_us;
            timing->epilogue_bias_worker_sum_us += phases.bias_us;
            timing->epilogue_multiply_rne_worker_sum_us += phases.multiply_rne_us;
            timing->epilogue_clamp_worker_sum_us += phases.clamp_us;
            timing->epilogue_lut_worker_sum_us += phases.lut_us;
            timing->epilogue_store_worker_sum_us += phases.store_us;
        }
    }
    if (min_worker != nullptr && max_worker != nullptr) {
        *min_worker = *std::min_element(context.total_us.begin(), context.total_us.begin() + options.workers);
        *max_worker = *std::max_element(context.total_us.begin(), context.total_us.begin() + options.workers);
    }
    return Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

struct PersistentSlice::Impl {
    std::filesystem::path package;
    std::string trusted_manifest;
    std::vector<Tensor> tensors;
    std::vector<Operation> operations;
    std::vector<std::int8_t> arena;
    std::int8_t* external_arena = nullptr;
    std::size_t external_arena_bytes = 0;
    std::vector<std::int8_t*> external_tensors;
    std::vector<std::vector<std::int8_t>> captured_tensors;
    std::unique_ptr<WorkerPool> pool;
    std::unique_ptr<WorkerPool> cluster1_pool;
    std::size_t total_packed_weight_bytes = 0;
    int model4_preact_id = 0;
    int model4_postact_id = 1;
    int model5_output_id = 2;
    int model6_output_id = -1;
    int slice_output_id = -1;
    std::string error;
    bool ready = false;

    const Tensor& tensor(int id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= tensors.size() || tensors[static_cast<std::size_t>(id)].id != id) {
            throw std::runtime_error("invalid tensor id");
        }
        return tensors[static_cast<std::size_t>(id)];
    }

    std::int8_t* data(int id) {
        const Tensor& descriptor = tensor(id);
        if (external_tensors[static_cast<std::size_t>(id)] != nullptr) {
            return external_tensors[static_cast<std::size_t>(id)];
        }
        std::int8_t* base = external_arena != nullptr ? external_arena : arena.data();
        return base + descriptor.offset;
    }
    const std::int8_t* data(int id) const {
        const Tensor& descriptor = tensor(id);
        if (external_tensors[static_cast<std::size_t>(id)] != nullptr) {
            return external_tensors[static_cast<std::size_t>(id)];
        }
        const std::int8_t* base = external_arena != nullptr ? external_arena : arena.data();
        return base + descriptor.offset;
    }

    int tensor_id_for_key(std::string_view key) const noexcept {
        const auto found = std::find_if(tensors.begin(), tensors.end(), [&](const Tensor& tensor) {
            return tensor.key == key;
        });
        return found == tensors.end() ? -1 : found->id;
    }
};

namespace {

void validate_conv(ConvAsset& conv) {
    if (conv.input_h <= 0 || conv.input_w <= 0 || conv.input_c <= 0 || conv.input_c % 8 != 0 ||
        conv.output_h <= 0 || conv.output_w <= 0 || conv.output_c <= 0 || conv.output_c % 16 != 0 ||
        conv.kernel_h <= 0 || conv.kernel_w <= 0 || conv.stride_h <= 0 || conv.stride_w <= 0 ||
        conv.k != conv.kernel_h * conv.kernel_w * conv.input_c || conv.k % 8 != 0 ||
        conv.k_tiles != conv.k / 8 || conv.n_blocks != conv.output_c / 16 ||
        conv.weight_sums.size() != static_cast<std::size_t>(conv.output_c) ||
        conv.bias.size() != static_cast<std::size_t>(conv.output_c) ||
        conv.multiplier.size() != static_cast<std::size_t>(conv.output_c) ||
        conv.shift.size() != static_cast<std::size_t>(conv.output_c)) {
        throw std::runtime_error("invalid Conv package contract");
    }
    std::vector<std::int32_t> recomputed_sums(static_cast<std::size_t>(conv.output_c));
    int maximum_weight = 0;
    for (int output_channel = 0; output_channel < conv.output_c; ++output_channel) {
        const int block = output_channel / 16;
        const int lane = output_channel % 16;
        std::int64_t sum = 0;
        for (int tile = 0; tile < conv.k_tiles; ++tile) {
            const std::size_t base = (((static_cast<std::size_t>(block) * conv.k_tiles + tile) * 16U + lane) * 8U);
            for (int inner = 0; inner < 8; ++inner) {
                const int weight = conv.packed_weights[base + static_cast<std::size_t>(inner)];
                sum += weight;
                maximum_weight = std::max(maximum_weight, std::abs(weight));
            }
        }
        if (sum < std::numeric_limits<std::int32_t>::min() || sum > std::numeric_limits<std::int32_t>::max()) {
            throw std::runtime_error("weight sum overflow");
        }
        recomputed_sums[static_cast<std::size_t>(output_channel)] = static_cast<std::int32_t>(sum);
    }
    if (recomputed_sums != conv.weight_sums) throw std::runtime_error("packaged weight sums mismatch loaded weights");
    std::uint64_t maximum_bias = 0;
    for (std::int32_t value : conv.bias) {
        const auto magnitude = value < 0 ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(value))
                                         : static_cast<std::uint64_t>(value);
        maximum_bias = std::max(maximum_bias, magnitude);
    }
    const auto safety = int8_v1::accumulator_safety_bound(
        static_cast<std::size_t>(conv.k), static_cast<std::uint8_t>(conv.input_zero_point),
        static_cast<std::uint8_t>(maximum_weight), maximum_bias);
    if (!safety.valid || !safety.int32_safe || safety.absolute_bound != conv.accumulator_bound) {
        throw std::runtime_error("recomputed accumulator bound mismatch or unsafe Conv");
    }
    conv.corrected_bias.resize(static_cast<std::size_t>(conv.output_c));
    conv.multiplier_m63.resize(static_cast<std::size_t>(conv.output_c));
    conv.vsmul_q62_compatible = true;
    const std::int64_t correction = 128 - static_cast<std::int64_t>(conv.input_zero_point);
    for (int channel = 0; channel < conv.output_c; ++channel) {
        if (conv.multiplier[static_cast<std::size_t>(channel)] < 0 ||
            conv.shift[static_cast<std::size_t>(channel)] < 0 ||
            conv.shift[static_cast<std::size_t>(channel)] > 126) {
            throw std::runtime_error("invalid prevalidated requant asset");
        }
        conv.corrected_bias[static_cast<std::size_t>(channel)] = conv.bias[static_cast<std::size_t>(channel)] +
            correction * conv.weight_sums[static_cast<std::size_t>(channel)];
        const std::int64_t multiplier = conv.multiplier[static_cast<std::size_t>(channel)];
        if (conv.shift[static_cast<std::size_t>(channel)] != 62 || multiplier <= 0 ||
            multiplier > std::numeric_limits<std::int64_t>::max() / 2) {
            conv.vsmul_q62_compatible = false;
            conv.multiplier_m63[static_cast<std::size_t>(channel)] = 0;
        } else {
            conv.multiplier_m63[static_cast<std::size_t>(channel)] = multiplier * 2;
        }
    }
}

void transform_lut_explicit_rvv(const std::int8_t* source, std::int8_t* destination,
                                const std::int8_t* lut, std::size_t count) noexcept {
#if defined(Y26_K1X_ENABLE_IME_ASM) && defined(__riscv)
    __asm__ volatile(
        "li t1, 128\n\t"
        "1:\n\t"
        "beqz %[count], 2f\n\t"
        "vsetvli t0, %[count], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[source])\n\t"
        "vxor.vx v0, v0, t1\n\t"
        "vluxei8.v v1, (%[lut]), v0\n\t"
        "vse8.v v1, (%[destination])\n\t"
        "add %[source], %[source], t0\n\t"
        "add %[destination], %[destination], t0\n\t"
        "sub %[count], %[count], t0\n\t"
        "j 1b\n\t"
        "2:\n\t"
        : [source] "+r"(source), [destination] "+r"(destination), [count] "+r"(count)
        : [lut] "r"(lut)
        : "memory", "t0", "t1", "v0", "v1");
#else
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] = lut[int8_v1::semantic_code(source[index])];
    }
#endif
}

bool identity_lut(const std::int8_t* lut) noexcept {
    for (int code = 0; code < 256; ++code) {
        if (lut[code] != int8_v1::signed_storage(static_cast<std::uint8_t>(code))) return false;
    }
    return true;
}

struct LutRunContext {
    const std::int8_t* source = nullptr;
    std::int8_t* destination = nullptr;
    const std::int8_t* lut = nullptr;
    std::size_t bytes = 0;
    int workers = 1;
    NonConvStrategy strategy = NonConvStrategy::parallel_scalar;
};

WorkerPool& nonconv_pool(PersistentSlice::Impl& executor, const RunOptions& options) {
    if (options.nonconv == NonConvStrategy::cluster1_explicit_rvv) {
        if (!executor.cluster1_pool) throw std::runtime_error("cluster1 pool is unavailable");
        return *executor.cluster1_pool;
    }
    return *executor.pool;
}

int nonconv_workers(const WorkerPool& pool, const RunOptions& options) noexcept {
    return std::min(options.workers, pool.capacity());
}

NonConvStrategy local_nonconv_strategy(NonConvStrategy strategy) noexcept {
    return strategy == NonConvStrategy::cluster1_explicit_rvv
        ? NonConvStrategy::explicit_rvv_lut : strategy;
}

void run_lut_worker(void* opaque, int worker_index, WorkerScratch&) {
    auto& context = *static_cast<LutRunContext*>(opaque);
    const std::size_t begin = context.bytes * static_cast<std::size_t>(worker_index) /
                              static_cast<std::size_t>(context.workers);
    const std::size_t end = context.bytes * static_cast<std::size_t>(worker_index + 1) /
                            static_cast<std::size_t>(context.workers);
    if (context.strategy == NonConvStrategy::explicit_rvv_lut) {
        transform_lut_explicit_rvv(context.source + begin, context.destination + begin,
                                   context.lut, end - begin);
        return;
    }
    for (std::size_t index = begin; index < end; ++index) {
        context.destination[index] = context.lut[int8_v1::semantic_code(context.source[index])];
    }
}

void run_lut(PersistentSlice::Impl& executor, const Operation& operation,
             const RunOptions& options) {
    const Tensor& input = executor.tensor(operation.inputs[0]);
    const Tensor& output = executor.tensor(operation.outputs[0]);
    const std::int8_t* source = executor.data(operation.inputs[0]);
    std::int8_t* destination = executor.data(operation.outputs[0]);
    if (input.bytes != output.bytes ||
        int8_v1::ranges_overlap(source, input.bytes, destination, output.bytes)) {
        throw std::runtime_error("LUT input/output contract failure");
    }
    if (options.nonconv == NonConvStrategy::serial_scalar) {
        for (std::size_t index = 0; index < output.bytes; ++index) {
            destination[index] = operation.lut[int8_v1::semantic_code(source[index])];
        }
        return;
    }
    WorkerPool& pool = nonconv_pool(executor, options);
    const int workers = nonconv_workers(pool, options);
    LutRunContext context {source, destination, operation.lut.data(), output.bytes,
                           workers, local_nonconv_strategy(options.nonconv)};
    pool.dispatch(workers, run_lut_worker, &context, options.scheduler);
}

struct AddRunContext {
    const std::int8_t* left = nullptr;
    const std::int8_t* right = nullptr;
    std::int8_t* destination = nullptr;
    const std::int8_t* lut = nullptr;
    std::size_t bytes = 0;
    int workers = 1;
};

void run_add_worker(void* opaque, int worker_index, WorkerScratch&) {
    auto& context = *static_cast<AddRunContext*>(opaque);
    const std::size_t begin = context.bytes * static_cast<std::size_t>(worker_index) /
                              static_cast<std::size_t>(context.workers);
    const std::size_t end = context.bytes * static_cast<std::size_t>(worker_index + 1) /
                            static_cast<std::size_t>(context.workers);
    for (std::size_t index = begin; index < end; ++index) {
        context.destination[index] = context.lut[
            static_cast<std::size_t>(int8_v1::semantic_code(context.left[index])) * 256U +
            int8_v1::semantic_code(context.right[index])];
    }
}

void run_add(PersistentSlice::Impl& executor, const Operation& operation,
             const RunOptions& options) {
    const Tensor& output = executor.tensor(operation.outputs[0]);
    const Tensor& left_tensor = executor.tensor(operation.inputs[0]);
    const Tensor& right_tensor = executor.tensor(operation.inputs[1]);
    if (left_tensor.bytes != output.bytes || right_tensor.bytes != output.bytes) {
        throw std::runtime_error("Add shape mismatch");
    }
    const std::int8_t* left = executor.data(operation.inputs[0]);
    const std::int8_t* right = executor.data(operation.inputs[1]);
    std::int8_t* destination = executor.data(operation.outputs[0]);
    if (int8_v1::ranges_overlap(left, left_tensor.bytes, destination, output.bytes) ||
        int8_v1::ranges_overlap(right, right_tensor.bytes, destination, output.bytes)) {
        throw std::runtime_error("Add input/output overlap");
    }
    if (options.nonconv == NonConvStrategy::serial_scalar) {
        for (std::size_t index = 0; index < output.bytes; ++index) {
            destination[index] = operation.add_lut[
                static_cast<std::size_t>(int8_v1::semantic_code(left[index])) * 256U +
                int8_v1::semantic_code(right[index])];
        }
        return;
    }
    WorkerPool& pool = nonconv_pool(executor, options);
    const int workers = nonconv_workers(pool, options);
    AddRunContext context {left, right, destination, operation.add_lut.data(), output.bytes,
                           workers};
    pool.dispatch(workers, run_add_worker, &context, options.scheduler);
}

struct ConcatRunContext {
    const std::int8_t* source = nullptr;
    std::int8_t* destination = nullptr;
    const std::int8_t* lut = nullptr;
    std::size_t bytes = 0;
    int workers = 1;
    NonConvStrategy strategy = NonConvStrategy::parallel_scalar;
};

void run_concat_worker(void* opaque, int worker_index, WorkerScratch&) {
    auto& context = *static_cast<ConcatRunContext*>(opaque);
    const std::size_t begin = context.bytes * static_cast<std::size_t>(worker_index) /
                              static_cast<std::size_t>(context.workers);
    const std::size_t end = context.bytes * static_cast<std::size_t>(worker_index + 1) /
                            static_cast<std::size_t>(context.workers);
    if (context.strategy == NonConvStrategy::explicit_rvv_lut) {
        transform_lut_explicit_rvv(context.source + begin, context.destination + begin,
                                   context.lut, end - begin);
        return;
    }
    for (std::size_t index = begin; index < end; ++index) {
        context.destination[index] = context.lut[int8_v1::semantic_code(context.source[index])];
    }
}

void run_concat(PersistentSlice::Impl& executor, const Operation& operation,
                const RunOptions& options) {
    const Tensor& output = executor.tensor(operation.outputs[0]);
    std::int8_t* destination = executor.data(operation.outputs[0]);
    const std::size_t pixels = static_cast<std::size_t>(output.h) * output.w;
    int output_block = 0;
    for (std::size_t input_index = 0; input_index < operation.inputs.size(); ++input_index) {
        const int tensor_id = operation.inputs[input_index];
        if (tensor_id < 0) continue;
        const Tensor& input = executor.tensor(tensor_id);
        if (input.h != output.h || input.w != output.w || input.c % 8 != 0) {
            throw std::runtime_error("Concat tensor mismatch");
        }
        const std::int8_t* source = executor.data(tensor_id);
        std::int8_t* segment_destination = destination + static_cast<std::size_t>(output_block) * pixels * 8U;
        const bool identity = identity_lut(operation.concat_lut[input_index].data());
        if (source == segment_destination && identity) {
            output_block += input.c / 8;
            continue;
        }
        if (int8_v1::ranges_overlap(source, input.bytes, destination, output.bytes)) {
            throw std::runtime_error("Concat input/output overlap");
        }
        if (identity && options.nonconv == NonConvStrategy::serial_scalar) {
            std::memcpy(segment_destination, source, input.bytes);
        } else if (options.nonconv == NonConvStrategy::serial_scalar) {
            for (std::size_t index = 0; index < input.bytes; ++index) {
                segment_destination[index] = operation.concat_lut[input_index][
                    int8_v1::semantic_code(source[index])];
            }
        } else {
            WorkerPool& pool = nonconv_pool(executor, options);
            const int workers = nonconv_workers(pool, options);
            ConcatRunContext context {source, segment_destination,
                operation.concat_lut[input_index].data(), input.bytes, workers,
                local_nonconv_strategy(options.nonconv)};
            pool.dispatch(workers, run_concat_worker, &context, options.scheduler);
        }
        output_block += input.c / 8;
    }
    if (output_block != output.c / 8) throw std::runtime_error("Concat channel count mismatch");
}

struct MaxPoolRunContext {
    const std::int8_t* source = nullptr;
    std::int8_t* destination = nullptr;
    int input_h = 0;
    int input_w = 0;
    int output_h = 0;
    int output_w = 0;
    int channel_blocks = 0;
    int kernel_h = 0;
    int kernel_w = 0;
    int stride_h = 0;
    int stride_w = 0;
    int pad_h = 0;
    int pad_w = 0;
    int workers = 1;
};

void maxpool_c8(const MaxPoolRunContext& context, int channel_block,
                int output_y, int output_x) noexcept {
    const int input_y_origin = output_y * context.stride_h - context.pad_h;
    const int input_x_origin = output_x * context.stride_w - context.pad_w;
    const std::size_t destination_offset =
        (((static_cast<std::size_t>(channel_block) * context.output_h + output_y) *
          context.output_w + output_x) * 8U);
#if defined(__riscv_vector)
    constexpr std::size_t lanes = 8;
    vuint8m1_t maximum = __riscv_vmv_v_x_u8m1(0, lanes);
    for (int kernel_y = 0; kernel_y < context.kernel_h; ++kernel_y) {
        const int input_y = input_y_origin + kernel_y;
        if (input_y < 0 || input_y >= context.input_h) continue;
        for (int kernel_x = 0; kernel_x < context.kernel_w; ++kernel_x) {
            const int input_x = input_x_origin + kernel_x;
            if (input_x < 0 || input_x >= context.input_w) continue;
            const std::size_t source_offset =
                (((static_cast<std::size_t>(channel_block) * context.input_h + input_y) *
                  context.input_w + input_x) * 8U);
            vuint8m1_t value = __riscv_vle8_v_u8m1(
                reinterpret_cast<const std::uint8_t*>(context.source + source_offset), lanes);
            value = __riscv_vxor_vx_u8m1(value, 0x80U, lanes);
            maximum = __riscv_vmaxu_vv_u8m1(maximum, value, lanes);
        }
    }
    maximum = __riscv_vxor_vx_u8m1(maximum, 0x80U, lanes);
    __riscv_vse8_v_u8m1(
        reinterpret_cast<std::uint8_t*>(context.destination + destination_offset), maximum, lanes);
#else
    std::array<std::uint8_t, 8> maximum {};
    for (int kernel_y = 0; kernel_y < context.kernel_h; ++kernel_y) {
        const int input_y = input_y_origin + kernel_y;
        if (input_y < 0 || input_y >= context.input_h) continue;
        for (int kernel_x = 0; kernel_x < context.kernel_w; ++kernel_x) {
            const int input_x = input_x_origin + kernel_x;
            if (input_x < 0 || input_x >= context.input_w) continue;
            const std::size_t source_offset =
                (((static_cast<std::size_t>(channel_block) * context.input_h + input_y) *
                  context.input_w + input_x) * 8U);
            for (int lane = 0; lane < 8; ++lane) {
                maximum[static_cast<std::size_t>(lane)] = std::max(
                    maximum[static_cast<std::size_t>(lane)],
                    int8_v1::semantic_code(context.source[source_offset + static_cast<std::size_t>(lane)]));
            }
        }
    }
    for (int lane = 0; lane < 8; ++lane) {
        context.destination[destination_offset + static_cast<std::size_t>(lane)] =
            int8_v1::signed_storage(maximum[static_cast<std::size_t>(lane)]);
    }
#endif
}

void run_maxpool_worker(void* opaque, int worker_index, WorkerScratch&) {
    const auto& context = *static_cast<MaxPoolRunContext*>(opaque);
    const int begin = context.channel_blocks * worker_index / context.workers;
    const int end = context.channel_blocks * (worker_index + 1) / context.workers;
    for (int channel_block = begin; channel_block < end; ++channel_block) {
        for (int output_y = 0; output_y < context.output_h; ++output_y) {
            for (int output_x = 0; output_x < context.output_w; ++output_x) {
                maxpool_c8(context, channel_block, output_y, output_x);
            }
        }
    }
}

void run_maxpool(PersistentSlice::Impl& executor, const Operation& operation,
                 const RunOptions& options) {
    const Tensor& input = executor.tensor(operation.inputs[0]);
    const Tensor& output = executor.tensor(operation.outputs[0]);
    const std::int8_t* source = executor.data(operation.inputs[0]);
    std::int8_t* destination = executor.data(operation.outputs[0]);
    if (int8_v1::ranges_overlap(source, input.bytes, destination, output.bytes)) {
        throw std::runtime_error("MaxPool input/output overlap");
    }
    WorkerPool& pool = nonconv_pool(executor, options);
    const int workers = nonconv_workers(pool, options);
    MaxPoolRunContext context {
        source, destination, input.h, input.w, output.h, output.w, input.c / 8,
        operation.pool_kernel_h, operation.pool_kernel_w,
        operation.pool_stride_h, operation.pool_stride_w,
        operation.pool_pad_h, operation.pool_pad_w, workers,
    };
    pool.dispatch(workers, run_maxpool_worker, &context, options.scheduler);
}

}  // namespace

PersistentSlice::PersistentSlice() : impl_(std::make_unique<Impl>()) {}
PersistentSlice::~PersistentSlice() = default;
PersistentSlice::PersistentSlice(PersistentSlice&&) noexcept = default;
PersistentSlice& PersistentSlice::operator=(PersistentSlice&&) noexcept = default;

int PersistentSlice::prepare(const std::filesystem::path& package_dir,
                             const std::string& trusted_manifest_sha256,
                             int worker_capacity) {
    return prepare_with_contract(package_dir, trusted_manifest_sha256, worker_capacity,
                                 int8_v1::kContractId, int8_v1::kGeneralProfile);
}

int PersistentSlice::prepare_with_contract(const std::filesystem::path& package_dir,
                                           const std::string& trusted_manifest_sha256,
                                           int worker_capacity,
                                           const std::string& expected_contract_id,
                                           const std::string& expected_profile_id,
                                           bool enable_cluster1_pool,
                                           WorkerWakePolicy wake_policy) {
    if (worker_capacity < 1 || worker_capacity > kMaximumWorkers) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    impl_->error.clear();
    try {
        const auto integrity = int8_v1::verify_package(package_dir, trusted_manifest_sha256,
            expected_contract_id, expected_profile_id, int8_v1::kNchwc8LayoutId);
        if (!integrity.ok) throw std::runtime_error("package integrity failed: " + integrity.error);
        Impl prepared;
        prepared.package = std::filesystem::canonical(package_dir);
        prepared.trusted_manifest = integrity.manifest_sha256;
        std::size_t required_arena = 0;
        for (const Row& row : read_tsv(prepared.package / "tensors.tsv")) {
            Tensor tensor;
            tensor.id = integer(row, "id");
            tensor.key = field(row, "key");
            tensor.name = field(row, "logical_name");
            tensor.h = integer(row, "h");
            tensor.w = integer(row, "w");
            tensor.c = integer(row, "c");
            tensor.zero_point = integer(row, "zero_point");
            tensor.offset = size_value(row, "arena_offset");
            tensor.bytes = size_value(row, "bytes");
            const std::size_t expected = static_cast<std::size_t>(tensor.h) * tensor.w * tensor.c;
            if (tensor.id != static_cast<int>(prepared.tensors.size()) || tensor.h <= 0 || tensor.w <= 0 ||
                tensor.c <= 0 || tensor.c % 8 != 0 || tensor.bytes != expected ||
                field(row, "physical_layout") != int8_v1::kNchwc8LayoutId ||
                tensor.offset > std::numeric_limits<std::size_t>::max() - tensor.bytes) {
                throw std::runtime_error("invalid tensor descriptor");
            }
            required_arena = std::max(required_arena, tensor.offset + tensor.bytes);
            prepared.tensors.push_back(std::move(tensor));
        }
        prepared.arena.resize(required_arena);
        prepared.external_tensors.resize(prepared.tensors.size(), nullptr);
        std::size_t maximum_panel = 0;
        for (const Row& row : read_tsv(prepared.package / "operations.tsv")) {
            Operation operation;
            operation.index = integer(row, "index");
            operation.kind = field(row, "kind");
            operation.name = field(row, "name");
            operation.inputs = {integer(row, "input0"), integer(row, "input1"), integer(row, "input2"),
                                optional_integer(row, "input3", -1)};
            operation.outputs = {integer(row, "output0"), integer(row, "output1")};
            if (operation.index != static_cast<int>(prepared.operations.size())) {
                throw std::runtime_error("operation index mismatch");
            }
            if (operation.kind == "conv") {
                ConvAsset& conv = operation.conv;
                const Tensor& input = prepared.tensor(operation.inputs[0]);
                const Tensor& output = prepared.tensor(operation.outputs[0]);
                conv.input_h = input.h;
                conv.input_w = input.w;
                conv.input_c = input.c;
                conv.output_h = output.h;
                conv.output_w = output.w;
                conv.output_c = integer(row, "output_c");
                conv.kernel_h = integer(row, "kernel_h");
                conv.kernel_w = integer(row, "kernel_w");
                conv.stride_h = integer(row, "stride_h");
                conv.stride_w = integer(row, "stride_w");
                conv.pad_h = integer(row, "pad_h");
                conv.pad_w = integer(row, "pad_w");
                conv.input_zero_point = input.zero_point;
                conv.conv_output_zero_point = integer(row, "conv_output_zero_point");
                conv.k = integer(row, "k");
                conv.k_tiles = integer(row, "k_tiles");
                conv.n_blocks = integer(row, "n_blocks");
                conv.accumulator_bound = size_value(row, "accumulator_absolute_bound");
                conv.packed_weights = read_binary<std::int8_t>(prepared.package / field(row, "packed_weights_file"),
                    static_cast<std::size_t>(conv.n_blocks) * conv.k_tiles * 16U * 8U);
                conv.weight_sums = read_binary<std::int32_t>(prepared.package / field(row, "weight_sums_file"), conv.output_c);
                conv.bias = read_binary<std::int32_t>(prepared.package / field(row, "bias_file"), conv.output_c);
                conv.multiplier = read_binary<std::int64_t>(prepared.package / field(row, "requant_multiplier_file"), conv.output_c);
                conv.shift = read_binary<std::int32_t>(prepared.package / field(row, "requant_shift_file"), conv.output_c);
                for (int segment_index = 0; segment_index < 2; ++segment_index) {
                    const std::string suffix = std::to_string(segment_index);
                    const int count = integer(row, ("segment" + suffix + "_count").c_str());
                    if (count <= 0) continue;
                    Segment& segment = conv.segments[static_cast<std::size_t>(conv.segment_count++)];
                    segment.channel_begin = integer(row, ("segment" + suffix + "_begin").c_str());
                    segment.channel_count = count;
                    segment.tensor_id = operation.outputs[static_cast<std::size_t>(segment_index)];
                    const auto lut = read_binary<std::int8_t>(prepared.package /
                        field(row, ("segment" + suffix + "_lut_file").c_str()), 256);
                    std::copy(lut.begin(), lut.end(), segment.lut.begin());
                }
                validate_conv(conv);
                maximum_panel = std::max(maximum_panel, static_cast<std::size_t>(12) * conv.k);
                prepared.total_packed_weight_bytes += conv.packed_weights.size();
            } else if (operation.kind == "lut") {
                const auto lut = read_binary<std::int8_t>(prepared.package / field(row, "lut_file"), 256);
                std::copy(lut.begin(), lut.end(), operation.lut.begin());
            } else if (operation.kind == "add_silu") {
                operation.add_lut = read_binary<std::int8_t>(prepared.package / field(row, "add_lut_file"), 256U * 256U);
            } else if (operation.kind == "maxpool") {
                operation.pool_kernel_h = integer(row, "kernel_h");
                operation.pool_kernel_w = integer(row, "kernel_w");
                operation.pool_stride_h = integer(row, "stride_h");
                operation.pool_stride_w = integer(row, "stride_w");
                operation.pool_pad_h = integer(row, "pad_h");
                operation.pool_pad_w = integer(row, "pad_w");
                const Tensor& input = prepared.tensor(operation.inputs[0]);
                const Tensor& output = prepared.tensor(operation.outputs[0]);
                if (operation.pool_kernel_h <= 0 || operation.pool_kernel_w <= 0 ||
                    operation.pool_stride_h <= 0 || operation.pool_stride_w <= 0) {
                    throw std::runtime_error("invalid MaxPool geometry");
                }
                const int expected_h = (input.h + 2 * operation.pool_pad_h - operation.pool_kernel_h) /
                                           operation.pool_stride_h + 1;
                const int expected_w = (input.w + 2 * operation.pool_pad_w - operation.pool_kernel_w) /
                                           operation.pool_stride_w + 1;
                if (input.c != output.c || output.h != expected_h || output.w != expected_w ||
                    input.zero_point != output.zero_point) {
                    throw std::runtime_error("invalid MaxPool package contract");
                }
            } else if (operation.kind == "concat") {
                for (int input_index = 0; input_index < 4; ++input_index) {
                    if (operation.inputs[static_cast<std::size_t>(input_index)] < 0) continue;
                    const auto lut = read_binary<std::int8_t>(prepared.package /
                        field(row, ("concat" + std::to_string(input_index) + "_lut_file").c_str()), 256);
                    std::copy(lut.begin(), lut.end(), operation.concat_lut[static_cast<std::size_t>(input_index)].begin());
                }
            } else {
                throw std::runtime_error("unsupported Stage49 operation: " + operation.kind);
            }
            prepared.operations.push_back(std::move(operation));
        }
        prepared.model4_preact_id = prepared.tensor_id_for_key("model4.preact");
        prepared.model4_postact_id = prepared.tensor_id_for_key("model4.postact");
        prepared.model5_output_id = prepared.tensor_id_for_key("model.5.output");
        prepared.model6_output_id = prepared.tensor_id_for_key("model.6.output");
        prepared.slice_output_id = prepared.tensors.empty() ? -1 : prepared.tensors.back().id;
        if (prepared.tensors.size() < 3 || prepared.operations.size() < 2 || maximum_panel == 0 ||
            prepared.model4_preact_id < 0 || prepared.model4_postact_id < 0 ||
            prepared.model5_output_id < 0 || prepared.slice_output_id < 0) {
            throw std::runtime_error("unexpected persistent integer-slice package surface");
        }
        prepared.pool = std::make_unique<WorkerPool>(
            worker_capacity, maximum_panel, 0, wake_policy);
        if (enable_cluster1_pool) {
            prepared.cluster1_pool = std::make_unique<WorkerPool>(
                worker_capacity, 0, 4, WorkerWakePolicy::condition_variable);
        }
        prepared.ready = true;
        *impl_ = std::move(prepared);
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::exception& error) {
        impl_->error = error.what();
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

namespace {

int execute_range(PersistentSlice::Impl& executor, int first_operation, int last_operation,
                  const RunOptions& options, SliceTiming* timing) {
    if (!executor.ready || !executor.pool || options.workers < 1 ||
        options.workers > executor.pool->capacity() || first_operation < 0 ||
        last_operation >= static_cast<int>(executor.operations.size()) || first_operation > last_operation) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    if (timing != nullptr) {
        *timing = {};
        timing->operations.reserve(static_cast<std::size_t>(last_operation - first_operation + 1));
    }
    if (!options.counter_collection_already_started) {
        executor.pool->begin_counter_collection(options.counter_event);
    }
    if (options.capture_intermediates) {
        executor.captured_tensors.clear();
        executor.captured_tensors.resize(executor.tensors.size());
        const int initial_tensor = first_operation == 0 ? executor.model4_preact_id : executor.model4_postact_id;
        const Tensor& tensor = executor.tensor(initial_tensor);
        executor.captured_tensors[static_cast<std::size_t>(initial_tensor)].assign(
            executor.data(initial_tensor), executor.data(initial_tensor) + tensor.bytes);
    }
    const auto total_begin = Clock::now();
    for (int operation_index = first_operation; operation_index <= last_operation; ++operation_index) {
        Operation& operation = executor.operations[static_cast<std::size_t>(operation_index)];
        OperationTiming row;
        row.operation_index = operation.index;
        row.kind = operation.kind;
        row.name = operation.name;
        const auto begin = Clock::now();
        int status = Y26_CONV_STATUS_SUCCESS;
        if (operation.kind == "conv") {
            std::array<std::int8_t*, 2> outputs {executor.data(operation.outputs[0]), nullptr};
            if (operation.outputs[1] >= 0) outputs[1] = executor.data(operation.outputs[1]);
            double min_worker = 0.0;
            double max_worker = 0.0;
            status = run_conv(*executor.pool, operation.conv, executor.data(operation.inputs[0]), outputs,
                              options, timing != nullptr ? &row : nullptr, &min_worker, &max_worker);
            if (timing != nullptr) {
                timing->min_worker_us = timing->min_worker_us == 0.0 ? min_worker : std::min(timing->min_worker_us, min_worker);
                timing->max_worker_us = std::max(timing->max_worker_us, max_worker);
            }
        } else {
            try {
                if (operation.kind == "lut") run_lut(executor, operation, options);
                else if (operation.kind == "add_silu") run_add(executor, operation, options);
                else if (operation.kind == "concat") run_concat(executor, operation, options);
                else if (operation.kind == "maxpool") run_maxpool(executor, operation, options);
                else status = Y26_CONV_STATUS_INVALID_ARGUMENT;
            } catch (const std::exception&) {
                status = Y26_CONV_STATUS_INVALID_ARGUMENT;
            }
        }
        row.wall_us = elapsed_us(begin, Clock::now());
        if (status != Y26_CONV_STATUS_SUCCESS) return status;
        if (options.capture_intermediates) {
            for (int output_id : operation.outputs) {
                if (output_id < 0) continue;
                const Tensor& tensor = executor.tensor(output_id);
                executor.captured_tensors[static_cast<std::size_t>(output_id)].assign(
                    executor.data(output_id), executor.data(output_id) + tensor.bytes);
            }
        }
        if (timing != nullptr) {
            if (operation.kind == "conv") timing->conv_us += row.wall_us;
            else if (operation.kind == "lut") timing->lut_us += row.wall_us;
            else if (operation.kind == "add_silu") timing->add_us += row.wall_us;
            else if (operation.kind == "concat") timing->concat_us += row.wall_us;
            else if (operation.kind == "maxpool") timing->maxpool_us += row.wall_us;
            timing->operations.push_back(std::move(row));
        }
    }
    if (timing != nullptr) {
        timing->total_us = elapsed_us(total_begin, Clock::now());
        timing->affinity_ok = executor.pool->affinity_ok() &&
                              (!executor.cluster1_pool || executor.cluster1_pool->affinity_ok()) ? 1 : 0;
        timing->worker_counters = executor.pool->worker_counters();
    }
    return Y26_CONV_STATUS_SUCCESS;
}

}  // namespace

int PersistentSlice::run_model5(const std::int8_t* input, std::int8_t* output,
                                const RunOptions& options, SliceTiming* timing) {
    if (!impl_ || !impl_->ready || input == nullptr || output == nullptr) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    const Tensor& input_tensor = impl_->tensor(impl_->model4_postact_id);
    const Tensor& output_tensor = impl_->tensor(impl_->model5_output_id);
    if (int8_v1::ranges_overlap(input, input_tensor.bytes, output, output_tensor.bytes)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memcpy(impl_->data(impl_->model4_postact_id), input, input_tensor.bytes);
    const int status = execute_range(*impl_, 1, 1, options, timing);
    if (status == Y26_CONV_STATUS_SUCCESS) std::memcpy(output, impl_->data(impl_->model5_output_id), output_tensor.bytes);
    return status;
}

int PersistentSlice::run_slice(const std::int8_t* input, std::int8_t* output,
                               const RunOptions& options, SliceTiming* timing) {
    if (!impl_ || !impl_->ready || input == nullptr || output == nullptr) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    const Tensor& input_tensor = impl_->tensor(impl_->model4_preact_id);
    const Tensor& output_tensor = impl_->tensor(impl_->slice_output_id);
    if (int8_v1::ranges_overlap(input, input_tensor.bytes, output, output_tensor.bytes)) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    std::memcpy(impl_->data(impl_->model4_preact_id), input, input_tensor.bytes);
    const int status = execute_range(*impl_, 0, static_cast<int>(impl_->operations.size()) - 1, options, timing);
    if (status == Y26_CONV_STATUS_SUCCESS) std::memcpy(output, impl_->data(impl_->slice_output_id), output_tensor.bytes);
    return status;
}

int PersistentSlice::load_tensor(int tensor_id, const std::int8_t* source, std::size_t bytes) {
    if (!impl_ || !impl_->ready || source == nullptr) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    try {
        const Tensor& tensor = impl_->tensor(tensor_id);
        if (bytes != tensor.bytes) return Y26_CONV_STATUS_INVALID_ARGUMENT;
        std::memcpy(impl_->data(tensor_id), source, bytes);
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::exception&) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

int PersistentSlice::run_model5_resident(const RunOptions& options, SliceTiming* timing) {
    if (!impl_ || !impl_->ready) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    return execute_range(*impl_, 1, 1, options, timing);
}

int PersistentSlice::run_slice_resident(const RunOptions& options, SliceTiming* timing) {
    if (!impl_ || !impl_->ready) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    return execute_range(*impl_, 0, static_cast<int>(impl_->operations.size()) - 1, options, timing);
}

int PersistentSlice::run_operation_resident(int operation_index, const RunOptions& options,
                                            SliceTiming* timing) {
    if (!impl_ || !impl_->ready) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    return execute_range(*impl_, operation_index, operation_index, options, timing);
}

int PersistentSlice::run_range_resident(int first_operation, int last_operation,
                                        const RunOptions& options, SliceTiming* timing) {
    if (!impl_ || !impl_->ready) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    return execute_range(*impl_, first_operation, last_operation, options, timing);
}

int PersistentSlice::bind_external_arena(std::int8_t* arena, std::size_t bytes) {
    if (!impl_ || !impl_->ready || arena == nullptr || bytes < impl_->arena.size()) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    impl_->external_arena = arena;
    impl_->external_arena_bytes = bytes;
    return Y26_CONV_STATUS_SUCCESS;
}

int PersistentSlice::bind_external_tensor(int tensor_id, std::int8_t* data,
                                          std::size_t bytes) {
    if (!impl_ || !impl_->ready || data == nullptr || tensor_id < 0 ||
        static_cast<std::size_t>(tensor_id) >= impl_->external_tensors.size()) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    try {
        if (impl_->tensor(tensor_id).bytes != bytes) return Y26_CONV_STATUS_INVALID_ARGUMENT;
        impl_->external_tensors[static_cast<std::size_t>(tensor_id)] = data;
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::exception&) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

void PersistentSlice::clear_external_tensor_bindings() noexcept {
    if (!impl_) return;
    std::fill(impl_->external_tensors.begin(), impl_->external_tensors.end(), nullptr);
}

int PersistentSlice::dispatch_external(int active_workers, ExternalJob job, void* context) {
    if (!impl_ || !impl_->ready || !impl_->pool || job == nullptr || active_workers < 1 ||
        active_workers > impl_->pool->capacity()) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    struct ExternalDispatchContext {
        ExternalJob job = nullptr;
        void* context = nullptr;
        int workers = 0;
    } dispatch {job, context, active_workers};
    impl_->pool->dispatch(active_workers,
        [](void* opaque, int worker, WorkerScratch&) {
            auto& request = *static_cast<ExternalDispatchContext*>(opaque);
            request.job(request.context, worker, request.workers);
        },
        &dispatch, SchedulerStrategy::active_workers_complete);
    return Y26_CONV_STATUS_SUCCESS;
}

void PersistentSlice::begin_active_window() noexcept {
    if (impl_ && impl_->pool) impl_->pool->begin_active_window();
}

void PersistentSlice::end_active_window() noexcept {
    if (impl_ && impl_->pool) impl_->pool->end_active_window();
}

void PersistentSlice::begin_worker_counter_collection(const std::string& event_group) {
    if (impl_ && impl_->ready && impl_->pool) {
        impl_->pool->begin_counter_collection(event_group);
    }
}

std::vector<WorkerCounter> PersistentSlice::worker_counters() const {
    return impl_ && impl_->ready && impl_->pool
        ? impl_->pool->worker_counters()
        : std::vector<WorkerCounter> {};
}

bool PersistentSlice::worker_affinity_ok() const noexcept {
    return impl_ && impl_->pool && impl_->pool->affinity_ok();
}

int PersistentSlice::copy_tensor(int tensor_id, std::int8_t* destination, std::size_t bytes) const {
    if (!impl_ || !impl_->ready || destination == nullptr) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    try {
        const Tensor& tensor = impl_->tensor(tensor_id);
        if (bytes != tensor.bytes) return Y26_CONV_STATUS_INVALID_ARGUMENT;
        std::memcpy(destination, impl_->data(tensor_id), bytes);
        return Y26_CONV_STATUS_SUCCESS;
    } catch (const std::exception&) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
}

int PersistentSlice::copy_captured_tensor(int tensor_id, std::int8_t* destination, std::size_t bytes) const {
    if (!impl_ || !impl_->ready || destination == nullptr || tensor_id < 0 ||
        static_cast<std::size_t>(tensor_id) >= impl_->captured_tensors.size()) {
        return Y26_CONV_STATUS_INVALID_ARGUMENT;
    }
    const auto& captured = impl_->captured_tensors[static_cast<std::size_t>(tensor_id)];
    if (captured.size() != bytes) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    std::memcpy(destination, captured.data(), bytes);
    return Y26_CONV_STATUS_SUCCESS;
}

std::size_t PersistentSlice::tensor_bytes(int tensor_id) const noexcept {
    if (!impl_ || tensor_id < 0 || static_cast<std::size_t>(tensor_id) >= impl_->tensors.size()) return 0;
    return impl_->tensors[static_cast<std::size_t>(tensor_id)].bytes;
}
int PersistentSlice::tensor_count() const noexcept { return impl_ ? static_cast<int>(impl_->tensors.size()) : 0; }
int PersistentSlice::operation_count() const noexcept { return impl_ ? static_cast<int>(impl_->operations.size()) : 0; }
int PersistentSlice::input_tensor_id() const noexcept { return impl_ ? impl_->model4_preact_id : -1; }
int PersistentSlice::model5_output_tensor_id() const noexcept { return impl_ ? impl_->model5_output_id : -1; }
int PersistentSlice::model6_output_tensor_id() const noexcept { return impl_ ? impl_->model6_output_id : -1; }
int PersistentSlice::output_tensor_id() const noexcept { return impl_ ? impl_->slice_output_id : -1; }
int PersistentSlice::operation_input_tensor_id(int operation_index, int slot) const noexcept {
    if (!impl_ || operation_index < 0 || static_cast<std::size_t>(operation_index) >= impl_->operations.size() ||
        slot < 0 || slot >= static_cast<int>(impl_->operations[static_cast<std::size_t>(operation_index)].inputs.size())) {
        return -1;
    }
    return impl_->operations[static_cast<std::size_t>(operation_index)].inputs[static_cast<std::size_t>(slot)];
}
int PersistentSlice::operation_output_tensor_id(int operation_index, int slot) const noexcept {
    if (!impl_ || operation_index < 0 || static_cast<std::size_t>(operation_index) >= impl_->operations.size() ||
        slot < 0 || slot >= static_cast<int>(impl_->operations[static_cast<std::size_t>(operation_index)].outputs.size())) {
        return -1;
    }
    return impl_->operations[static_cast<std::size_t>(operation_index)].outputs[static_cast<std::size_t>(slot)];
}
int PersistentSlice::tensor_id_for_key(const std::string& key) const noexcept {
    return impl_ ? impl_->tensor_id_for_key(key) : -1;
}
std::size_t PersistentSlice::arena_bytes() const noexcept { return impl_ ? impl_->arena.size() : 0; }
std::size_t PersistentSlice::packed_weight_bytes() const noexcept { return impl_ ? impl_->total_packed_weight_bytes : 0; }
int PersistentSlice::prefault_memory(bool advise_hugepage, std::uint64_t* touched_checksum) noexcept {
    if (!impl_ || !impl_->ready) return Y26_CONV_STATUS_INVALID_ARGUMENT;
    std::uint64_t checksum = 0;
    constexpr std::size_t page_bytes = 4096;
    auto touch = [&](const auto& bytes) {
        for (std::size_t offset = 0; offset < bytes.size(); offset += page_bytes) {
            checksum += static_cast<std::uint8_t>(bytes[offset]);
        }
        if (!bytes.empty()) checksum += static_cast<std::uint8_t>(bytes.back());
    };
    touch(impl_->arena);
    for (const auto& operation : impl_->operations) {
        if (operation.kind == "conv") touch(operation.conv.packed_weights);
    }
#if defined(__linux__)
    if (advise_hugepage && !impl_->arena.empty()) {
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
        const auto mask = static_cast<std::uintptr_t>(page_size - 1);
        const auto begin = reinterpret_cast<std::uintptr_t>(impl_->arena.data()) & ~mask;
        const auto end =
            (reinterpret_cast<std::uintptr_t>(impl_->arena.data() + impl_->arena.size()) + mask) & ~mask;
        if (madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE) != 0) {
            return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
        }
    }
#else
    if (advise_hugepage) return Y26_CONV_STATUS_RUNTIME_SAFETY_FAILED;
#endif
    if (touched_checksum != nullptr) *touched_checksum = checksum;
    return Y26_CONV_STATUS_SUCCESS;
}
const std::string& PersistentSlice::manifest_sha256() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->trusted_manifest : empty;
}
const std::string& PersistentSlice::last_error() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->error : empty;
}

const char* compute_route_name(ComputeRoute value) noexcept {
    return value == ComputeRoute::ime ? "ime" : "scalar";
}
const char* kernel_shape_name(KernelShape value) noexcept {
    if (value == KernelShape::m4n16) return "m4n16";
    if (value == KernelShape::m8n16) return "m8n16";
    return "m12n16";
}
const char* load_strategy_name(LoadStrategy value) noexcept {
    if (value == LoadStrategy::four_u64) return "four_u64";
    if (value == LoadStrategy::vlse64) return "vlse64";
    if (value == LoadStrategy::vlseg2_even) return "vlseg2_even";
    if (value == LoadStrategy::vlseg2_pair_vlse) return "vlseg2_pair_vlse";
    return "vlseg2_pair_shift";
}
const char* epilogue_strategy_name(EpilogueStrategy value) noexcept {
    if (value == EpilogueStrategy::generic_scalar) return "e0_generic_scalar";
    if (value == EpilogueStrategy::inline_scalar) return "e1_inline_scalar";
    return "e2_rvv_q62";
}
const char* partition_policy_name(PartitionPolicy value) noexcept {
    return value == PartitionPolicy::spatial ? "spatial" : "output_channel";
}
const char* nonconv_strategy_name(NonConvStrategy value) noexcept {
    if (value == NonConvStrategy::serial_scalar) return "serial_scalar";
    if (value == NonConvStrategy::parallel_scalar) return "parallel_scalar";
    if (value == NonConvStrategy::cluster1_explicit_rvv) return "cluster1_explicit_rvv";
    return "explicit_rvv_lut";
}
const char* scheduler_strategy_name(SchedulerStrategy value) noexcept {
    return value == SchedulerStrategy::all_workers_complete
        ? "all_workers_complete" : "active_workers_complete";
}

}  // namespace y26::stage49
