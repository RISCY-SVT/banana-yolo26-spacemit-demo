#include "../kernels/stage49_persistent_slice.cpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>

namespace {

struct JobContext {
    std::atomic<std::uint64_t> epoch {0};
    std::array<std::atomic<std::uint64_t>, 4> observed_epoch {};
    std::array<std::atomic<std::uint64_t>, 4> calls {};
    std::atomic<int> duplicate_calls {0};
};

void count_job(void* opaque, int worker, y26::stage49::WorkerScratch&) {
    auto& context = *static_cast<JobContext*>(opaque);
    const std::uint64_t epoch = context.epoch.load(std::memory_order_acquire);
    const std::uint64_t prior = context.observed_epoch[static_cast<std::size_t>(worker)].exchange(
        epoch, std::memory_order_acq_rel);
    if (prior == epoch) context.duplicate_calls.fetch_add(1, std::memory_order_relaxed);
    context.calls[static_cast<std::size_t>(worker)].fetch_add(1, std::memory_order_relaxed);
}

int parse_positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (text == end || *end != '\0' || value <= 0 || value > 1000000) {
        std::cerr << "invalid " << name << ": " << text << '\n';
        std::exit(2);
    }
    return static_cast<int>(value);
}

int run_lifecycle(int transitions, int constructions) {
    using y26::stage49::SchedulerStrategy;
    using y26::stage49::WorkerPool;
    using y26::stage49::WorkerWakePolicy;

    std::uint64_t expected_dispatches = 0;
    std::array<std::uint64_t, 4> expected_calls {};
    JobContext context;

    {
        WorkerPool pool(4, 64, 0, WorkerWakePolicy::frame_gated_spin);
        for (int transition = 0; transition < transitions; ++transition) {
            pool.begin_active_window();
            if (transition % 3 != 0) {
                const int active = transition % 4 + 1;
                context.epoch.store(++expected_dispatches, std::memory_order_release);
                pool.dispatch(active, count_job, &context,
                              SchedulerStrategy::all_workers_complete);
                for (int worker = 0; worker < active; ++worker) {
                    ++expected_calls[static_cast<std::size_t>(worker)];
                }
            }
            pool.end_active_window();

            // Wake and park without publishing a generation. The previous job
            // must not replay during this empty active window.
            if (transition % 5 == 0) {
                pool.begin_active_window();
                pool.end_active_window();
            }
        }
    }

    int failures = 0;
    for (std::size_t worker = 0; worker < expected_calls.size(); ++worker) {
        const std::uint64_t actual = context.calls[worker].load(std::memory_order_relaxed);
        if (actual != expected_calls[worker]) {
            std::cerr << "worker " << worker << " calls=" << actual
                      << " expected=" << expected_calls[worker] << '\n';
            ++failures;
        }
    }

    // Repeatedly destroy pools while all workers are parked. This also covers
    // construction followed by immediate lifecycle transitions.
    for (int iteration = 0; iteration < constructions; ++iteration) {
        WorkerPool pool(4, 64, 0, WorkerWakePolicy::frame_gated_spin);
        pool.begin_active_window();
        context.epoch.store(++expected_dispatches, std::memory_order_release);
        pool.dispatch(iteration % 4 + 1, count_job, &context,
                      SchedulerStrategy::all_workers_complete);
        pool.end_active_window();
    }

    failures += context.duplicate_calls.load(std::memory_order_relaxed);

    std::cout << "stage60m_worker_lifecycle transitions=" << transitions
              << " constructions=" << constructions
              << " dispatches=" << expected_dispatches
              << " duplicate_calls=" << context.duplicate_calls.load()
              << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    int transitions = 2000;
    int constructions = 20;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--transitions" && index + 1 < argc) {
            transitions = parse_positive(argv[++index], "transitions");
        } else if (argument == "--constructions" && index + 1 < argc) {
            constructions = parse_positive(argv[++index], "constructions");
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--transitions N] [--constructions N]\n";
            return 2;
        }
    }

    std::atomic<bool> complete {false};
    std::thread watchdog([&]() {
        for (int tick = 0; tick < 6000; ++tick) {
            if (complete.load(std::memory_order_acquire)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cerr << "stage60m worker lifecycle watchdog timeout\n";
        std::_Exit(124);
    });

    const int status = run_lifecycle(transitions, constructions);
    complete.store(true, std::memory_order_release);
    watchdog.join();
    return status;
}
