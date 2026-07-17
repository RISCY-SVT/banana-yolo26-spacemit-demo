#include <linux/videodev2.h>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
volatile sig_atomic_t g_stop = 0;

void StopSignal(int) { g_stop = 1; }

std::uint64_t MonotonicNs() {
    struct timespec value {};
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL +
        static_cast<std::uint64_t>(value.tv_nsec);
}

int IoctlRetry(int fd, unsigned long request, void* argument) {
    int status;
    do status = ioctl(fd, request, argument); while (status < 0 && errno == EINTR);
    return status;
}

std::uint32_t Fourcc(const std::string& value) {
    if (value.size() != 4) throw std::runtime_error("FOURCC must contain four characters");
    return v4l2_fourcc(value[0], value[1], value[2], value[3]);
}

std::string FourccText(std::uint32_t value) {
    std::string text(4, '\0');
    text[0] = static_cast<char>(value & 0xffU);
    text[1] = static_cast<char>((value >> 8U) & 0xffU);
    text[2] = static_cast<char>((value >> 16U) & 0xffU);
    text[3] = static_cast<char>((value >> 24U) & 0xffU);
    return text;
}

std::string TimestampClock(std::uint32_t flags) {
    switch (flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) {
        case V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC: return "monotonic";
        case V4L2_BUF_FLAG_TIMESTAMP_COPY: return "copy";
        default: return "unknown";
    }
}

std::string TimestampSource(std::uint32_t flags) {
    return (flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK) == V4L2_BUF_FLAG_TSTAMP_SRC_SOE
        ? "soe" : "eof";
}

struct Mapping {
    void* address = MAP_FAILED;
    std::size_t length = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " DEVICE WIDTH HEIGHT FOURCC FPS DURATION_SECONDS OUTPUT_TSV\n";
        return 2;
    }

    const std::string device = argv[1];
    const unsigned width = static_cast<unsigned>(std::stoul(argv[2]));
    const unsigned height = static_cast<unsigned>(std::stoul(argv[3]));
    const std::string requested_fourcc = argv[4];
    const double requested_fps = std::stod(argv[5]);
    const double duration_seconds = std::stod(argv[6]);
    const std::string output_path = argv[7];
    if (width == 0 || height == 0 || requested_fps <= 0.0 || duration_seconds <= 0.0) {
        std::cerr << "width, height, FPS, and duration must be positive\n";
        return 2;
    }

    struct sigaction action {};
    sigemptyset(&action.sa_mask);
    action.sa_handler = StopSignal;
    if (sigaction(SIGINT, &action, nullptr) != 0 ||
        sigaction(SIGTERM, &action, nullptr) != 0 ||
        sigaction(SIGHUP, &action, nullptr) != 0) {
        std::cerr << "cannot install signal handlers\n";
        return 2;
    }

    int fd = -1;
    std::vector<Mapping> mappings;
    bool streaming = false;
    try {
        fd = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("open failed: " + std::string(std::strerror(errno)));

        v4l2_capability capability {};
        if (IoctlRetry(fd, VIDIOC_QUERYCAP, &capability) != 0)
            throw std::runtime_error("VIDIOC_QUERYCAP failed");
        const std::uint32_t caps = capability.device_caps != 0
            ? capability.device_caps : capability.capabilities;
        if ((caps & V4L2_CAP_VIDEO_CAPTURE) == 0 || (caps & V4L2_CAP_STREAMING) == 0)
            throw std::runtime_error("device lacks single-plane capture/MMAP streaming");

        v4l2_format format {};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = width;
        format.fmt.pix.height = height;
        format.fmt.pix.pixelformat = Fourcc(requested_fourcc);
        format.fmt.pix.field = V4L2_FIELD_ANY;
        if (IoctlRetry(fd, VIDIOC_S_FMT, &format) != 0)
            throw std::runtime_error("VIDIOC_S_FMT failed");

        v4l2_streamparm parameters {};
        parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parameters.parm.capture.timeperframe.numerator = 1000;
        parameters.parm.capture.timeperframe.denominator =
            static_cast<unsigned>(requested_fps * 1000.0 + 0.5);
        if (IoctlRetry(fd, VIDIOC_S_PARM, &parameters) != 0)
            throw std::runtime_error("VIDIOC_S_PARM failed");
        if (IoctlRetry(fd, VIDIOC_G_PARM, &parameters) != 0)
            throw std::runtime_error("VIDIOC_G_PARM failed");

        v4l2_requestbuffers request {};
        request.count = 4;
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        if (IoctlRetry(fd, VIDIOC_REQBUFS, &request) != 0 || request.count < 2)
            throw std::runtime_error("VIDIOC_REQBUFS returned fewer than two buffers");
        mappings.resize(request.count);
        for (unsigned index = 0; index < request.count; ++index) {
            v4l2_buffer buffer {};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = index;
            if (IoctlRetry(fd, VIDIOC_QUERYBUF, &buffer) != 0)
                throw std::runtime_error("VIDIOC_QUERYBUF failed");
            mappings[index].length = buffer.length;
            mappings[index].address = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, fd, buffer.m.offset);
            if (mappings[index].address == MAP_FAILED)
                throw std::runtime_error("mmap failed");
            if (IoctlRetry(fd, VIDIOC_QBUF, &buffer) != 0)
                throw std::runtime_error("initial VIDIOC_QBUF failed");
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (IoctlRetry(fd, VIDIOC_STREAMON, &type) != 0)
            throw std::runtime_error("VIDIOC_STREAMON failed");
        streaming = true;

        std::ostringstream rows;
        rows << "index\tbuffer_index\tbuffer_count\tsequence"
                "\tkernel_timestamp_sec\tkernel_timestamp_usec"
                "\tkernel_timestamp_ns\ttimestamp_clock\ttimestamp_source\tbytesused"
                "\tflags_hex\tdqbuf_return_monotonic_ns\twait_us\trequeue_us\tsequence_gap\n";
        const auto begin = Clock::now();
        std::uint64_t count = 0;
        std::uint64_t gaps = 0;
        std::uint32_t previous_sequence = 0;
        bool have_previous = false;
        std::uint64_t first_return_ns = 0;
        std::uint64_t last_return_ns = 0;
        std::string timestamp_clock = "unknown";
        std::string timestamp_source = "unknown";

        while (!g_stop && std::chrono::duration<double>(Clock::now() - begin).count() <
                              duration_seconds) {
            pollfd descriptor{fd, POLLIN, 0};
            const std::uint64_t wait_begin_ns = MonotonicNs();
            const int poll_status = poll(&descriptor, 1, 2000);
            if (poll_status < 0 && errno == EINTR) continue;
            if (poll_status <= 0) throw std::runtime_error("poll timeout/failure");

            v4l2_buffer buffer {};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            if (IoctlRetry(fd, VIDIOC_DQBUF, &buffer) != 0) {
                if (errno == EAGAIN) continue;
                throw std::runtime_error("VIDIOC_DQBUF failed");
            }
            const std::uint64_t dq_return_ns = MonotonicNs();
            const std::uint32_t buffer_index = buffer.index;
            const std::uint32_t sequence = buffer.sequence;
            const std::uint32_t flags = buffer.flags;
            const std::uint32_t bytesused = buffer.bytesused;
            const struct timeval kernel_timestamp = buffer.timestamp;
            const std::uint64_t gap = have_previous && sequence > previous_sequence
                ? static_cast<std::uint64_t>(sequence - previous_sequence - 1U) : 0U;
            gaps += gap;
            previous_sequence = sequence;
            have_previous = true;
            timestamp_clock = TimestampClock(flags);
            timestamp_source = TimestampSource(flags);
            const std::uint64_t kernel_ns =
                static_cast<std::uint64_t>(kernel_timestamp.tv_sec) * 1000000000ULL +
                static_cast<std::uint64_t>(kernel_timestamp.tv_usec) * 1000ULL;
            const std::uint64_t requeue_begin_ns = MonotonicNs();
            if (IoctlRetry(fd, VIDIOC_QBUF, &buffer) != 0)
                throw std::runtime_error("VIDIOC_QBUF failed");
            const std::uint64_t requeue_end_ns = MonotonicNs();
            if (count == 0) first_return_ns = dq_return_ns;
            last_return_ns = dq_return_ns;
            rows << count << '\t' << buffer_index << '\t' << request.count << '\t'
                 << sequence << '\t' << kernel_timestamp.tv_sec << '\t'
                 << kernel_timestamp.tv_usec << '\t' << kernel_ns << '\t' << timestamp_clock
                 << '\t' << timestamp_source << '\t' << bytesused << "\t0x" << std::hex
                 << flags << std::dec << '\t' << dq_return_ns << '\t'
                 << (dq_return_ns - wait_begin_ns) / 1000ULL << '\t'
                 << (requeue_end_ns - requeue_begin_ns) / 1000ULL << '\t' << gap << '\n';
            ++count;
        }

        if (streaming) {
            (void)IoctlRetry(fd, VIDIOC_STREAMOFF, &type);
            streaming = false;
        }
        const double effective_fps = parameters.parm.capture.timeperframe.numerator != 0
            ? static_cast<double>(parameters.parm.capture.timeperframe.denominator) /
                parameters.parm.capture.timeperframe.numerator : 0.0;
        const double dequeued_fps = count > 1 && last_return_ns > first_return_ns
            ? 1.0e9 * static_cast<double>(count - 1) /
                static_cast<double>(last_return_ns - first_return_ns) : 0.0;
        std::ofstream output(output_path, std::ios::out | std::ios::trunc);
        output << "# device=" << device << " requested=" << width << 'x' << height << '@'
               << requested_fps << ' ' << requested_fourcc << " effective="
               << format.fmt.pix.width << 'x' << format.fmt.pix.height << '@' << effective_fps
               << ' ' << FourccText(format.fmt.pix.pixelformat) << " buffers=" << request.count
               << " timestamp_clock=" << timestamp_clock
               << " timestamp_source=" << timestamp_source << "\n" << rows.str();
        if (!output.good()) throw std::runtime_error("cannot write output TSV");
        std::cout << std::fixed << std::setprecision(6)
                  << "SUMMARY buffers=" << count << " sequence_gaps=" << gaps
                  << " raw_v4l2_buffer_fps=" << dequeued_fps
                  << " timestamp_clock=" << timestamp_clock
                  << " timestamp_source=" << timestamp_source
                  << " effective_format=" << format.fmt.pix.width << 'x'
                  << format.fmt.pix.height << '@' << effective_fps << ' '
                  << FourccText(format.fmt.pix.pixelformat) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "v4l2 probe failed: " << error.what() << '\n';
        if (streaming && fd >= 0) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            (void)IoctlRetry(fd, VIDIOC_STREAMOFF, &type);
        }
        for (const Mapping& mapping : mappings) {
            if (mapping.address != MAP_FAILED) munmap(mapping.address, mapping.length);
        }
        if (fd >= 0) close(fd);
        return 1;
    }
    for (const Mapping& mapping : mappings) {
        if (mapping.address != MAP_FAILED) munmap(mapping.address, mapping.length);
    }
    if (fd >= 0) close(fd);
    return 0;
}
