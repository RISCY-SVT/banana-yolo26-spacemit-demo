/**
 * @file logger.cpp
 * @brief Timestamped stderr and file logging implementation.
 */

#include "banana_demo/util/logger.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace banana_demo {

namespace {

/** @brief Format the current wall clock time for log lines. */
std::string TimestampNow()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%F %T");
    return oss.str();
}

}  // namespace

Logger::Logger(bool quiet, const std::string& log_path) : quiet_(quiet)
{
    if (!log_path.empty())
        file_.open(log_path, std::ios::out | std::ios::app);
}

void Logger::Info(const std::string& message)
{
    Write("INFO", message);
}

void Logger::Warn(const std::string& message)
{
    Write("WARN", message);
}

void Logger::Error(const std::string& message)
{
    Write("ERROR", message);
}

void Logger::Write(const char* level, const std::string& message)
{
    std::lock_guard lock(mutex_);
    // Keep console and file logs identical so copied snippets remain comparable.
    const std::string line = "[" + TimestampNow() + "] " + level + " " + message;
    if (!quiet_ || std::string(level) == "ERROR")
        std::cerr << line << '\n';
    if (file_)
        file_ << line << '\n';
}

}  // namespace banana_demo
