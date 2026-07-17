/**
 * @file logger.h
 * @brief Minimal timestamped logger used by the demo application.
 */

#pragma once

#include <fstream>
#include <mutex>
#include <string>

namespace banana_demo {

/**
 * @brief Timestamped stderr plus optional file logger.
 */
class Logger
{
public:
    /**
     * @brief Construct a logger.
     * @param quiet Suppress non-error stderr lines when `true`.
     * @param log_path Optional append-only log file path.
     */
    Logger(bool quiet, const std::string& log_path);

    /** @brief Emit an informational message. */
    void Info(const std::string& message);
    /** @brief Emit a warning message. */
    void Warn(const std::string& message);
    /** @brief Emit an error message. */
    void Error(const std::string& message);

private:
    /** @brief Format and dispatch one log line. */
    void Write(const char* level, const std::string& message);

    /** Whether non-error console logging is suppressed. */
    bool quiet_ = false;
    /** Optional append-only log file stream. */
    std::ofstream file_;
    std::mutex mutex_;
};

}  // namespace banana_demo
