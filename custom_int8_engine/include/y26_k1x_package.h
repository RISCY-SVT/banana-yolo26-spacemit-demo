#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace y26::int8_v1 {

inline constexpr int kPackageSchemaVersion = 2;

struct PackageVerification {
    bool ok = false;
    std::string manifest_sha256;
    std::size_t files_verified = 0;
    std::string error;
};

std::string sha256_file(const std::filesystem::path& path);

PackageVerification verify_package(const std::filesystem::path& package_dir,
                                   const std::string& trusted_manifest_sha256,
                                   const std::string& expected_contract_id,
                                   const std::string& expected_profile_id,
                                   const std::string& expected_layout_id,
                                   int expected_schema_version = kPackageSchemaVersion);

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes,
                    const void* rhs, std::size_t rhs_bytes) noexcept;

}  // namespace y26::int8_v1
