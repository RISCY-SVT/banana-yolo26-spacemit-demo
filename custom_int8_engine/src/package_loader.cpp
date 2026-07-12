#include "y26_k1x_package.h"

#include <array>
#include <bit>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace y26::int8_v1 {
namespace {

static_assert(std::endian::native == std::endian::little,
              "K1X_INT8_V1 packages require a little-endian target");

struct Sha256State {
    std::array<std::uint32_t, 8> h {};
    std::uint64_t bit_length = 0;
    std::array<std::uint8_t, 64> buffer {};
    std::size_t buffer_length = 0;
};

constexpr std::array<std::uint32_t, 64> kSha256Constants {{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
}};

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t count) noexcept {
    return (value >> count) | (value << (32U - count));
}

void sha256_init(Sha256State& state) noexcept {
    state.h = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    state.bit_length = 0;
    state.buffer_length = 0;
}

void sha256_transform(Sha256State& state, const std::uint8_t* block) noexcept {
    std::array<std::uint32_t, 64> words {};
    for (int index = 0; index < 16; ++index) {
        words[static_cast<std::size_t>(index)] =
            (static_cast<std::uint32_t>(block[index * 4]) << 24U) |
            (static_cast<std::uint32_t>(block[index * 4 + 1]) << 16U) |
            (static_cast<std::uint32_t>(block[index * 4 + 2]) << 8U) |
            static_cast<std::uint32_t>(block[index * 4 + 3]);
    }
    for (int index = 16; index < 64; ++index) {
        const std::uint32_t a = words[static_cast<std::size_t>(index - 15)];
        const std::uint32_t b = words[static_cast<std::size_t>(index - 2)];
        const std::uint32_t s0 = rotate_right(a, 7) ^ rotate_right(a, 18) ^ (a >> 3U);
        const std::uint32_t s1 = rotate_right(b, 17) ^ rotate_right(b, 19) ^ (b >> 10U);
        words[static_cast<std::size_t>(index)] = words[static_cast<std::size_t>(index - 16)] + s0 +
            words[static_cast<std::size_t>(index - 7)] + s1;
    }
    std::uint32_t a = state.h[0];
    std::uint32_t b = state.h[1];
    std::uint32_t c = state.h[2];
    std::uint32_t d = state.h[3];
    std::uint32_t e = state.h[4];
    std::uint32_t f = state.h[5];
    std::uint32_t g = state.h[6];
    std::uint32_t h = state.h[7];
    for (int index = 0; index < 64; ++index) {
        const std::uint32_t sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + sigma1 + choice + kSha256Constants[static_cast<std::size_t>(index)] +
            words[static_cast<std::size_t>(index)];
        const std::uint32_t sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2 = sigma0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state.h[0] += a;
    state.h[1] += b;
    state.h[2] += c;
    state.h[3] += d;
    state.h[4] += e;
    state.h[5] += f;
    state.h[6] += g;
    state.h[7] += h;
}

void sha256_update(Sha256State& state, const std::uint8_t* data, std::size_t bytes) noexcept {
    for (std::size_t index = 0; index < bytes; ++index) {
        state.buffer[state.buffer_length++] = data[index];
        if (state.buffer_length == state.buffer.size()) {
            sha256_transform(state, state.buffer.data());
            state.bit_length += 512;
            state.buffer_length = 0;
        }
    }
}

std::array<std::uint8_t, 32> sha256_finish(Sha256State& state) noexcept {
    const std::size_t payload = state.buffer_length;
    state.buffer[state.buffer_length++] = 0x80U;
    if (state.buffer_length > 56) {
        while (state.buffer_length < 64) state.buffer[state.buffer_length++] = 0;
        sha256_transform(state, state.buffer.data());
        state.buffer.fill(0);
        state.buffer_length = 0;
    }
    while (state.buffer_length < 56) state.buffer[state.buffer_length++] = 0;
    state.bit_length += static_cast<std::uint64_t>(payload) * 8U;
    for (int index = 0; index < 8; ++index) {
        state.buffer[63 - index] = static_cast<std::uint8_t>(state.bit_length >> (index * 8));
    }
    sha256_transform(state, state.buffer.data());
    std::array<std::uint8_t, 32> digest {};
    for (int word = 0; word < 8; ++word) {
        for (int byte = 0; byte < 4; ++byte) {
            digest[static_cast<std::size_t>(word * 4 + byte)] =
                static_cast<std::uint8_t>(state.h[static_cast<std::size_t>(word)] >> (24 - byte * 8));
        }
    }
    return digest;
}

std::string digest_hex(const std::array<std::uint8_t, 32>& digest) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::uint8_t byte : digest) output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}

std::vector<std::string> split_tsv(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    for (;;) {
        const std::size_t end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return fields;
}

bool valid_sha256(std::string_view value) noexcept {
    if (value.size() != 64) return false;
    for (char ch : value) {
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    }
    return true;
}

void skip_space(const std::string& text, std::size_t& cursor) {
    while (cursor < text.size() && (text[cursor] == ' ' || text[cursor] == '\n' ||
                                    text[cursor] == '\r' || text[cursor] == '\t')) ++cursor;
}

std::string parse_json_string(const std::string& text, std::size_t& cursor) {
    if (cursor >= text.size() || text[cursor++] != '"') throw std::runtime_error("expected JSON string");
    std::string result;
    while (cursor < text.size()) {
        const char ch = text[cursor++];
        if (ch == '"') return result;
        if (ch == '\\') {
            if (cursor >= text.size()) throw std::runtime_error("truncated JSON escape");
            const char escaped = text[cursor++];
            if (escaped == '"' || escaped == '\\' || escaped == '/') result.push_back(escaped);
            else if (escaped == 'n') result.push_back('\n');
            else if (escaped == 'r') result.push_back('\r');
            else if (escaped == 't') result.push_back('\t');
            else throw std::runtime_error("unsupported JSON escape");
        } else {
            result.push_back(ch);
        }
    }
    throw std::runtime_error("unterminated JSON string");
}

std::unordered_map<std::string, std::string> parse_flat_json(const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open package.json");
    const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    std::unordered_map<std::string, std::string> values;
    std::size_t cursor = 0;
    skip_space(text, cursor);
    if (cursor >= text.size() || text[cursor++] != '{') throw std::runtime_error("package.json is not an object");
    for (;;) {
        skip_space(text, cursor);
        if (cursor < text.size() && text[cursor] == '}') {
            ++cursor;
            break;
        }
        const std::string key = parse_json_string(text, cursor);
        skip_space(text, cursor);
        if (cursor >= text.size() || text[cursor++] != ':') throw std::runtime_error("missing JSON colon");
        skip_space(text, cursor);
        std::string value;
        if (cursor < text.size() && text[cursor] == '"') {
            value = parse_json_string(text, cursor);
        } else {
            const std::size_t begin = cursor;
            while (cursor < text.size() && text[cursor] != ',' && text[cursor] != '}') ++cursor;
            value = text.substr(begin, cursor - begin);
            const std::size_t end = value.find_last_not_of(" \n\r\t");
            value = end == std::string::npos ? std::string() : value.substr(0, end + 1);
            if (value.empty() || value.find_first_of("[{") != std::string::npos) {
                throw std::runtime_error("package.json must be a flat scalar object");
            }
        }
        if (!values.emplace(key, value).second) throw std::runtime_error("duplicate JSON key");
        skip_space(text, cursor);
        if (cursor < text.size() && text[cursor] == ',') {
            ++cursor;
            continue;
        }
        if (cursor < text.size() && text[cursor] == '}') {
            ++cursor;
            break;
        }
        throw std::runtime_error("malformed package.json");
    }
    skip_space(text, cursor);
    if (cursor != text.size()) throw std::runtime_error("trailing package.json data");
    return values;
}

const std::string& required_json(const std::unordered_map<std::string, std::string>& values,
                                 const char* key) {
    const auto found = values.find(key);
    if (found == values.end()) throw std::runtime_error(std::string("missing package.json key: ") + key);
    return found->second;
}

std::uint64_t parse_u64(std::string_view value, const char* field) {
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error(std::string("invalid unsigned field: ") + field);
    }
    return result;
}

}  // namespace

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file for SHA-256: " + path.string());
    Sha256State state;
    sha256_init(state);
    std::array<std::uint8_t, 65536> buffer {};
    while (stream) {
        stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) sha256_update(state, buffer.data(), static_cast<std::size_t>(count));
    }
    if (!stream.eof()) throw std::runtime_error("failed while hashing file: " + path.string());
    return digest_hex(sha256_finish(state));
}

PackageVerification verify_package(const std::filesystem::path& package_dir,
                                   const std::string& trusted_manifest_sha256,
                                   const std::string& expected_contract_id,
                                   const std::string& expected_profile_id,
                                   const std::string& expected_layout_id,
                                   int expected_schema_version) {
    PackageVerification result;
    try {
        if (!valid_sha256(trusted_manifest_sha256)) throw std::runtime_error("invalid trusted manifest SHA-256");
        const auto root_status = std::filesystem::symlink_status(package_dir);
        if (!std::filesystem::is_directory(root_status) || std::filesystem::is_symlink(root_status)) {
            throw std::runtime_error("package root must be a non-symlink directory");
        }
        const std::filesystem::path root = std::filesystem::canonical(package_dir);
        const std::filesystem::path manifest_path = root / "asset_hashes.tsv";
        result.manifest_sha256 = sha256_file(manifest_path);
        if (result.manifest_sha256 != trusted_manifest_sha256) {
            throw std::runtime_error("trusted package manifest SHA-256 mismatch");
        }
        std::ifstream manifest(manifest_path);
        std::string line;
        if (!manifest || !std::getline(manifest, line) || line != "path\tbytes\tsha256") {
            throw std::runtime_error("malformed asset_hashes.tsv header");
        }
        std::set<std::filesystem::path> expected_files {"asset_hashes.tsv"};
        while (std::getline(manifest, line)) {
            if (line.empty()) continue;
            const auto fields = split_tsv(line);
            if (fields.size() != 3 || fields[0].empty() || !valid_sha256(fields[2])) {
                throw std::runtime_error("malformed asset_hashes.tsv row");
            }
            const std::filesystem::path relative(fields[0]);
            if (relative.is_absolute() || relative.lexically_normal() != relative) {
                throw std::runtime_error("unsafe asset path: " + fields[0]);
            }
            for (const auto& part : relative) {
                if (part == ".." || part == ".") throw std::runtime_error("unsafe asset path: " + fields[0]);
            }
            if (!expected_files.insert(relative).second) throw std::runtime_error("duplicate asset path");
            const std::filesystem::path asset = root / relative;
            const auto status = std::filesystem::symlink_status(asset);
            if (!std::filesystem::is_regular_file(status) || std::filesystem::is_symlink(status)) {
                throw std::runtime_error("asset is not a regular file: " + fields[0]);
            }
            const auto executable = std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec |
                                    std::filesystem::perms::others_exec;
            if ((status.permissions() & executable) != std::filesystem::perms::none) {
                throw std::runtime_error("executable package asset rejected: " + fields[0]);
            }
            if (std::filesystem::file_size(asset) != parse_u64(fields[1], "bytes")) {
                throw std::runtime_error("asset size mismatch: " + fields[0]);
            }
            if (sha256_file(asset) != fields[2]) throw std::runtime_error("asset SHA-256 mismatch: " + fields[0]);
            ++result.files_verified;
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
            const auto status = entry.symlink_status();
            if (std::filesystem::is_symlink(status)) throw std::runtime_error("package symlink rejected");
            if (std::filesystem::is_regular_file(status)) {
                const auto relative = std::filesystem::relative(entry.path(), root);
                if (!expected_files.contains(relative)) throw std::runtime_error("unexpected package file: " + relative.string());
            } else if (!std::filesystem::is_directory(status)) {
                throw std::runtime_error("unsupported package entry type");
            }
        }
        const auto package = parse_flat_json(root / "package.json");
        if (required_json(package, "contract_id") != expected_contract_id ||
            required_json(package, "profile_id") != expected_profile_id ||
            required_json(package, "layout_id") != expected_layout_id ||
            required_json(package, "byte_order") != "little-endian") {
            throw std::runtime_error("package identity mismatch");
        }
        const auto schema = parse_u64(required_json(package, "schema_version"), "schema_version");
        if (schema != static_cast<std::uint64_t>(expected_schema_version)) {
            throw std::runtime_error("package schema version mismatch");
        }
        if (!valid_sha256(required_json(package, "model_sha256")) ||
            required_json(package, "source_lineage_id").empty()) {
            throw std::runtime_error("missing package source lineage");
        }
        result.ok = true;
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

bool ranges_overlap(const void* lhs, std::size_t lhs_bytes,
                    const void* rhs, std::size_t rhs_bytes) noexcept {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) return false;
    const auto left = reinterpret_cast<std::uintptr_t>(lhs);
    const auto right = reinterpret_cast<std::uintptr_t>(rhs);
    if (left > std::numeric_limits<std::uintptr_t>::max() - lhs_bytes ||
        right > std::numeric_limits<std::uintptr_t>::max() - rhs_bytes) return true;
    return left < right + rhs_bytes && right < left + lhs_bytes;
}

}  // namespace y26::int8_v1
