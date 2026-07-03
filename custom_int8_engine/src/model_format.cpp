#include "y26_k1x_engine.h"

#include <algorithm>

namespace y26_k1x {

StatusCode validate_model_header(const ModelHeaderV0& header) {
    if (!std::equal(header.magic.begin(), header.magic.end(), kModelMagicV0.begin(), kModelMagicV0.end())) {
        return StatusCode::kInvalidModel;
    }
    if (header.version != kModelFormatVersionV0) {
        return StatusCode::kUnsupported;
    }
    if (header.header_size != sizeof(ModelHeaderV0)) {
        return StatusCode::kInvalidModel;
    }
    if (header.endianness != 0x01020304U) {
        return StatusCode::kUnsupported;
    }
    if (header.alignment < 16 || (header.alignment & (header.alignment - 1U)) != 0U) {
        return StatusCode::kInvalidModel;
    }
    return StatusCode::kOk;
}

std::string_view status_message(StatusCode code) {
    switch (code) {
        case StatusCode::kOk:
            return "ok";
        case StatusCode::kInvalidArgument:
            return "invalid argument";
        case StatusCode::kInvalidModel:
            return "invalid model";
        case StatusCode::kUnsupported:
            return "unsupported";
        case StatusCode::kNotImplemented:
            return "not implemented";
    }
    return "unknown";
}

}  // namespace y26_k1x
