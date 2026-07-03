#include "y26_k1x_engine.h"

#include <cassert>

int main() {
    y26_k1x::ModelHeaderV0 header{};
    header.header_size = sizeof(y26_k1x::ModelHeaderV0);
    assert(y26_k1x::validate_model_header(header) == y26_k1x::StatusCode::kOk);

    header.magic[0] = 'X';
    assert(y26_k1x::validate_model_header(header) == y26_k1x::StatusCode::kInvalidModel);

    header.magic = y26_k1x::kModelMagicV0;
    header.version = 99;
    assert(y26_k1x::validate_model_header(header) == y26_k1x::StatusCode::kUnsupported);
    return 0;
}
