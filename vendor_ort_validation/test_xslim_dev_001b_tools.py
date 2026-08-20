from __future__ import annotations

import hashlib

import numpy as np
import onnx
import xslim_dev_001b_candidates as tools
from onnx import TensorProto, helper, numpy_helper


def test_qdq_uses_ties_to_even_and_signed_rails() -> None:
    values = np.asarray([-200.0, -2.5, -1.5, 1.5, 2.5, 200.0], dtype=np.float32)
    rebuilt, codes = tools.qdq(values, 1.0, 0)
    assert codes.tolist() == [-128, -2, -2, 2, 2, 127]
    assert rebuilt.tolist() == [-128.0, -2.0, -2.0, 2.0, 2.0, 127.0]


def test_patch_extraction_honors_padding_and_stride() -> None:
    values = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    descriptor = {
        "kernel_shape": [3, 3],
        "strides": [2, 2],
        "dilations": [1, 1],
        "pads": [1, 1, 1, 1],
    }
    patches = tools.extract_patches(values, [(0, 0), (1, 1)], descriptor)
    np.testing.assert_array_equal(
        patches[0, 0],
        np.asarray([[0, 0, 0], [0, 0, 1], [0, 4, 5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        patches[1, 0],
        np.asarray([[5, 6, 7], [9, 10, 11], [13, 14, 15]], dtype=np.float32),
    )


def test_patch_position_selection_is_bounded_repeatable_and_ordered() -> None:
    first = tools.selected_positions("image.jpg", "tensor", 320, 320, 8)
    second = tools.selected_positions("image.jpg", "tensor", 320, 320, 8)
    changed = tools.selected_positions("image.jpg", "other-tensor", 320, 320, 8)

    assert first == second
    assert first != changed
    assert len(first) == len(set(first)) == 8
    assert all(0 <= row < 320 and 0 <= column < 320 for row, column in first)
    assert tools.selected_positions("image.jpg", "tensor", 2, 2, 10)
    assert tools.selected_positions("image.jpg", "tensor", 2, 2, 0) == []
    with np.testing.assert_raises_regex(ValueError, "positive dimensions"):
        tools.selected_positions("image.jpg", "tensor", 0, 2, 1)


def test_inversion_count_and_rank_order_are_deterministic() -> None:
    assert tools.inversion_count(np.asarray([0, 1, 2, 3])) == 0
    assert tools.inversion_count(np.asarray([3, 2, 1, 0])) == 6
    teacher = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    candidate = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert tools.rank_inversions(teacher, candidate, k=4) == 6


def test_topk_overlap_matches_exact_position_class_pairs() -> None:
    teacher = {
        tools.TAIL_DEBUG_OUTPUTS[0]: np.asarray([[10, 11, 12]]),
        tools.TAIL_DEBUG_OUTPUTS[1]: np.asarray([[1, 1, 2]]),
    }
    candidate = {
        tools.TAIL_DEBUG_OUTPUTS[0]: np.asarray([[10, 11, 99]]),
        tools.TAIL_DEBUG_OUTPUTS[1]: np.asarray([[1, 2, 2]]),
    }
    assert tools.topk_overlap(teacher, candidate) == 1

    malformed = dict(candidate)
    malformed[tools.TAIL_DEBUG_OUTPUTS[1]] = np.asarray([[1, 2]])
    with np.testing.assert_raises_regex(ValueError, "position/class shapes differ"):
        tools.topk_overlap(teacher, malformed)


def test_terminal_qparam_edit_preserves_graph_signature() -> None:
    scale_q = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="scale-q")
    zero_q = numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="zero-q")
    scale_dq = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="scale-dq")
    zero_dq = numpy_helper.from_array(np.asarray(0, dtype=np.int8), name="zero-dq")
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["input", "scale-q", "zero-q"], ["q"], name="q"),
            helper.make_node("DequantizeLinear", ["q", "scale-dq", "zero-dq"], ["output"], name="dq"),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [scale_q, zero_q, scale_dq, zero_dq],
    )
    model = helper.make_model(graph)
    before = tools.graph_signature(model)
    for name in ("scale-q", "scale-dq"):
        tools.set_scalar_initializer(model, name, 0.25, np.dtype(np.float32))
    for name in ("zero-q", "zero-dq"):
        tools.set_scalar_initializer(model, name, 123, np.dtype(np.int8))
    assert tools.graph_signature(model) == before
    onnx.checker.check_model(model)


def test_candidate_scale_search_is_repeatable() -> None:
    values = np.asarray([-10.0, -1.0, 0.0, 1.0, 4.0], dtype=np.float32)
    assert tools.candidate_scales(values, 8) == tools.candidate_scales(values[::-1], 8)


def test_component_archive_is_byte_reproducible(tmp_path) -> None:
    arrays = {
        "weight::b": np.arange(12, dtype=np.int8).reshape(3, 4),
        "bias::a": np.asarray([1.5, -2.0], dtype=np.float32),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    tools.save_npz_deterministic(first, arrays)
    tools.save_npz_deterministic(second, dict(reversed(list(arrays.items()))))

    assert first.read_bytes() == second.read_bytes()
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()
    with np.load(first, allow_pickle=False) as loaded:
        assert sorted(loaded.files) == sorted(arrays)
        for name, expected in arrays.items():
            np.testing.assert_array_equal(loaded[name], expected)


def test_fixed_candidate_matrix_only_conditions_combined_lane() -> None:
    none_qualified = tools.candidate_lane_groups(
        {"R7": {"qualified": False}, "R0": {"qualified": False}}
    )
    assert none_qualified == {
        "C2_T6_RANK_QP": (),
        "C3_R7_BR": ("R7",),
        "C4_R0_BR": ("R0",),
    }

    r7_qualified = tools.candidate_lane_groups(
        {"R7": {"qualified": True}, "R0": {"qualified": False}}
    )
    assert r7_qualified["C5_COMBINED"] == ("R7",)

    both_qualified = tools.candidate_lane_groups(
        {"R7": {"qualified": True}, "R0": {"qualified": True}}
    )
    assert both_qualified["C5_COMBINED"] == ("R7", "R0")
