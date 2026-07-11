#!/usr/bin/env python3
"""Stage45 graph, profile, and directional COCO accuracy diagnostics."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tensor_shape(value: Any) -> list[int] | None:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    shape: list[int] = []
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value >= 0:
            shape.append(int(dim.dim_value))
        else:
            return None
    return shape


def element_bytes(dtype: int) -> int:
    import onnx

    return {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.BOOL: 1,
        onnx.TensorProto.BFLOAT16: 2,
    }.get(dtype, 0)


def shape_count(shape: list[int] | None) -> int | None:
    if shape is None:
        return None
    result = 1
    for dim in shape:
        result *= dim
    return result


def block_from_name(name: str) -> str:
    import re

    match = re.search(r"/model\.(\d+)(?:/|$)", name)
    return f"model.{match.group(1)}" if match else "graph"


def model_audit(args: argparse.Namespace) -> None:
    import onnx

    model_path = Path(args.model).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(model_path), load_external_data=True)
    try:
        shaped = onnx.shape_inference.infer_shapes(model, data_prop=True)
        shape_status = "onnx_shape_inference_pass"
    except Exception as error:  # noqa: BLE001
        shaped = model
        shape_status = f"shape_inference_failed:{type(error).__name__}:{error}"

    value_map: dict[str, tuple[list[int] | None, int]] = {}
    for value in [*shaped.graph.input, *shaped.graph.value_info, *shaped.graph.output]:
        value_map[value.name] = (tensor_shape(value), int(value.type.tensor_type.elem_type))
    initializer_map = {value.name: value for value in shaped.graph.initializer}
    for name, value in initializer_map.items():
        value_map[name] = ([int(dim) for dim in value.dims], int(value.data_type))

    consumers: dict[str, list[int]] = collections.defaultdict(list)
    for index, node in enumerate(shaped.graph.node):
        for name in node.input:
            if name:
                consumers[name].append(index)
    last_use = {name: max(indices) for name, indices in consumers.items() if indices}

    constant_derived = set(initializer_map)
    constant_foldable = {
        "Cast", "Concat", "Constant", "DequantizeLinear", "Expand", "Flatten", "Gather",
        "Mul", "QuantizeLinear", "Reshape", "Shape", "Slice", "Split", "Squeeze",
        "Tile", "Transpose", "Unsqueeze",
    }
    for node in shaped.graph.node:
        inputs = [name for name in node.input if name]
        if node.op_type == "Constant" or (node.op_type in constant_foldable and inputs and all(name in constant_derived for name in inputs)):
            constant_derived.update(name for name in node.output if name)

    op_counts: collections.Counter[str] = collections.Counter()
    block_counts: collections.Counter[str] = collections.Counter()
    operator_rows: list[dict[str, Any]] = []
    mac_rows: list[dict[str, Any]] = []
    unresolved_shapes = 0
    total_macs = 0
    total_float_output_bytes = 0
    for index, node in enumerate(shaped.graph.node):
        op_counts[node.op_type] += 1
        block = block_from_name(node.name or (node.output[0] if node.output else ""))
        block_counts[block] += 1
        output_bytes = 0
        output_shapes: list[str] = []
        output_dtypes: list[str] = []
        for name in node.output:
            shape, dtype = value_map.get(name, (None, 0))
            count = shape_count(shape)
            size = element_bytes(dtype)
            if count is not None and size:
                output_bytes += count * size
                if dtype == onnx.TensorProto.FLOAT:
                    total_float_output_bytes += count * size
            else:
                unresolved_shapes += 1
            output_shapes.append("x".join(map(str, shape)) if shape is not None else "dynamic_or_unknown")
            output_dtypes.append(onnx.TensorProto.DataType.Name(dtype) if dtype else "UNKNOWN")

        macs = 0
        mac_status = "not_mac_op"
        if node.op_type == "Conv" and len(node.input) >= 2:
            x_shape = value_map.get(node.input[0], (None, 0))[0]
            w_shape = value_map.get(node.input[1], (None, 0))[0]
            y_shape = value_map.get(node.output[0], (None, 0))[0] if node.output else None
            group = 1
            for attr in node.attribute:
                if attr.name == "group":
                    group = int(attr.i)
            if x_shape and w_shape and y_shape and len(x_shape) == 4 and len(w_shape) == 4 and len(y_shape) == 4:
                macs = int(y_shape[0] * y_shape[1] * y_shape[2] * y_shape[3] * (x_shape[1] // group) * w_shape[2] * w_shape[3])
                mac_status = "static"
            else:
                mac_status = "unresolved_shape"
        elif node.op_type in {"MatMul", "Gemm"} and len(node.input) >= 2:
            a_shape = value_map.get(node.input[0], (None, 0))[0]
            b_shape = value_map.get(node.input[1], (None, 0))[0]
            y_shape = value_map.get(node.output[0], (None, 0))[0] if node.output else None
            if a_shape and b_shape and y_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
                k = int(a_shape[-1])
                count = shape_count(y_shape)
                if count is not None:
                    macs = int(count * k)
                    mac_status = "static"
            if macs == 0:
                mac_status = "unresolved_shape"
        total_macs += macs
        row = {
            "index": index,
            "block": block,
            "name": node.name,
            "op_type": node.op_type,
            "inputs": ";".join(node.input),
            "outputs": ";".join(node.output),
            "output_shapes": ";".join(output_shapes),
            "output_dtypes": ";".join(output_dtypes),
            "output_bytes": output_bytes,
            "macs": macs,
            "flops_2_per_mac": macs * 2,
            "mac_status": mac_status,
        }
        operator_rows.append(row)
        if node.op_type in {"Conv", "MatMul", "Gemm"}:
            mac_rows.append(row)

    active: dict[str, int] = {}
    active_all: dict[str, int] = {}
    live_rows: list[dict[str, Any]] = []
    for value in shaped.graph.input:
        if value.name in initializer_map:
            continue
        shape, dtype = value_map.get(value.name, (None, 0))
        count = shape_count(shape)
        if count is not None and element_bytes(dtype):
            active[value.name] = count * element_bytes(dtype)
            active_all[value.name] = count * element_bytes(dtype)
    peak_live = sum(active.values())
    peak_live_all = sum(active_all.values())
    peak_index = -1
    peak_names = sorted(active)
    for index, node in enumerate(shaped.graph.node):
        for name in node.output:
            if name in initializer_map:
                continue
            shape, dtype = value_map.get(name, (None, 0))
            count = shape_count(shape)
            if count is not None and element_bytes(dtype):
                active_all[name] = count * element_bytes(dtype)
                if name not in constant_derived:
                    active[name] = count * element_bytes(dtype)
        live = sum(active.values())
        live_all = sum(active_all.values())
        if live > peak_live:
            peak_live = live
            peak_index = index
            peak_names = sorted(active)
        live_rows.append({
            "node_index": index,
            "node_name": node.name,
            "block": block_from_name(node.name or (node.output[0] if node.output else "")),
            "live_bytes": live,
            "live_tensor_count": len(active),
            "live_all_noninitializer_bytes": live_all,
            "live_all_noninitializer_tensor_count": len(active_all),
        })
        peak_live_all = max(peak_live_all, live_all)
        for name in node.input:
            if last_use.get(name) == index:
                active.pop(name, None)
                active_all.pop(name, None)

    tensor_rows: list[dict[str, Any]] = []
    for name, (shape, dtype) in sorted(value_map.items()):
        count = shape_count(shape)
        size = count * element_bytes(dtype) if count is not None else None
        tensor_rows.append({
            "name": name,
            "shape": "x".join(map(str, shape)) if shape is not None else "dynamic_or_unknown",
            "dtype": onnx.TensorProto.DataType.Name(dtype) if dtype else "UNKNOWN",
            "bytes": size if size is not None else "unknown",
            "producer_block": block_from_name(name),
            "last_consumer_index": last_use.get(name, "none"),
            "is_initializer": int(name in initializer_map),
            "is_constant_derived": int(name in constant_derived),
        })

    write_tsv(out / "operator_manifest.tsv", operator_rows, list(operator_rows[0]))
    write_tsv(out / "macs_flops.tsv", mac_rows, list(operator_rows[0]))
    write_tsv(out / "memory_liveness.tsv", live_rows, list(live_rows[0]))
    write_tsv(out / "tensor_manifest.tsv", tensor_rows, list(tensor_rows[0]))
    summary = {
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "shape_status": shape_status,
        "node_count": len(shaped.graph.node),
        "initializer_count": len(shaped.graph.initializer),
        "op_counts": dict(sorted(op_counts.items())),
        "block_node_counts": dict(sorted(block_counts.items())),
        "conv_matmul_gemm_macs": total_macs,
        "flops_2_per_mac": total_macs * 2,
        "peak_live_activation_bytes": peak_live,
        "peak_live_all_noninitializer_bytes": peak_live_all,
        "peak_live_node_index": peak_index,
        "peak_live_tensor_names": peak_names,
        "materialized_float_output_bytes_sum": total_float_output_bytes,
        "unresolved_output_shapes_or_dtypes": unresolved_shapes,
        "input_contracts": [
            {"name": value.name, "shape": tensor_shape(value), "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)}
            for value in shaped.graph.input
        ],
        "output_contracts": [
            {"name": value.name, "shape": tensor_shape(value), "dtype": onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)}
            for value in shaped.graph.output
        ],
    }
    (out / "graph_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def make_session(model: Path, opt_level: str, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options.graph_optimization_level = levels[opt_level]
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = True
    options.enable_cpu_mem_arena = True
    return ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])


def letterbox(path: Path, size: int):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode {path}")
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_width) / 2.0
    pad_y = (size - new_height) / 2.0
    x0 = int(round(pad_x - 0.1))
    y0 = int(round(pad_y - 0.1))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[y0:y0 + new_height, x0:x0 + new_width] = resized
    tensor = np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1))[None].copy()
    return tensor, width, height, ratio, pad_x, pad_y


def accuracy_predict(args: argparse.Namespace) -> None:
    model = Path(args.model).resolve()
    images_dir = Path(args.images).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(images_dir.glob("*.jpg"))
    if args.limit > 0:
        images = images[: args.limit]
    session = make_session(model, args.opt_level, args.threads)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    durations: list[float] = []
    for index, path in enumerate(images):
        tensor, width, height, ratio, pad_x, pad_y = letterbox(path, args.imgsz)
        begin = time.perf_counter()
        raw = np.asarray(session.run([output_name], {input_name: tensor})[0], dtype=np.float32)
        durations.append((time.perf_counter() - begin) * 1.0e3)
        rows = raw[0] if raw.ndim == 3 else raw
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise RuntimeError(f"expected e2e [N,6] output, got {raw.shape}")
        image_id = int(path.stem)
        count = 0
        for x1, y1, x2, y2, score, class_id_raw in rows:
            score_f = float(score)
            class_id = int(class_id_raw)
            if score_f <= args.conf or not 0 <= class_id < 80:
                continue
            bx1 = float(np.clip((float(x1) - pad_x) / ratio, 0, width - 1))
            by1 = float(np.clip((float(y1) - pad_y) / ratio, 0, height - 1))
            bx2 = float(np.clip((float(x2) - pad_x) / ratio, 0, width - 1))
            by2 = float(np.clip((float(y2) - pad_y) / ratio, 0, height - 1))
            box_width = max(0.0, bx2 - bx1)
            box_height = max(0.0, by2 - by1)
            if box_width == 0.0 or box_height == 0.0:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": COCO_CATEGORY_IDS[class_id],
                "bbox": [bx1, by1, box_width, box_height],
                "score": score_f,
            })
            count += 1
        manifest.append({
            "index": index,
            "image_id": image_id,
            "path": str(path),
            "sha256": sha256_file(path),
            "width": width,
            "height": height,
            "detections": count,
            "inference_ms": durations[-1],
            "output_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        })
        if (index + 1) % args.log_every == 0 or index + 1 == len(images):
            print(f"progress={index + 1}/{len(images)} last_ms={durations[-1]:.3f}", flush=True)
    output.write_text(json.dumps(predictions, separators=(",", ":")) + "\n")
    write_tsv(output.with_suffix(".manifest.tsv"), manifest, list(manifest[0]) if manifest else ["index"])
    summary = {
        "model": str(model),
        "model_sha256": sha256_file(model),
        "runtime": "host_onnxruntime",
        "runtime_version": __import__("onnxruntime").__version__,
        "provider": "CPUExecutionProvider",
        "optimization": args.opt_level,
        "threads": args.threads,
        "images": len(images),
        "predictions": len(predictions),
        "confidence": args.conf,
        "preprocess": "letterbox_114_rgb_nchw_float32_div255_opencv_linear",
        "mean_inference_ms": statistics.fmean(durations) if durations else None,
        "stddev_inference_ms": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "predictions_sha256": sha256_file(output),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def accuracy_eval(args: argparse.Namespace) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    annotations = Path(args.annotations).resolve()
    predictions_path = Path(args.predictions).resolve()
    manifest_path = predictions_path.with_suffix(".manifest.tsv")
    with manifest_path.open() as stream:
        image_ids = [int(row["image_id"]) for row in csv.DictReader(stream, delimiter="\t")]
    coco = COCO(str(annotations))
    predictions = json.loads(predictions_path.read_text())
    result = coco.loadRes(predictions)
    evaluator = COCOeval(coco, result, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    valid_precision = precision[precision > -1]
    valid_recall = recall[recall > -1]
    categories = coco.loadCats(evaluator.params.catIds)
    per_class = []
    for class_index, category in enumerate(categories):
        values = precision[:, :, class_index, 0, 2]
        values = values[values > -1]
        per_class.append({
            "category_id": category["id"],
            "name": category["name"],
            "ap50_95": float(np.mean(values)) if values.size else float("nan"),
        })
    stats = evaluator.stats.tolist()
    summary = {
        "annotations": str(annotations),
        "annotations_sha256": sha256_file(annotations),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "image_count": len(image_ids),
        "prediction_count": len(predictions),
        "map50_95": stats[0],
        "map50": stats[1],
        "map75": stats[2],
        "ap_small": stats[3],
        "ap_medium": stats[4],
        "ap_large": stats[5],
        "ar_maxdet1": stats[6],
        "ar_maxdet10": stats[7],
        "ar_maxdet100": stats[8],
        "precision_mean_valid_cocoeval_grid": float(np.mean(valid_precision)) if valid_precision.size else None,
        "recall_mean_valid_cocoeval_grid": float(np.mean(valid_recall)) if valid_recall.size else None,
    }
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    write_tsv(out.with_suffix(".per_class.tsv"), per_class, list(per_class[0]))
    print(json.dumps(summary, indent=2))


def profile_summarize(args: argparse.Namespace) -> None:
    raw_rows: list[dict[str, Any]] = []
    for profile_name in args.profile:
        profile = Path(profile_name).resolve()
        profile_sha256 = sha256_file(profile)
        events = json.loads(profile.read_text())
        for event in events:
            event_args = event.get("args", {})
            if event.get("cat") != "Node" or "dur" not in event:
                continue
            name = str(event.get("name", ""))
            op = str(event_args.get("op_name", event_args.get("op", "unknown")))
            provider = str(event_args.get("provider", "unknown"))
            raw_rows.append({
                "profile": str(profile),
                "profile_sha256": profile_sha256,
                "name": name,
                "block": block_from_name(name),
                "op_type": op,
                "provider": provider,
                "duration_us": float(event["dur"]),
                "thread_id": event.get("tid", ""),
            })
    aggregates: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for row in raw_rows:
        aggregates[(row["block"], row["op_type"], row["provider"])].append(row["duration_us"])
    summary_rows = []
    for (block, op, provider), durations in sorted(aggregates.items()):
        summary_rows.append({
            "block": block,
            "op_type": op,
            "provider": provider,
            "events": len(durations),
            "total_us": sum(durations),
            "mean_us": statistics.fmean(durations),
            "median_us": statistics.median(durations),
            "p95_us": float(np.percentile(durations, 95)),
        })
    output = Path(args.output).resolve()
    write_tsv(output, raw_rows, list(raw_rows[0]) if raw_rows else ["profile"])
    write_tsv(output.with_name(output.stem + "_summary.tsv"), summary_rows,
              list(summary_rows[0]) if summary_rows else ["block"])
    print(f"node_events={len(raw_rows)} groups={len(summary_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("model-audit")
    command.add_argument("--model", required=True)
    command.add_argument("--out-dir", required=True)
    command.set_defaults(func=model_audit)

    command = sub.add_parser("accuracy-predict")
    command.add_argument("--model", required=True)
    command.add_argument("--images", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--opt-level", choices=["disable", "basic", "extended", "all"], required=True)
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--limit", type=int, default=500)
    command.add_argument("--imgsz", type=int, default=640)
    command.add_argument("--conf", type=float, default=0.001)
    command.add_argument("--log-every", type=int, default=25)
    command.set_defaults(func=accuracy_predict)

    command = sub.add_parser("accuracy-eval")
    command.add_argument("--annotations", required=True)
    command.add_argument("--predictions", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=accuracy_eval)

    command = sub.add_parser("profile-summarize")
    command.add_argument("--profile", action="append", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=profile_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
