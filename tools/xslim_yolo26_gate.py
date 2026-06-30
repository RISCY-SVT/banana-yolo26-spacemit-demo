#!/usr/bin/env python3
"""@file xslim_yolo26_gate.py
@brief XSlim candidate generation, graph inspection, and CPU oracle checks.
@details This R&D helper keeps XSlim configuration and validation reproducible
without touching the frozen YOLO11 production repository. It supports YOLO26
end-to-end `[1,300,6]` outputs and traditional `[1,84,8400]` outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnx
import onnxruntime as ort


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Candidate:
    """One model candidate and its expected output contract."""

    name: str
    path: Path
    contract: str


def sha256_file(path: Path) -> str:
    """Return SHA256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_table(path: Path, rows: Iterable[dict], fields: list[str], delimiter: str = "\t") -> None:
    """Write a TSV or Markdown table based on the file suffix."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".md":
        with path.open("w") as f:
            f.write("| " + " | ".join(fields) + " |\n")
            f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(str(row.get(k, "")).replace("\n", " ") for k in fields) + " |\n")
    else:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)


def parse_candidate(text: str) -> Candidate:
    """Parse `name:path:contract` candidate syntax."""
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"candidate must be name:path:contract, got {text}")
    return Candidate(parts[0], Path(parts[1]).resolve(), parts[2])


def load_manifest(path: Path) -> list[dict]:
    """Load a YOLO26 standard sanity manifest."""
    return json.loads((path / "manifest.json").read_text())


def letterbox(path: Path, size: int = 640) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return source image, NCHW input, scale, x pad, and y pad."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    src_h, src_w = image.shape[:2]
    ratio = min(size / src_w, size / src_h)
    new_w = int(round(src_w * ratio))
    new_h = int(round(src_h * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    x0 = int(round(pad_x - 0.1))
    y0 = int(round(pad_y - 0.1))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...].copy()
    return image, nchw, ratio, pad_x, pad_y


def unletterbox_xyxy(box: list[float], ratio: float, pad_x: float, pad_y: float, shape: tuple[int, int, int]) -> list[float]:
    """Map model-space xyxy to original image-space xyxy."""
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return [
        float(np.clip((x1 - pad_x) / ratio, 0, w - 1)),
        float(np.clip((y1 - pad_y) / ratio, 0, h - 1)),
        float(np.clip((x2 - pad_x) / ratio, 0, w - 1)),
        float(np.clip((y2 - pad_y) / ratio, 0, h - 1)),
    ]


def nms_classwise(dets: list[dict], iou_thres: float = 0.7, max_det: int = 300) -> list[dict]:
    """Apply small class-wise NMS for traditional YOLO output."""
    selected: list[dict] = []
    for class_id in sorted({d["class_id"] for d in dets}):
        cls_dets = sorted([d for d in dets if d["class_id"] == class_id], key=lambda x: x["score"], reverse=True)
        while cls_dets and len(selected) < max_det:
            best = cls_dets.pop(0)
            selected.append(best)
            keep = []
            bx1, by1, bx2, by2 = best["xyxy"]
            b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            for det in cls_dets:
                x1, y1, x2, y2 = det["xyxy"]
                inter_w = max(0.0, min(bx2, x2) - max(bx1, x1))
                inter_h = max(0.0, min(by2, y2) - max(by1, y1))
                inter = inter_w * inter_h
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                union = b_area + area - inter
                iou = inter / union if union > 0 else 0.0
                if iou <= iou_thres:
                    keep.append(det)
            cls_dets = keep
    return sorted(selected, key=lambda x: x["score"], reverse=True)[:max_det]


def decode_output(output: np.ndarray, contract: str, ratio: float, pad_x: float, pad_y: float, shape: tuple[int, int, int], conf: float, iou: float) -> list[dict]:
    """Decode E2E or traditional YOLO26 output."""
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if contract == "auto":
        if arr.ndim == 2 and arr.shape[-1] == 6:
            contract = "e2e"
        elif arr.ndim == 2 and arr.shape[0] >= 84:
            contract = "traditional"
    dets: list[dict] = []
    if contract == "e2e":
        if arr.ndim != 2 or arr.shape[-1] != 6:
            raise RuntimeError(f"expected e2e [N,6], got {list(np.asarray(output).shape)}")
        for x1, y1, x2, y2, score, cls in arr:
            if float(score) <= conf:
                continue
            dets.append({
                "xyxy": unletterbox_xyxy([float(x1), float(y1), float(x2), float(y2)], ratio, pad_x, pad_y, shape),
                "score": float(score),
                "class_id": int(cls),
            })
        return dets
    if contract == "traditional":
        if arr.ndim != 2 or arr.shape[0] < 84:
            raise RuntimeError(f"expected traditional [84,N], got {list(np.asarray(output).shape)}")
        boxes = arr[:4, :].T
        scores = arr[4:84, :].T
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(scores.shape[0]), class_ids]
        for box, score, class_id in zip(boxes, confs, class_ids):
            if float(score) <= conf:
                continue
            cx, cy, bw, bh = [float(v) for v in box]
            xyxy = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]
            dets.append({
                "xyxy": unletterbox_xyxy(xyxy, ratio, pad_x, pad_y, shape),
                "score": float(score),
                "class_id": int(class_id),
            })
        return nms_classwise(dets, iou)
    raise RuntimeError(f"unsupported contract={contract}")


def draw_detections(image_path: Path, detections: list[dict], out_path: Path) -> None:
    """Write a compact annotated output image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        label = COCO80[det["class_id"]] if 0 <= det["class_id"] < len(COCO80) else str(det["class_id"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (20, 220, 80), 2)
        cv2.putText(image, f"{label} {det['score']:.2f}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 220, 80), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def make_configs(args: argparse.Namespace) -> None:
    """Generate bounded XSlim JSON configs for YOLO26."""
    repo = Path(args.repo).resolve()
    out_dir = Path(args.out_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    preprocess = (repo / "tools/xslim_yolo26_letterbox_preprocess.py:preprocess_impl").resolve()
    small = (repo / ".deps/datasets/xslim_yolo26_calib/calib_small.txt").resolve()
    rep = (repo / ".deps/datasets/xslim_yolo26_calib/calib_representative.txt").resolve()
    e2e = (repo / ".deps/probes/models_forensics/yolo26n_latest_e2e640.onnx").resolve()
    trad = (repo / ".deps/probes/models_forensics/yolo26n_latest_traditional640.onnx").resolve()
    specs = [
        ("e2e_small_default", e2e, small, False, "default", 0, 1),
        ("e2e_rep_default", e2e, rep, False, "default", 0, 1),
        ("e2e_small_skip_onnxsim", e2e, small, True, "default", 0, 1),
        ("e2e_rep_precision1", e2e, rep, False, "default", 1, 1),
        ("traditional_small_default", trad, small, False, "default", 0, 1),
        ("traditional_rep_default", trad, rep, False, "default", 0, 1),
        ("e2e_small_minmax", e2e, small, False, "minmax", 0, 1),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model, data_list, skip_sim, calib_type, precision_level, finetune_level in specs:
        cfg = {
            "model_parameters": {
                "onnx_model": str(model),
                "output_prefix": name,
                "working_dir": str(work_dir / name),
                "skip_onnxsim": skip_sim,
            },
            "calibration_parameters": {
                "calibration_step": sum(1 for _ in data_list.open()),
                "calibration_batch_size": 1,
                "calibration_device": "cpu",
                "calibration_type": calib_type,
                "input_parameters": [{
                    "input_name": "images",
                    "input_shape": [1, 3, 640, 640],
                    "file_type": "img",
                    "color_format": "rgb",
                    "mean_value": [0.0, 0.0, 0.0],
                    "std_value": [1.0, 1.0, 1.0],
                    "preprocess_file": str(preprocess),
                    "data_list_path": str(data_list),
                }],
            },
            "quantization_parameters": {
                "precision_level": precision_level,
                "finetune_level": finetune_level,
                "analysis_enable": False,
            },
        }
        cfg_path = out_dir / f"{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        rows.append({
            "name": name,
            "model": str(model),
            "data_list": str(data_list),
            "skip_onnxsim": skip_sim,
            "calibration_type": calib_type,
            "precision_level": precision_level,
            "finetune_level": finetune_level,
            "config": str(cfg_path),
            "expected_output": str(work_dir / name / f"{name}.onnx"),
        })
    fields = list(rows[0])
    write_table(out_dir / "xslim_config_matrix.tsv", rows, fields)
    write_table(out_dir / "xslim_config_matrix.md", rows, fields)
    print(f"configs={len(rows)}")
    print(f"config_dir={out_dir}")


def inspect_models(args: argparse.Namespace) -> None:
    """Inspect candidate ONNX graphs."""
    rows = []
    for cand in [parse_candidate(x) for x in args.candidate]:
        model = onnx.load(cand.path)
        ops = Counter(node.op_type for node in model.graph.node)
        conv_kernel_shape = 0
        first_conv = ""
        for node in model.graph.node:
            if node.op_type == "Conv":
                if not first_conv:
                    first_conv = node.name
                if any(attr.name == "kernel_shape" for attr in node.attribute):
                    conv_kernel_shape += 1
        inputs = []
        outputs = []
        for value in model.graph.input:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim]
            inputs.append(f"{value.name}:{dims}")
        for value in model.graph.output:
            dims = [d.dim_value if d.dim_value else d.dim_param for d in value.type.tensor_type.shape.dim]
            outputs.append(f"{value.name}:{dims}")
        rows.append({
            "name": cand.name,
            "contract": cand.contract,
            "path": str(cand.path),
            "sha256": sha256_file(cand.path),
            "bytes": cand.path.stat().st_size,
            "opset": ",".join(f"{op.domain or 'ai.onnx'}:{op.version}" for op in model.opset_import),
            "inputs": "; ".join(inputs),
            "outputs": "; ".join(outputs),
            "nodes": len(model.graph.node),
            "QuantizeLinear": ops.get("QuantizeLinear", 0),
            "DequantizeLinear": ops.get("DequantizeLinear", 0),
            "QLinearConv": ops.get("QLinearConv", 0),
            "QLinearMatMul": ops.get("QLinearMatMul", 0),
            "Conv": ops.get("Conv", 0),
            "MatMul": ops.get("MatMul", 0),
            "YoloDecode": ops.get("YoloDecode", 0),
            "conv_with_kernel_shape": conv_kernel_shape,
            "first_conv": first_conv,
            "op_histogram": json.dumps(dict(sorted(ops.items())), sort_keys=True),
        })
    fields = list(rows[0]) if rows else []
    write_table(Path(args.out_tsv), rows, fields)
    write_table(Path(args.out_md), rows, fields)
    print(f"inspected={len(rows)}")


def cpu_oracle(args: argparse.Namespace) -> None:
    """Run ONNX Runtime CPU oracle for XSlim candidates."""
    suite = load_manifest(Path(args.suite))
    if args.public_only:
        suite = [row for row in suite if row.get("visibility") != "private-reference"]
    if args.limit:
        suite = suite[: args.limit]
    out = Path(args.out).resolve()
    rows = []
    for cand in [parse_candidate(x) for x in args.candidate]:
        session = ort.InferenceSession(str(cand.path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        for entry in suite:
            image_path = Path(entry["path"])
            src, tensor, ratio, pad_x, pad_y = letterbox(image_path, args.imgsz)
            input_bin = out / "input_bins" / f"{entry['image_id']}_640_nchw_f32.bin"
            input_bin.parent.mkdir(parents=True, exist_ok=True)
            tensor.astype(np.float32).tofile(input_bin)
            try:
                output = session.run([output_name], {input_name: tensor})[0].astype(np.float32)
                dets = decode_output(output, cand.contract, ratio, pad_x, pad_y, src.shape, args.conf, args.iou)
                out_bin = out / "raw_outputs" / f"{cand.name}_{entry['image_id']}_cpu.bin"
                out_bin.parent.mkdir(parents=True, exist_ok=True)
                output.tofile(out_bin)
                out_img = out / "xslim_cpu" / f"{cand.name}_{entry['image_id']}.jpg"
                draw_detections(image_path, dets, out_img)
                classes = sorted({COCO80[d["class_id"]] if 0 <= d["class_id"] < len(COCO80) else str(d["class_id"]) for d in dets})
                top = ";".join(
                    f"{COCO80[d['class_id']] if 0 <= d['class_id'] < len(COCO80) else d['class_id']}:{d['score']:.3f}"
                    for d in sorted(dets, key=lambda d: d["score"], reverse=True)[:5]
                )
                verdict = "pass" if (entry["image_id"].startswith("blank") and len(dets) == 0) or not entry["image_id"].startswith("blank") else "review"
                rows.append({
                    "candidate": cand.name,
                    "image_id": entry["image_id"],
                    "contract": cand.contract,
                    "status": "ok",
                    "output_shape": "x".join(str(x) for x in output.shape),
                    "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                    "object_count": len(dets),
                    "classes": ",".join(classes),
                    "top_detections": top,
                    "verdict": verdict,
                    "output_image": str(out_img),
                    "raw_output": str(out_bin),
                    "input_bin": str(input_bin),
                    "error": "",
                })
            except Exception as exc:  # noqa: BLE001 - oracle matrix should preserve candidate failures.
                rows.append({
                    "candidate": cand.name,
                    "image_id": entry["image_id"],
                    "contract": cand.contract,
                    "status": "fail",
                    "output_shape": "",
                    "output_sha256": "",
                    "object_count": "",
                    "classes": "",
                    "top_detections": "",
                    "verdict": "fail",
                    "output_image": "",
                    "raw_output": "",
                    "input_bin": str(input_bin),
                    "error": repr(exc),
                })
    fields = [
        "candidate", "image_id", "contract", "status", "output_shape", "output_sha256", "object_count",
        "classes", "top_detections", "verdict", "output_image", "raw_output", "input_bin", "error",
    ]
    write_table(out / "xslim_cpu_oracle_matrix.tsv", rows, fields)
    write_table(out / "xslim_cpu_oracle_matrix.md", rows, fields)
    print(f"cpu_oracle_rows={len(rows)}")


def decode_board(args: argparse.Namespace) -> None:
    """Decode board output bins produced by `rt204_tensor_probe`."""
    suite = load_manifest(Path(args.suite))
    if args.public_only:
        suite = [row for row in suite if row.get("visibility") != "private-reference"]
    out = Path(args.out).resolve()
    board = Path(args.board_outputs).resolve()
    rows = []
    for cand in [parse_candidate(x) for x in args.candidate]:
        for entry in suite:
            output_bin = board / f"{cand.name}_{entry['image_id']}_rt204_ep.bin"
            if not output_bin.exists():
                continue
            image_path = Path(entry["path"])
            src, _, ratio, pad_x, pad_y = letterbox(image_path, args.imgsz)
            data = np.fromfile(output_bin, dtype=np.float32)
            if cand.contract == "traditional":
                output = data.reshape((1, 84, -1))
            else:
                output = data.reshape((1, -1, 6))
            dets = decode_output(output, cand.contract, ratio, pad_x, pad_y, src.shape, args.conf, args.iou)
            out_img = out / "xslim_rt204" / f"{cand.name}_{entry['image_id']}.jpg"
            draw_detections(image_path, dets, out_img)
            classes = sorted({COCO80[d["class_id"]] if 0 <= d["class_id"] < len(COCO80) else str(d["class_id"]) for d in dets})
            rows.append({
                "candidate": cand.name,
                "image_id": entry["image_id"],
                "contract": cand.contract,
                "status": "ok",
                "object_count": len(dets),
                "classes": ",".join(classes),
                "output_image": str(out_img),
                "raw_output": str(output_bin),
            })
    fields = ["candidate", "image_id", "contract", "status", "object_count", "classes", "output_image", "raw_output"]
    write_table(out / "xslim_rt204_decode_matrix.tsv", rows, fields)
    write_table(out / "xslim_rt204_decode_matrix.md", rows, fields)
    print(f"decoded_board_rows={len(rows)}")


def main() -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-configs")
    p.add_argument("--repo", default=".")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.set_defaults(func=make_configs)

    p = sub.add_parser("inspect")
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-tsv", required=True)
    p.set_defaults(func=inspect_models)

    p = sub.add_parser("cpu-oracle")
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--public-only", action="store_true")
    p.set_defaults(func=cpu_oracle)

    p = sub.add_parser("decode-board")
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--board-outputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--public-only", action="store_true")
    p.set_defaults(func=decode_board)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
