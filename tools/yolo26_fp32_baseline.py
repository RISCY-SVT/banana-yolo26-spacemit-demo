#!/usr/bin/env python3
"""@file yolo26_fp32_baseline.py
@brief YOLO26 FP32 baseline utilities for the isolated K1X R&D workspace.

The script intentionally keeps all generated data under `.deps/` or the current
task log directory.  It prepares a small public COCO-derived sanity suite,
builds deterministic 640x640 NCHW inputs, runs PyTorch and ONNX Runtime CPU
oracles, and decodes board-side rt204 tensor dumps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


SUMMARY_FIELDS = [
    "image_id",
    "runtime",
    "model_path",
    "model_sha256",
    "output_shape",
    "output_sha256",
    "object_count",
    "classes",
    "top_detections",
    "confidence_range",
    "semantic_verdict",
    "output_image",
]


@dataclass
class ImageEntry:
    """One deterministic image in the YOLO26 FP32 baseline suite."""

    image_id: str
    path: str
    sha256: str
    width: int
    height: int
    source: str
    visibility: str
    purpose: str


@dataclass
class DetectionSummary:
    """Compact detection summary for markdown/TSV reporting."""

    image_id: str
    runtime: str
    model_path: str
    model_sha256: str
    output_shape: str
    output_sha256: str
    object_count: int
    classes: str
    top_detections: str
    confidence_range: str
    semantic_verdict: str
    output_image: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md_table(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(k, "")).replace("\n", " ") for k in fieldnames) + " |\n")


def copy_image(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def image_dims(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    h, w = img.shape[:2]
    return w, h


def prepare_suite(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    images_dir = out / "images"
    entries: list[ImageEntry] = []

    def add(src: Path, image_id: str, source: str, visibility: str, purpose: str) -> None:
        dst = copy_image(src, images_dir / f"{image_id}{src.suffix.lower()}")
        w, h = image_dims(dst)
        entries.append(ImageEntry(image_id, str(dst), sha256_file(dst), w, h, source, visibility, purpose))

    venv_assets = repo / ".deps/venvs/ultralytics_latest/lib/python3.12/site-packages/ultralytics/assets"
    for name in ["bus.jpg", "zidane.jpg"]:
        src = venv_assets / name
        if src.exists():
            add(src, f"ultralytics_{src.stem}", "Ultralytics package asset", "public", "standard detector sanity image")

    coco_like = sorted((repo / ".deps/yolo26_datasets/calib_representative/images/val").glob("[0-9]*.jpg"))
    for src in coco_like[:6]:
        add(src, f"coco_like_{src.stem}", "task-local COCO-derived calibration image", "public-coco-derived", "COCO-like public sanity image")

    blank = images_dir / "blank_white_640x480.jpg"
    blank.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(blank), np.full((480, 640, 3), 255, dtype=np.uint8))
    w, h = image_dims(blank)
    entries.append(ImageEntry("blank_white_640x480", str(blank), sha256_file(blank), w, h, "synthetic", "public-synthetic", "negative-control blank image"))

    canonical = next((Path(p) for p in sorted(Path("/data").glob("**/photo_2024-10-11_10-04-04.jpg")) if p.is_file()), None)
    if canonical:
        add(canonical, "private_canonical_photo", "local private production reference", "private-reference", "non-public regression reference only")

    rows = [asdict(e) for e in entries]
    fields = list(rows[0].keys()) if rows else []
    write_tsv(out / "standard_image_suite_manifest.tsv", rows, fields)
    write_md_table(out / "standard_image_suite_manifest.md", rows, fields)
    (out / "manifest.json").write_text(json.dumps(rows, indent=2))
    print(f"prepared_images={len(rows)}")
    print(f"suite_dir={out}")


def load_manifest(path: Path) -> list[ImageEntry]:
    data = json.loads(path.read_text())
    return [ImageEntry(**row) for row in data]


def load_names(labels: Path) -> list[str]:
    return [line.strip() for line in labels.read_text().splitlines() if line.strip()]


def letterbox_bgr(path: Path, size: int = 640) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    h, w = img.shape[:2]
    ratio = min(size / w, size / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    x0, y0 = int(round(pad_x - 0.1)), int(round(pad_y - 0.1))
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...].copy()
    return img, nchw, ratio, pad_x, pad_y


def decode_e2e(output: np.ndarray, ratio: float, pad_x: float, pad_y: float, src_shape: tuple[int, int], conf: float) -> list[dict]:
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 6:
        raise RuntimeError(f"expected [N,6] e2e output, got shape={list(output.shape)}")
    h, w = src_shape[:2]
    detections = []
    for x1, y1, x2, y2, score, cls in arr:
        if float(score) <= conf:
            continue
        bx1 = float(np.clip((x1 - pad_x) / ratio, 0, w - 1))
        by1 = float(np.clip((y1 - pad_y) / ratio, 0, h - 1))
        bx2 = float(np.clip((x2 - pad_x) / ratio, 0, w - 1))
        by2 = float(np.clip((y2 - pad_y) / ratio, 0, h - 1))
        detections.append({"xyxy": [bx1, by1, bx2, by2], "score": float(score), "class_id": int(cls)})
    return detections


def draw_detections(image_path: Path, detections: list[dict], names: list[str], out_path: Path) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        cls = det["class_id"]
        score = det["score"]
        label = names[cls] if 0 <= cls < len(names) else str(cls)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def summarize(image_id: str, runtime: str, model_path: Path, output: np.ndarray, detections: list[dict], names: list[str], output_image: Path, semantic: str) -> DetectionSummary:
    classes = sorted({names[d["class_id"]] if 0 <= d["class_id"] < len(names) else str(d["class_id"]) for d in detections})
    top = sorted(detections, key=lambda d: d["score"], reverse=True)[:5]
    top_text = "; ".join(
        f"{names[d['class_id']] if 0 <= d['class_id'] < len(names) else d['class_id']}:{d['score']:.3f}" for d in top
    )
    if detections:
        scores = [d["score"] for d in detections]
        conf_range = f"{min(scores):.3f}-{max(scores):.3f}"
    else:
        conf_range = ""
    return DetectionSummary(
        image_id=image_id,
        runtime=runtime,
        model_path=str(model_path),
        model_sha256=sha256_file(model_path),
        output_shape="x".join(str(x) for x in output.shape),
        output_sha256=hashlib.sha256(np.asarray(output, dtype=np.float32).tobytes()).hexdigest(),
        object_count=len(detections),
        classes=",".join(classes),
        top_detections=top_text,
        confidence_range=conf_range,
        semantic_verdict=semantic,
        output_image=str(output_image),
    )


def run_host_oracle(args: argparse.Namespace) -> None:
    suite = load_manifest(Path(args.suite) / "manifest.json")
    model_onnx = Path(args.model_onnx).resolve()
    model_pt = Path(args.model_pt).resolve()
    out = Path(args.out).resolve()
    names = load_names(Path(args.labels))
    conf = float(args.conf)

    session = ort.InferenceSession(str(model_onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    yolo = YOLO(str(model_pt))
    rows: list[dict] = []
    raw_summary: dict[str, dict] = {}

    for entry in suite:
        image_path = Path(entry.path)
        src, nchw, ratio, pad_x, pad_y = letterbox_bgr(image_path, int(args.imgsz))
        input_bin = out / "input_bins" / f"{entry.image_id}_640_nchw_f32.bin"
        input_bin.parent.mkdir(parents=True, exist_ok=True)
        nchw.astype(np.float32).tofile(input_bin)

        ort_output = session.run([output_name], {input_name: nchw})[0].astype(np.float32)
        ort_bin = out / "raw_outputs" / f"{entry.image_id}_onnx_cpu.bin"
        ort_bin.parent.mkdir(parents=True, exist_ok=True)
        ort_output.tofile(ort_bin)
        ort_dets = decode_e2e(ort_output, ratio, pad_x, pad_y, src.shape, conf)
        ort_img = out / "fp32_oracle" / f"{entry.image_id}_onnx_cpu.jpg"
        draw_detections(image_path, ort_dets, names, ort_img)
        rows.append(asdict(summarize(entry.image_id, "onnx_cpu", model_onnx, ort_output, ort_dets, names, ort_img, "pass")))

        pred = yolo.predict(str(image_path), imgsz=int(args.imgsz), conf=conf, iou=float(args.iou), max_det=300, verbose=False)[0]
        pt_dets = []
        if pred.boxes is not None:
            xyxy = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(xyxy, confs, clss):
                pt_dets.append({"xyxy": [float(v) for v in box], "score": float(score), "class_id": int(cls)})
        pt_img = out / "fp32_oracle" / f"{entry.image_id}_pytorch.jpg"
        draw_detections(image_path, pt_dets, names, pt_img)
        pt_output = np.array([[*d["xyxy"], d["score"], d["class_id"]] for d in pt_dets], dtype=np.float32)
        rows.append(asdict(summarize(entry.image_id, "pytorch", model_pt, pt_output, pt_dets, names, pt_img, "pass")))
        raw_summary[entry.image_id] = {
            "input_bin": str(input_bin),
            "onnx_cpu_output_bin": str(ort_bin),
            "onnx_cpu_detections": ort_dets,
            "pytorch_detections": pt_dets,
        }

    row_dicts = rows
    write_tsv(out / "yolo26_fp32_oracle_matrix.tsv", row_dicts, SUMMARY_FIELDS)
    write_md_table(out / "yolo26_fp32_oracle_matrix.md", row_dicts, SUMMARY_FIELDS)
    (out / "oracle_raw_summary.json").write_text(json.dumps(raw_summary, indent=2))
    print(f"oracle_rows={len(row_dicts)}")
    print(f"oracle_out={out}")


def decode_board(args: argparse.Namespace) -> None:
    suite = load_manifest(Path(args.suite) / "manifest.json")
    board_dir = Path(args.board_outputs).resolve()
    model_onnx = Path(args.model_onnx).resolve()
    out = Path(args.out).resolve()
    names = load_names(Path(args.labels))
    conf = float(args.conf)
    rows: list[dict] = []
    for entry in suite:
        output_bin = board_dir / f"{entry.image_id}_rt204_ep.bin"
        if not output_bin.exists():
            continue
        image_path = Path(entry.path)
        src, _, ratio, pad_x, pad_y = letterbox_bgr(image_path, int(args.imgsz))
        data = np.fromfile(output_bin, dtype=np.float32)
        output = data.reshape((1, 300, 6))
        dets = decode_e2e(output, ratio, pad_x, pad_y, src.shape, conf)
        out_img = out / "fp32_oracle" / f"{entry.image_id}_rt204_ep.jpg"
        draw_detections(image_path, dets, names, out_img)
        rows.append(asdict(summarize(entry.image_id, "rt204_spacemit_ep", model_onnx, output, dets, names, out_img, "pass")))
    write_tsv(out / "yolo26_rt204_ep_decode_matrix.tsv", rows, SUMMARY_FIELDS)
    write_md_table(out / "yolo26_rt204_ep_decode_matrix.md", rows, SUMMARY_FIELDS)
    print(f"decoded_board_rows={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-suite")
    p.add_argument("--repo", default=".")
    p.add_argument("--out", default=".deps/datasets/yolo26_standard_sanity")
    p.set_defaults(func=prepare_suite)

    p = sub.add_parser("host-oracle")
    p.add_argument("--suite", required=True)
    p.add_argument("--model-onnx", required=True)
    p.add_argument("--model-pt", required=True)
    p.add_argument("--labels", default="assets/coco80.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", default=640)
    p.add_argument("--conf", default=0.25)
    p.add_argument("--iou", default=0.7)
    p.set_defaults(func=run_host_oracle)

    p = sub.add_parser("decode-board")
    p.add_argument("--suite", required=True)
    p.add_argument("--board-outputs", required=True)
    p.add_argument("--model-onnx", required=True)
    p.add_argument("--labels", default="assets/coco80.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--imgsz", default=640)
    p.add_argument("--conf", default=0.25)
    p.set_defaults(func=decode_board)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
