#!/usr/bin/env python3
"""Unified evaluator for context-memory generated videos.

The evaluator uses human reference annotations for the context target:

* reference keyframe + bbox from eval_annotations.jsonl
* OWL-ViT only proposes candidate objects in generated frames
* SAM/SAM2 turns boxes into object masks
* DINOv2, masked color histograms, and a simple masked pixel score compare
  each generated candidate against the reference target

`--scoring-path owl_sam_dino` forces the object-localisation path for every
reference; `--scoring-path direct_dino` compares the reference bbox at the
corresponding generated-frame location without OWL/SAM. `auto` now follows the
same bbox-seeded SAM/SAM2 path as `owl_sam_dino`; the historic 80%-area
full-frame fallback is intentionally disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_MODEL_ROOT = Path(
    os.environ.get("CONTEXT_MEMORY_MODEL_ROOT", _PROJECT_ROOT / "models")
)

try:
    from compute_context_memory_prune_ratio_strict import method_prune as strict_method_prune
except ImportError:
    try:
        from tools.compute_context_memory_prune_ratio_strict import (
            method_prune as strict_method_prune,
        )
    except ImportError:
        strict_method_prune = None

try:
    # q0 is the only cross-method compression denominator used by the current
    # context-memory workflow.  The legacy score-mask statistic is retained
    # separately below solely for historical comparison.
    from account_context_memory_q0 import _case_row as q0_case_row
except ImportError:
    q0_case_row = None


DEFAULT_METHODS = [
    "full_kv",
    "consistency_prune",
    "native_longlive_streaming",
    "dummy",
    "forcingkv",
    "longlive_rag",
]

METHOD_LABELS = {
    "full_kv": "fullKV",
    "ours": "ours",
    "consistency_prune": "ours",
    "consistency_prune_fkv": "ours_fkv",
    "consistency_prune_random": "ours_random",
    "native_longlive_streaming": "streaming",
    "dummy": "dummy",
    "forcingkv": "ForcingKV",
    "patchify_retrieve": "Patchify-Retrieve",
    "longlive_rag": "LongLive-RAG",
}

METHOD_ALIASES = {
    "fullkv": "full_kv",
    "full_kv": "full_kv",
    "ours": "consistency_prune",
    "ours_new": "consistency_prune",
    "ours_fkv": "consistency_prune_fkv",
    "consistency_prune_fkv": "consistency_prune_fkv",
    "ours_random": "consistency_prune_random",
    "consistency_prune_random": "consistency_prune_random",
    "consistency_prune": "consistency_prune",
    "streaming": "native_longlive_streaming",
    "native_longlive_streaming": "native_longlive_streaming",
    "longliverag": "longlive_rag",
    "longlive_rag": "longlive_rag",
    "dummy": "dummy",
    "dummyforcing": "dummy",
    "forcingkv": "forcingkv",
    "patchify_retrieve": "patchify_retrieve",
}


def canonical_method(method: Any) -> str:
    value = str(method or "").strip()
    return METHOD_ALIASES.get(value, value)

FIXED_LOCAL_ATTN_SIZE = {
    "native_longlive_streaming": 12,
    "dummy": 12,
    "forcingkv": 12,
    "longlive_rag": 12,
}

RAG_MEMORY_SIZE = {
    "longlive_rag": 6,
}

FIXED_SINK_SIZE = {
    "native_longlive_streaming": 3,
    "dummy": 3,
    "forcingkv": 1,
    "longlive_rag": 1,
}


def import_runtime_deps() -> None:
    """Import heavy video/model dependencies after argparse has handled --help."""
    global cv2, np, torch, F, Image
    global AutoImageProcessor, AutoModel, OwlViTForObjectDetection, OwlViTProcessor, Sam2Model, Sam2Processor
    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    import torch.nn.functional as _F
    from PIL import Image as _Image
    from transformers import (
        AutoImageProcessor as _AutoImageProcessor,
        AutoModel as _AutoModel,
        OwlViTForObjectDetection as _OwlViTForObjectDetection,
        OwlViTProcessor as _OwlViTProcessor,
        Sam2Model as _Sam2Model,
        Sam2Processor as _Sam2Processor,
    )

    cv2 = _cv2
    np = _np
    torch = _torch
    F = _F
    Image = _Image
    AutoImageProcessor = _AutoImageProcessor
    AutoModel = _AutoModel
    OwlViTForObjectDetection = _OwlViTForObjectDetection
    OwlViTProcessor = _OwlViTProcessor
    Sam2Model = _Sam2Model
    Sam2Processor = _Sam2Processor


@dataclass
class Reference:
    ref_id: str
    ref_role: str
    ref_frame: int
    ref_box_xyxy: tuple[int, int, int, int]
    ref_fps: float | None = None
    ref_mask_path: str | None = None
    ref_mask_source: str = ""
    ref_sam_iou: float | None = None


@dataclass
class Annotation:
    selected_case_id: str
    source_case_id: str | None
    clip_file: str
    prompt: str
    domain_category: str
    memory_level: str
    object_subtype: str
    background_motion: str
    length_bucket: str
    target_label: str
    target_aliases: list[str]
    validation: bool
    references: list[Reference]

    @property
    def match_mode(self) -> str:
        return "semantic_color" if self.domain_category == "animal" else "appearance_strict"

    @property
    def texts(self) -> list[str]:
        seen: set[str] = set()
        values: list[str] = []
        for item in [self.target_label, *self.target_aliases]:
            text = str(item).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                values.append(text)
        return values or ["object"]


@dataclass
class Candidate:
    box: tuple[int, int, int, int]
    class_score: float
    label: str


@dataclass
class Job:
    selected_case_id: str
    output_case_id: str
    prompt_variant: str
    profile: str
    method: str
    output_dir: Path
    video_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


@dataclass
class ScoredCandidate:
    candidate: Candidate | None
    mask: np.ndarray | None
    sam_iou: float
    class_score: float
    dino_score: float
    color_score: float
    pixel_score: float
    final_score: float
    detail_score: float
    memory_label: str
    memory_score: float


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_metadata_cases(metadata_path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(metadata_path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id:
            out[case_id] = row
        source_case_id = str(row.get("source_case_id") or "")
        if source_case_id:
            out[source_case_id] = row
    return out


def parse_references(row: dict[str, Any]) -> list[Reference]:
    raw_refs = row.get("references")
    if not isinstance(raw_refs, list) or not raw_refs:
        raw_refs = [
            {
                "ref_id": "ref_01",
                "ref_role": "primary",
                "ref_frame": row.get("ref_frame"),
                "ref_fps": row.get("ref_fps"),
                "ref_box_xyxy": row.get("ref_box_xyxy"),
            }
        ]
    refs: list[Reference] = []
    for idx, raw in enumerate(raw_refs, start=1):
        if not isinstance(raw, dict):
            continue
        ref_box = raw.get("ref_box_xyxy") or raw.get("bbox_xyxy")
        if not isinstance(ref_box, list) or len(ref_box) != 4:
            continue
        refs.append(
            Reference(
                ref_id=str(raw.get("ref_id") or f"ref_{idx:02d}"),
                ref_role=str(raw.get("ref_role") or ("primary" if idx == 1 else f"key_{idx:02d}")),
                ref_frame=int(raw.get("ref_frame", raw.get("frame", 0)) or 0),
                ref_box_xyxy=tuple(int(round(float(x))) for x in ref_box),
                ref_fps=as_float(raw.get("ref_fps", raw.get("fps"))),
                ref_mask_path=str(raw.get("review_mask_path") or raw.get("ref_mask_path") or raw.get("mask_path") or "") or None,
                ref_mask_source=str(raw.get("review_candidate_kind") or raw.get("candidate_kind") or raw.get("ref_mask_source") or ""),
                ref_sam_iou=as_float(raw.get("review_sam_iou") or raw.get("sam_iou")),
            )
        )
    return refs


def load_latest_annotations(annotation_path: Path, metadata_cases: dict[str, dict[str, Any]]) -> dict[str, Annotation]:
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(annotation_path):
        case_id = str(row.get("case_id") or "")
        if case_id:
            latest[case_id] = row

    annotations: dict[str, Annotation] = {}
    for selected_case_id, row in latest.items():
        if str(row.get("status") or "ok") in {"skip", "discard"}:
            continue
        meta = metadata_cases.get(selected_case_id, {})
        source_case_id = meta.get("source_case_id")
        references = parse_references(row)
        if not references:
            continue
        annotation_clip = str(
            row.get("clip_file") or row.get("clip_path") or row.get("video_file") or ""
        ).strip()
        metadata_clip = str(
            meta.get("clip_file") or meta.get("clip_path") or meta.get("video_file") or ""
        ).strip()
        # Some annotation rows use a benchmark-relative `clips/...` path,
        # while the canonical benchmark manifest records the same source as
        # an absolute path.  Prefer that absolute manifest path so evaluation
        # does not depend on a copied `benchmark_root/clips` tree.
        if annotation_clip and not Path(annotation_clip).is_absolute() and Path(metadata_clip).is_absolute():
            annotation_clip = metadata_clip
        if not annotation_clip:
            annotation_clip = metadata_clip
        ann = Annotation(
            selected_case_id=selected_case_id,
            source_case_id=str(source_case_id) if source_case_id else None,
            clip_file=annotation_clip,
            prompt=str(row.get("prompt") or meta.get("prompt") or ""),
            domain_category=str(row.get("domain_category") or meta.get("domain_category") or "unknown"),
            memory_level=str(row.get("memory_level") or "object"),
            object_subtype=str(row.get("object_subtype") or "recreate"),
            background_motion=str(row.get("background_motion") or "unlabeled"),
            length_bucket=str(row.get("length_bucket") or "unknown"),
            target_label=str(row.get("target_label") or "object"),
            target_aliases=[str(x) for x in row.get("target_aliases") or []],
            validation=as_bool(row.get("validation"), as_bool(meta.get("validation"))),
            references=references,
        )
        annotations[selected_case_id] = ann
        if ann.source_case_id:
            # Keep an exact case annotation authoritative.  Short/long
            # variants may share `source_case_id`; a later variant must not
            # overwrite the exact annotation for the already-seen case.
            annotations.setdefault(ann.source_case_id, ann)
    return annotations


def discover_jobs(
    output_roots: list[Path],
    annotations_by_key: dict[str, Annotation],
    methods: set[str] | None,
    variants: set[str] | None,
    video_name: str,
) -> list[Job]:
    jobs: list[Job] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for root in output_roots:
        for metadata_path in root.rglob("metadata.json"):
            metadata = read_json(metadata_path)
            if metadata.get("status") and metadata.get("status") != "ok":
                continue
            method = canonical_method(metadata.get("method") or metadata_path.parent.name)
            if methods and method not in methods:
                continue
            prompt_variant = str(metadata.get("prompt_variant") or metadata_path.parent.parent.parent.name)
            # All canonical runs place each operating point above
            # `strict/<case>/<method>`.  Prefer the explicit metadata value,
            # because it survives copied/archived result trees; the path is a
            # backward-compatible fallback for older completed runs.
            profile = str(
                metadata.get("matched_prune_profile")
                or metadata.get("profile")
                or metadata_path.parent.parent.parent.parent.name
            )
            if variants and prompt_variant not in variants:
                continue
            output_case_id = str(metadata.get("case_id") or metadata_path.parent.parent.name)
            ann = annotations_by_key.get(output_case_id)
            if ann is None:
                continue
            raw_video_path = str(metadata.get("continuation_video") or "").strip()
            video_path = Path(raw_video_path) if raw_video_path else metadata_path.parent / video_name
            if not video_path.is_file():
                video_path = metadata_path.parent / video_name
            if not video_path.is_file():
                continue
            key = (ann.selected_case_id, prompt_variant, profile, method, str(video_path))
            if key in seen:
                continue
            seen.add(key)
            jobs.append(
                Job(
                    selected_case_id=ann.selected_case_id,
                    output_case_id=output_case_id,
                    prompt_variant=prompt_variant,
                    profile=profile,
                    method=method,
                    output_dir=metadata_path.parent,
                    video_path=video_path,
                    metadata_path=metadata_path,
                    metadata=metadata,
                )
            )
    jobs.sort(key=lambda j: (j.prompt_variant, j.selected_case_id, j.profile, j.method))
    return jobs


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return frames, float(fps)


def to_pil_bgr(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def clamp_xyxy(raw: list[float] | tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(x) for x in raw]
    x1 = max(0, min(int(round(x1)), width - 2))
    y1 = max(0, min(int(round(y1)), height - 2))
    x2 = max(x1 + 2, min(int(round(x2)), width))
    y2 = max(y1 + 2, min(int(round(y2)), height))
    return x1, y1, x2, y2


def xyxy_to_xywh(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1


def xywh_to_xyxy(box: tuple[int, int, int, int]) -> list[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def box_mask(shape_hw: tuple[int, int], box_xywh: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape_hw
    x, y, bw, bh = box_xywh
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y : y + bh, x : x + bw] = 255
    return mask


def load_reference_mask(mask_path: str | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    h, w = shape_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def masked_crop(frame_bgr: np.ndarray, mask: np.ndarray, box_xywh: tuple[int, int, int, int], size: int | None = None) -> np.ndarray:
    x, y, w, h = box_xywh
    crop = frame_bgr[y : y + h, x : x + w].copy()
    crop_mask = mask[y : y + h, x : x + w]
    if crop.size == 0 or crop_mask.size == 0:
        crop = np.full((size or 32, size or 32, 3), 127, np.uint8)
        return crop
    bg = np.full_like(crop, 127)
    crop[crop_mask == 0] = bg[crop_mask == 0]
    if size is not None:
        crop = resize_pad_square(crop, size)
    return crop


def resize_pad_square(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 127, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


class VisualScorer:
    def __init__(
        self,
        owl_model: str,
        sam_model: str,
        dino_model: str,
        device: str,
        mask_mode: str,
    ):
        self.device = torch.device(device)
        self.mask_mode = mask_mode
        self.owl_processor = OwlViTProcessor.from_pretrained(owl_model, local_files_only=Path(owl_model).exists())
        self.owl_model = OwlViTForObjectDetection.from_pretrained(owl_model, local_files_only=Path(owl_model).exists()).to(self.device).eval()
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model, local_files_only=Path(dino_model).exists())
        self.dino_model = AutoModel.from_pretrained(dino_model, local_files_only=Path(dino_model).exists()).to(self.device).eval()
        self.sam_processor = None
        self.sam_model = None
        if mask_mode == "sam":
            self.sam_processor = Sam2Processor.from_pretrained(sam_model, local_files_only=Path(sam_model).exists())
            self.sam_model = Sam2Model.from_pretrained(sam_model, local_files_only=Path(sam_model).exists()).to(self.device).eval()

    def detect(
        self,
        frame_bgr: np.ndarray,
        texts: list[str],
        threshold: float,
        top_k: int,
        min_area_frac: float,
        max_area_frac: float,
    ) -> list[Candidate]:
        height, width = frame_bgr.shape[:2]
        image = to_pil_bgr(frame_bgr)
        nested_texts = [texts]
        with torch.inference_mode():
            # OWL-ViT has a fixed text-position capacity. Rich benchmark labels
            # can exceed it, and current tokenizers do not truncate by default.
            inputs = self.owl_processor(
                text=nested_texts, images=image, return_tensors="pt",
                padding="max_length", truncation=True,
                max_length=int(self.owl_model.config.text_config.max_position_embeddings),
            )
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            outputs = self.owl_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
            result = self.owl_processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
                text_labels=nested_texts,
            )[0]
        candidates: list[Candidate] = []
        for score, raw_box, label in zip(result["scores"].detach().cpu(), result["boxes"].detach().cpu(), result["text_labels"]):
            x1, y1, x2, y2 = clamp_xyxy([float(x) for x in raw_box], width, height)
            area_frac = ((x2 - x1) * (y2 - y1)) / float(width * height)
            if area_frac < min_area_frac or area_frac > max_area_frac:
                continue
            candidates.append(Candidate(box=(x1, y1, x2 - x1, y2 - y1), class_score=float(score), label=str(label)))
        candidates.sort(key=lambda item: item.class_score, reverse=True)
        return candidates[:top_k]

    def mask_from_box(self, frame_bgr: np.ndarray, box_xywh: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
        if self.mask_mode != "sam" or self.sam_processor is None or self.sam_model is None:
            return box_mask(frame_bgr.shape[:2], box_xywh), 1.0
        image = to_pil_bgr(frame_bgr)
        with torch.inference_mode():
            inputs = self.sam_processor(image, input_boxes=[[xywh_to_xyxy(box_xywh)]], return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            outputs = self.sam_model(**inputs)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
            )[0]
            ious = outputs.iou_scores.detach().cpu()[0, 0]
        mask_idx = int(torch.argmax(ious).item())
        mask = masks[0, mask_idx].numpy().astype(np.uint8) * 255
        return mask, float(ious[mask_idx].item())

    def dino_feature(self, frame_bgr: np.ndarray, mask: np.ndarray, box_xywh: tuple[int, int, int, int]) -> torch.Tensor:
        crop = masked_crop(frame_bgr, mask, box_xywh)
        with torch.inference_mode():
            inputs = self.dino_processor(images=to_pil_bgr(crop), return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            out = self.dino_model(**inputs)
        feat = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return F.normalize(feat.float(), dim=-1)


def color_hist(frame_bgr: np.ndarray, mask: np.ndarray, box_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box_xywh
    crop = frame_bgr[y : y + h, x : x + w]
    crop_mask = mask[y : y + h, x : x + w]
    if crop.size == 0 or crop_mask.size == 0 or int((crop_mask > 0).sum()) < 8:
        return np.zeros((16 * 8 * 8,), dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], crop_mask, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).reshape(-1)
    hist_sum = float(hist.sum())
    if hist_sum > 0:
        hist /= hist_sum
    return hist


def color_similarity(ref_hist: np.ndarray, cand_hist: np.ndarray) -> float:
    if ref_hist.sum() <= 0 or cand_hist.sum() <= 0:
        return 0.0
    # Bhattacharyya coefficient in [0, 1], where 1 means identical histograms.
    return float(np.sqrt(ref_hist * cand_hist).sum())


def pixel_similarity(
    ref_frame: np.ndarray,
    ref_mask: np.ndarray,
    ref_box: tuple[int, int, int, int],
    cand_frame: np.ndarray,
    cand_mask: np.ndarray,
    cand_box: tuple[int, int, int, int],
    size: int,
) -> float:
    ref = masked_crop(ref_frame, ref_mask, ref_box, size=size).astype(np.float32)
    cand = masked_crop(cand_frame, cand_mask, cand_box, size=size).astype(np.float32)
    mae = float(np.mean(np.abs(ref - cand))) / 255.0
    return max(0.0, min(1.0, 1.0 - mae))


def final_score(match_mode: str, class_score: float, dino_score: float, color_score: float) -> float:
    if match_mode == "semantic_color":
        return 0.50 * class_score + 0.40 * color_score + 0.10 * dino_score
    return 0.20 * class_score + 0.45 * dino_score + 0.35 * color_score


def detail_score(match_mode: str, dino_score: float, color_score: float, pixel_score: float, use_pixel: bool) -> float:
    if match_mode == "semantic_color":
        return color_score
    if use_pixel:
        return 0.45 * dino_score + 0.35 * color_score + 0.20 * pixel_score
    return 0.55 * dino_score + 0.45 * color_score


def classify_memory(class_score: float, detail: float, class_threshold: float, detail_threshold: float) -> tuple[str, float]:
    if class_score < class_threshold:
        return "class_wrong_or_missing", 0.0
    if detail < detail_threshold:
        return "class_ok_detail_wrong", -1.0
    return "class_ok_detail_ok", 1.0


def score_frame(
    scorer: VisualScorer,
    ann: Annotation,
    frame: np.ndarray,
    ref_frame: np.ndarray,
    ref_mask: np.ndarray,
    ref_box: tuple[int, int, int, int],
    ref_feat: torch.Tensor,
    ref_hist: np.ndarray,
    owl_threshold: float,
    top_k: int,
    min_area_frac: float,
    max_area_frac: float,
    pixel_size: int,
    use_pixel_in_detail: bool,
    class_threshold: float,
    detail_threshold: float,
) -> ScoredCandidate:
    candidates = scorer.detect(frame, ann.texts, owl_threshold, top_k, min_area_frac, max_area_frac)
    best: ScoredCandidate | None = None
    for cand in candidates:
        mask, sam_iou = scorer.mask_from_box(frame, cand.box)
        feat = scorer.dino_feature(frame, mask, cand.box)
        dino = float((ref_feat * feat).sum().item())
        hist = color_hist(frame, mask, cand.box)
        color = color_similarity(ref_hist, hist)
        pixel = pixel_similarity(ref_frame, ref_mask, ref_box, frame, mask, cand.box, pixel_size)
        total = final_score(ann.match_mode, cand.class_score, dino, color)
        detail = detail_score(ann.match_mode, dino, color, pixel, use_pixel_in_detail)
        memory_label, memory_score = classify_memory(cand.class_score, detail, class_threshold, detail_threshold)
        scored = ScoredCandidate(cand, mask, sam_iou, cand.class_score, dino, color, pixel, total, detail, memory_label, memory_score)
        if best is None or scored.dino_score > best.dino_score:
            best = scored
    if best is not None:
        return best
    return ScoredCandidate(None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "class_wrong_or_missing", 0.0)


def score_full_frame(
    scorer: VisualScorer,
    ann: Annotation,
    frame: np.ndarray,
    ref_frame: np.ndarray,
    ref_mask: np.ndarray,
    ref_box: tuple[int, int, int, int],
    ref_feat: torch.Tensor,
    ref_hist: np.ndarray,
    pixel_size: int,
    use_pixel_in_detail: bool,
    class_threshold: float,
    detail_threshold: float,
) -> ScoredCandidate:
    height, width = frame.shape[:2]
    cand_box = (0, 0, width, height)
    mask = np.full((height, width), 255, dtype=np.uint8)
    feat = scorer.dino_feature(frame, mask, cand_box)
    dino = float((ref_feat * feat).sum().item())
    hist = color_hist(frame, mask, cand_box)
    color = color_similarity(ref_hist, hist)
    pixel = pixel_similarity(ref_frame, ref_mask, ref_box, frame, mask, cand_box, pixel_size)
    class_score = 1.0
    total = final_score(ann.match_mode, class_score, dino, color)
    detail = detail_score(ann.match_mode, dino, color, pixel, use_pixel_in_detail)
    memory_label, memory_score = classify_memory(class_score, detail, class_threshold, detail_threshold)
    return ScoredCandidate(
        Candidate(cand_box, class_score, "full_frame"),
        mask,
        1.0,
        class_score,
        dino,
        color,
        pixel,
        total,
        detail,
        memory_label,
        memory_score,
    )


def scale_xyxy(
    box_xyxy: tuple[int, int, int, int],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    return clamp_xyxy(
        (
            x1 * dst_width / max(1, src_width),
            y1 * dst_height / max(1, src_height),
            x2 * dst_width / max(1, src_width),
            y2 * dst_height / max(1, src_height),
        ),
        dst_width,
        dst_height,
    )


def score_direct_bbox(
    scorer: VisualScorer,
    ann: Annotation,
    frame: np.ndarray,
    ref_frame: np.ndarray,
    ref_mask: np.ndarray,
    ref_box: tuple[int, int, int, int],
    ref_feat: torch.Tensor,
    ref_hist: np.ndarray,
    pixel_size: int,
) -> ScoredCandidate:
    src_h, src_w = ref_frame.shape[:2]
    height, width = frame.shape[:2]
    ref_xyxy = (ref_box[0], ref_box[1], ref_box[0] + ref_box[2], ref_box[1] + ref_box[3])
    cand_box = xyxy_to_xywh(scale_xyxy(ref_xyxy, src_w, src_h, width, height))
    mask = box_mask((height, width), cand_box)
    feat = scorer.dino_feature(frame, mask, cand_box)
    dino = float((ref_feat * feat).sum().item())
    hist = color_hist(frame, mask, cand_box)
    color = color_similarity(ref_hist, hist)
    pixel = pixel_similarity(ref_frame, ref_mask, ref_box, frame, mask, cand_box, pixel_size)
    class_score = 1.0
    total = final_score(ann.match_mode, class_score, dino, color)
    detail = detail_score(ann.match_mode, dino, color, pixel, False)
    return ScoredCandidate(
        Candidate(cand_box, class_score, "direct_bbox"),
        mask,
        1.0,
        class_score,
        dino,
        color,
        pixel,
        total,
        detail,
        "",
        0.0,
    )


def draw_score_overlay(
    frame: np.ndarray,
    scored: ScoredCandidate,
    ann: Annotation,
    job: Job,
    frame_index: int,
) -> np.ndarray:
    vis = frame.copy()
    if scored.mask is not None and scored.candidate is not None:
        mask = scored.mask > 0
        color = np.asarray((0, 185, 0), dtype=np.float32)
        vis[mask] = (0.50 * vis[mask] + 0.50 * color).astype(np.uint8)
        x, y, w, h = scored.candidate.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 2)
    return vis


def draw_multi_score_overlay(frame: np.ndarray, scored_items: list[tuple[str, ScoredCandidate]]) -> np.ndarray:
    vis = frame.copy()
    palette = [
        (0, 220, 0),
        (0, 190, 240),
        (230, 170, 0),
        (220, 80, 220),
        (255, 90, 40),
        (80, 140, 255),
    ]
    for idx, (_ref_id, scored) in enumerate(scored_items):
        if scored.mask is None or scored.candidate is None:
            continue
        rgb = palette[idx % len(palette)]
        mask = scored.mask > 0
        color = np.asarray(rgb, dtype=np.float32)
        vis[mask] = (0.58 * vis[mask] + 0.42 * color).astype(np.uint8)
        x, y, w, h = scored.candidate.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), rgb, 2)
    return vis


def write_video(frames: list[np.ndarray], output_path: Path, fps: float) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def resize_pad(frame: np.ndarray, tile_w: int, image_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(tile_w / max(1, w), image_h / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((image_h, tile_w, 3), 18, dtype=np.uint8)
    top = (image_h - new_h) // 2
    left = (tile_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def read_frames_for_grid(path: Path, tile_w: int, image_h: int) -> tuple[list[np.ndarray], float]:
    if not path.exists():
        return [], 16.0
    frames, fps = read_video(path)
    return [resize_pad(frame, tile_w, image_h) for frame in frames], fps


def label_tile(frame: np.ndarray, label: str, tile_h: int) -> np.ndarray:
    tile_w = frame.shape[1]
    canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    header_h = tile_h - frame.shape[0]
    canvas[header_h:, :] = frame
    cv2.putText(canvas, label[:86], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def placeholder(label: str, tile_w: int, image_h: int) -> np.ndarray:
    canvas = np.full((image_h, tile_w, 3), 34, dtype=np.uint8)
    cv2.putText(canvas, "missing", (18, image_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, label[:34], (18, image_h // 2 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas


def make_grid_video(items: list[tuple[str, Path]], output_path: Path, tile_w: int, image_h: int, cols: int) -> bool:
    loaded = []
    max_len = 0
    fps = 16.0
    for label, path in items:
        frames, item_fps = read_frames_for_grid(path, tile_w, image_h)
        if frames:
            fps = item_fps or fps
        max_len = max(max_len, len(frames))
        loaded.append((label, frames))
    if max_len == 0:
        return False
    tile_h = image_h + 34
    rows = int(math.ceil(len(items) / cols))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (cols * tile_w, rows * tile_h),
    )
    for frame_idx in range(max_len):
        tiles = []
        for label, frames in loaded:
            if frames:
                frame = frames[min(frame_idx, len(frames) - 1)]
            else:
                frame = placeholder(label, tile_w, image_h)
            tiles.append(label_tile(frame, label, tile_h))
        while len(tiles) < rows * cols:
            tiles.append(np.full((tile_h, tile_w, 3), 245, dtype=np.uint8))
        row_images = []
        for row_idx in range(rows):
            start = row_idx * cols
            row_images.append(np.concatenate(tiles[start : start + cols], axis=1))
        writer.write(np.concatenate(row_images, axis=0))
    writer.release()
    return True


def ordered_reference_match(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_ref: dict[str, list[dict[str, Any]]] = {}
    ref_roles: dict[str, str] = {}
    for row in rows:
        ref_id = str(row.get("ref_id") or "ref_01")
        by_ref.setdefault(ref_id, []).append(row)
        ref_roles.setdefault(ref_id, str(row.get("ref_role") or ref_id))
    ref_ids = list(by_ref)
    if not ref_ids:
        return {
            "trajectory_order_valid": False,
            "trajectory_ordered_dino_score": None,
            "trajectory_ordered_min_dino": None,
            "trajectory_ordered_frames": "",
            "trajectory_ordered_refs": "",
            "trajectory_order_note": "no references",
        }
    if len(ref_ids) == 1:
        row = max(by_ref[ref_ids[0]], key=lambda r: float(r["dino_score"]))
        return {
            "trajectory_order_valid": True,
            "trajectory_ordered_dino_score": float(row["dino_score"]),
            "trajectory_ordered_min_dino": float(row["dino_score"]),
            "trajectory_ordered_class_score": float(row["class_score"]),
            "trajectory_ordered_color_score": float(row["color_score"]),
            "trajectory_ordered_pixel_score": float(row["pixel_score"]),
            "trajectory_ordered_frames": str(int(row["frame_index"])),
            "trajectory_ordered_refs": ref_ids[0],
            "trajectory_order_note": "single reference",
        }

    candidates = [
        sorted(by_ref[ref_id], key=lambda r: (int(r["frame_index"]), -float(r["dino_score"])))
        for ref_id in ref_ids
    ]
    dp: list[dict[int, tuple[float, int | None]]] = []
    first: dict[int, tuple[float, int | None]] = {}
    for j, row in enumerate(candidates[0]):
        first[j] = (float(row["dino_score"]), None)
    dp.append(first)

    for i in range(1, len(candidates)):
        cur_dp: dict[int, tuple[float, int | None]] = {}
        for j, row in enumerate(candidates[i]):
            frame = int(row["frame_index"])
            best_score = None
            best_prev = None
            for k, (prev_score, _) in dp[i - 1].items():
                prev_frame = int(candidates[i - 1][k]["frame_index"])
                if prev_frame >= frame:
                    continue
                score = prev_score + float(row["dino_score"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev = k
            if best_score is not None:
                cur_dp[j] = (best_score, best_prev)
        dp.append(cur_dp)

    if not dp[-1]:
        return {
            "trajectory_order_valid": False,
            "trajectory_ordered_dino_score": 0.0,
            "trajectory_ordered_min_dino": 0.0,
            "trajectory_ordered_class_score": 0.0,
            "trajectory_ordered_color_score": 0.0,
            "trajectory_ordered_pixel_score": 0.0,
            "trajectory_ordered_frames": "",
            "trajectory_ordered_refs": ",".join(ref_ids),
            "trajectory_order_note": "no strictly increasing frame path",
        }

    last_idx = max(dp[-1], key=lambda j: dp[-1][j][0])
    selected_indices = [last_idx]
    for i in range(len(candidates) - 1, 0, -1):
        prev_idx = dp[i][selected_indices[-1]][1]
        if prev_idx is None:
            break
        selected_indices.append(prev_idx)
    selected_indices.reverse()
    selected_rows = [candidates[i][idx] for i, idx in enumerate(selected_indices)]
    dino = [float(row["dino_score"]) for row in selected_rows]
    cls = [float(row["class_score"]) for row in selected_rows]
    color = [float(row["color_score"]) for row in selected_rows]
    pixel = [float(row["pixel_score"]) for row in selected_rows]
    return {
        "trajectory_order_valid": True,
        "trajectory_ordered_dino_score": float(np.mean(dino)),
        "trajectory_ordered_min_dino": float(np.min(dino)),
        "trajectory_ordered_class_score": float(np.mean(cls)),
        "trajectory_ordered_color_score": float(np.mean(color)),
        "trajectory_ordered_pixel_score": float(np.mean(pixel)),
        "trajectory_ordered_frames": ",".join(str(int(row["frame_index"])) for row in selected_rows),
        "trajectory_ordered_refs": ",".join(ref_ids),
        "trajectory_order_note": "strictly increasing frame_index",
    }


def summarize_scores(rows: list[dict[str, Any]], coverage_threshold: float) -> dict[str, Any]:
    if not rows:
        return {
            "num_eval_frames": 0,
            "num_references": 0,
            "dino_score": None,
            "dino_ref_best_mean": None,
            "dino_ref_best_min": None,
            "class_score": None,
            "color_score": None,
            "pixel_score": None,
            "coverage": 0.0,
            "reference_coverage": 0.0,
            "trajectory_order_valid": None,
        }
    for row in rows:
        row["coverage_hit"] = float(row["dino_score"]) >= coverage_threshold
    dino_scores = np.asarray([float(row["dino_score"]) for row in rows], dtype=np.float32)
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ref.setdefault(str(row.get("ref_id") or "ref_01"), []).append(row)
    ref_best_rows = [max(part, key=lambda r: float(r["dino_score"])) for part in by_ref.values()]
    ref_best_dino = [float(row["dino_score"]) for row in ref_best_rows]
    ref_best_class = [float(row["class_score"]) for row in ref_best_rows]
    ref_best_color = [float(row["color_score"]) for row in ref_best_rows]
    ref_best_pixel = [float(row["pixel_score"]) for row in ref_best_rows]
    memory_level = str(rows[0].get("memory_level") or "")
    ordered = ordered_reference_match(rows) if memory_level == "trajectory" else {
        "trajectory_order_valid": None,
        "trajectory_ordered_dino_score": None,
        "trajectory_ordered_min_dino": None,
        "trajectory_ordered_class_score": None,
        "trajectory_ordered_color_score": None,
        "trajectory_ordered_pixel_score": None,
        "trajectory_ordered_frames": "",
        "trajectory_ordered_refs": "",
        "trajectory_order_note": "not trajectory",
    }
    main_dino_score = (
        ordered["trajectory_ordered_dino_score"]
        if memory_level == "trajectory"
        else float(np.mean(ref_best_dino))
    )
    main_class_score = (
        ordered["trajectory_ordered_class_score"]
        if memory_level == "trajectory"
        else float(np.mean(ref_best_class))
    )
    main_color_score = (
        ordered["trajectory_ordered_color_score"]
        if memory_level == "trajectory"
        else float(np.mean(ref_best_color))
    )
    main_pixel_score = (
        ordered["trajectory_ordered_pixel_score"]
        if memory_level == "trajectory"
        else float(np.mean(ref_best_pixel))
    )
    return {
        "num_eval_frames": int(len(rows)),
        "num_references": int(len(by_ref)),
        "dino_score": main_dino_score,
        "dino_ref_best_mean": float(np.mean(ref_best_dino)),
        "dino_ref_best_min": float(np.min(ref_best_dino)),
        "class_ref_best_mean": main_class_score,
        "color_ref_best_mean": main_color_score,
        "pixel_ref_best_mean": main_pixel_score,
        **ordered,
        "class_score": main_class_score,
        "color_score": main_color_score,
        "pixel_score": main_pixel_score,
        "coverage": float(np.mean(dino_scores >= coverage_threshold)),
        "reference_coverage": float(np.mean(np.asarray(ref_best_dino, dtype=np.float32) >= coverage_threshold)),
        "class_mean": float(np.mean([row["class_score"] for row in rows])),
        "dino_mean": float(np.mean([row["dino_score"] for row in rows])),
        "color_mean": float(np.mean([row["color_score"] for row in rows])),
        "pixel_mean": float(np.mean([row["pixel_score"] for row in rows])),
        "class_best": float(np.max([row["class_score"] for row in rows])),
        "dino_best": float(np.max([row["dino_score"] for row in rows])),
        "color_best": float(np.max([row["color_score"] for row in rows])),
        "pixel_best": float(np.max([row["pixel_score"] for row in rows])),
    }


def load_efficiency(job: Job) -> dict[str, Any]:
    meta = job.metadata
    latent_shape = meta.get("latent_shape") if isinstance(meta.get("latent_shape"), list) else []
    total_latent_frames = (
        int(meta["used_latent_frames"])
        if meta.get("used_latent_frames") is not None
        else int(latent_shape[1])
        if len(latent_shape) > 1 and latent_shape[1]
        else None
    )
    is_lingbot = str(meta.get("model") or "").startswith("lingbot-world")
    out: dict[str, Any] = {
        "elapsed_s": meta.get("elapsed_s"),
        "generation_elapsed_s": meta.get("generation_elapsed_s"),
        "generation_timer_scope": None,
        "continuation_frames": meta.get("continuation_frames", meta.get("continuation_pixel_frames")),
        "context_decoded_frames": meta.get(
            "context_decoded_frames",
            meta.get("context_pixel_frames_in_combined_video"),
        ),
        "used_pixel_frames": (
            (meta.get("context") or {}).get("used_pixel_frames")
            if isinstance(meta.get("context"), dict)
            else meta.get("input_pixel_frames")
        ),
        "total_latent_frames": total_latent_frames,
        "kv_local_frames": None,
        "kv_effective_frames_with_memory": None,
        "kv_sink_frames": None,
        "kv_memory_frames": None,
        "kv_local_budget_frames": None,
        "kv_kept_ratio_mean": None,
        "kv_pruned_ratio_mean": None,
        "kv_compression_ratio": None,
        "kv_effective_with_memory_ratio": None,
        "kv_effective_with_memory_compression_ratio": None,
    }
    ar_metrics_path = job.output_dir / "ar_generation_metrics.json"
    ar_metrics = read_json(ar_metrics_path) if ar_metrics_path.is_file() else {}
    if ar_metrics:
        ar_elapsed = float(ar_metrics.get("elapsed_seconds", 0.0))
        ar_frames = int(ar_metrics.get("generated_pixel_frames", 0))
        if ar_elapsed <= 0 or ar_frames <= 0:
            raise ValueError(f"invalid AR generation metrics: {ar_metrics_path}")
        out["generation_elapsed_s"] = ar_elapsed
        out["continuation_frames"] = ar_frames
        out["generation_timer_scope"] = ar_metrics.get("timer_scope")
    if job.method == "full_kv":
        out["kv_local_frames"] = total_latent_frames
        out["kv_effective_frames_with_memory"] = total_latent_frames
        out["kv_kept_ratio_mean"] = 1.0
        out["kv_pruned_ratio_mean"] = 0.0
        out["kv_compression_ratio"] = 1.0
        out["kv_effective_with_memory_ratio"] = 1.0
        out["kv_effective_with_memory_compression_ratio"] = 1.0
    elif (
        is_lingbot
        and job.method in {"native_longlive_streaming", "longlive_rag"}
        and total_latent_frames
    ):
        if job.method == "native_longlive_streaming":
            local_frames = min(
                int(meta.get("model_local_attn_size") or 0),
                total_latent_frames,
            )
            sink_frames = min(
                int(meta.get("sink_size") or 0),
                local_frames,
            )
            memory_frames = 0
        else:
            policy = (
                meta.get("longliverag_policy")
                if isinstance(meta.get("longliverag_policy"), dict)
                else {}
            )
            local_frames = min(
                int(policy.get("max_live_frames") or 0),
                total_latent_frames,
            )
            sink_frames = min(
                int(policy.get("sink_frames") or 0),
                local_frames,
            )
            memory_frames = min(
                int(policy.get("memory_frames") or 0),
                max(0, local_frames - sink_frames),
            )
        kept_ratio = float(local_frames) / float(total_latent_frames)
        local_budget_frames = max(
            0, local_frames - sink_frames - memory_frames
        )
        out["kv_local_frames"] = local_frames
        out["kv_sink_frames"] = sink_frames
        out["kv_memory_frames"] = memory_frames
        out["kv_local_budget_frames"] = local_budget_frames
        out["kv_kept_ratio_mean"] = kept_ratio
        out["kv_pruned_ratio_mean"] = 1.0 - kept_ratio
        out["kv_compression_ratio"] = (
            1.0 / kept_ratio if kept_ratio > 0 else None
        )
        out["kv_effective_frames_with_memory"] = local_frames
        out["kv_effective_with_memory_ratio"] = kept_ratio
        out["kv_effective_with_memory_compression_ratio"] = (
            1.0 / kept_ratio if kept_ratio > 0 else None
        )
    elif (
        not is_lingbot
        and job.method in FIXED_LOCAL_ATTN_SIZE
        and total_latent_frames
    ):
        local_frames = min(FIXED_LOCAL_ATTN_SIZE[job.method], total_latent_frames)
        kept_ratio = float(local_frames) / float(total_latent_frames)
        sink_frames = min(FIXED_SINK_SIZE.get(job.method, 0), local_frames)
        memory_frames = min(RAG_MEMORY_SIZE.get(job.method, 0), max(0, local_frames - sink_frames))
        local_budget_frames = max(0, local_frames - sink_frames - memory_frames)
        out["kv_local_frames"] = local_frames
        out["kv_sink_frames"] = sink_frames
        out["kv_memory_frames"] = memory_frames
        out["kv_local_budget_frames"] = local_budget_frames
        out["kv_kept_ratio_mean"] = kept_ratio
        out["kv_pruned_ratio_mean"] = 1.0 - kept_ratio
        out["kv_compression_ratio"] = 1.0 / kept_ratio if kept_ratio > 0 else None
        # LongLive-RAG splits the same max_attention_size budget into
        # sink + retrieved memory + recent local window. Its memory slots are
        # not extra KV cache beyond local_attn_size.
        effective_frames = local_frames
        effective_ratio = float(effective_frames) / float(total_latent_frames)
        out["kv_effective_frames_with_memory"] = effective_frames
        out["kv_effective_with_memory_ratio"] = effective_ratio
        out["kv_effective_with_memory_compression_ratio"] = 1.0 / effective_ratio if effective_ratio > 0 else None
    stats_path = job.output_dir / "kv_prune_stats.jsonl"
    stats = load_jsonl(stats_path)
    if stats:
        kept_tokens = [float(row["kept"]) for row in stats if row.get("kept") is not None]
        total_tokens = [float(row["total"]) for row in stats if row.get("total") is not None]
        if kept_tokens and total_tokens and sum(total_tokens) > 0:
            mean_kept = float(sum(kept_tokens) / sum(total_tokens))
        else:
            kept = [float(row["kept_ratio"]) for row in stats if row.get("kept_ratio") is not None]
            mean_kept = float(np.mean(kept)) if kept else None
        if mean_kept is not None:
            out["kv_kept_ratio_mean"] = mean_kept
            out["kv_pruned_ratio_mean"] = 1.0 - mean_kept
            out["kv_compression_ratio"] = float(1.0 / mean_kept) if mean_kept > 0 else None
            out["kv_effective_with_memory_ratio"] = mean_kept
            out["kv_effective_with_memory_compression_ratio"] = float(1.0 / mean_kept) if mean_kept > 0 else None
    recorded_fps = (
        ar_metrics.get("fps_generated_pixels")
        if ar_metrics
        else meta.get("generation_fps")
    )
    elapsed = out.get("generation_elapsed_s") or out.get("elapsed_s")
    cont = out.get("continuation_frames")
    if recorded_fps is not None:
        out["generation_fps"] = float(recorded_fps)
    elif elapsed and cont:
        out["generation_fps"] = float(cont) / float(elapsed)
    else:
        out["generation_fps"] = None
    return out


def load_prune_and_mask_metrics(
    method: str, output_dir: Path
) -> dict[str, Any]:
    """Load only the strict q0 PR contract.

    Historical ``mask_ratio_aux.json`` files are deliberately ignored.  They
    mixed incompatible method scopes and are superseded by the standalone
    candidate-decision mask ratio (CDMR) tool.
    """
    prune = (
        strict_method_prune(method, output_dir)
        if strict_method_prune is not None
        else {
            "prune_ratio": None,
            "keep_ratio": None,
            "prune_source": "strict_prune_import_failed",
            "prune_scope": "unknown",
            "prune_note": (
                "compute_context_memory_prune_ratio_strict.py was not importable"
            ),
        }
    )
    return {
        "prune_ratio": prune.get("prune_ratio"),
        "prune_keep_ratio": prune.get("keep_ratio"),
        "prune_source": prune.get("prune_source"),
        "prune_scope": prune.get("prune_scope"),
        "prune_note": prune.get("prune_note"),
        "active_q0_kv_units": prune.get(
            "active_q0_kv_units", prune.get("kept_old_tokens")
        ),
        "full_q0_kv_units": prune.get(
            "full_q0_kv_units", prune.get("full_old_tokens")
        ),
        "pruned_q0_kv_units": prune.get(
            "pruned_q0_kv_units", prune.get("masked_context_tokens")
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def is_full_frame_reference(ref: Reference, width: int, height: int, area_threshold: float = 0.80) -> bool:
    role = ref.ref_role.lower()
    if any(key in role for key in ("full", "fullscreen", "scene_full")):
        return True
    x1, y1, x2, y2 = ref.ref_box_xyxy
    area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, width * height)
    return area / frame_area >= area_threshold


def evaluate_job(
    scorer: VisualScorer,
    ann: Annotation,
    job: Job,
    benchmark_root: Path,
    eval_root: Path,
    frame_stride: int,
    owl_threshold: float,
    top_k: int,
    min_area_frac: float,
    max_area_frac: float,
    pixel_size: int,
    use_pixel_in_detail: bool,
    class_threshold: float,
    detail_threshold: float,
    coverage_threshold: float,
    full_frame_area_threshold: float,
    scoring_path: str,
    max_frames: int | None,
    save_overlay: bool,
) -> tuple[dict[str, Any], Path | None]:
    source_video = benchmark_root / ann.clip_file
    source_frames, _ = read_video(source_video)
    frames, fps = read_video(job.video_path)
    eval_indices = list(range(0, len(frames), max(1, frame_stride)))
    if max_frames is not None:
        eval_indices = eval_indices[:max_frames]

    frame_rows: list[dict[str, Any]] = []
    overlay_by_frame: dict[int, list[tuple[str, ScoredCandidate]]] = {idx: [] for idx in eval_indices}
    reference_summaries: list[dict[str, Any]] = []
    reference_overlays: list[tuple[str, np.ndarray, np.ndarray, tuple[int, int, int, int]]] = []
    for ref in ann.references:
        ref_frame_index = max(0, min(ref.ref_frame, len(source_frames) - 1))
        ref_frame = source_frames[ref_frame_index]
        height, width = ref_frame.shape[:2]
        ref_box_xyxy = clamp_xyxy(ref.ref_box_xyxy, width, height)
        ref_box = xyxy_to_xywh(ref_box_xyxy)
        is_full_ref = is_full_frame_reference(ref, width, height, full_frame_area_threshold)
        # Every annotation bbox seeds SAM/SAM2, including boxes covering more
        # than 80% of the frame. Direct full-frame DINO is now an explicit
        # ablation only and is never selected by an area heuristic.
        use_direct_dino = scoring_path == "direct_dino"
        selected_ref_mask = load_reference_mask(ref.ref_mask_path, (height, width))
        if use_direct_dino:
            ref_mask = box_mask((height, width), ref_box)
            ref_sam_iou = 1.0
            ref_mask_source = "direct_bbox"
        elif selected_ref_mask is not None:
            ref_mask = selected_ref_mask
            ref_sam_iou = ref.ref_sam_iou if ref.ref_sam_iou is not None else 1.0
            ref_mask_source = ref.ref_mask_source or "review_mask"
        else:
            ref_mask, ref_sam_iou = scorer.mask_from_box(ref_frame, ref_box)
            ref_mask_source = "sam_auto"
        ref_feat = scorer.dino_feature(ref_frame, ref_mask, ref_box)
        ref_hist = color_hist(ref_frame, ref_mask, ref_box)
        reference_overlays.append((ref.ref_id, ref_frame, ref_mask, ref_box))
        reference_summaries.append(
            {
                "ref_id": ref.ref_id,
                "ref_role": ref.ref_role,
                "ref_frame": ref_frame_index,
                "ref_box_xyxy": list(ref_box_xyxy),
                "ref_sam_iou": ref_sam_iou,
                "ref_mask_source": ref_mask_source,
                "ref_mask_path": ref.ref_mask_path,
                "is_full_frame_reference": is_full_ref,
                "scoring_path": scoring_path,
                "used_direct_dino": use_direct_dino,
                "full_frame_area_threshold": full_frame_area_threshold,
            }
        )

        for frame_index in eval_indices:
            frame = frames[frame_index]
            if use_direct_dino:
                scored = score_direct_bbox(
                    scorer=scorer,
                    ann=ann,
                    frame=frame,
                    ref_frame=ref_frame,
                    ref_mask=ref_mask,
                    ref_box=ref_box,
                    ref_feat=ref_feat,
                    ref_hist=ref_hist,
                    pixel_size=pixel_size,
                )
            else:
                scored = score_frame(
                    scorer=scorer,
                    ann=ann,
                    frame=frame,
                    ref_frame=ref_frame,
                    ref_mask=ref_mask,
                    ref_box=ref_box,
                    ref_feat=ref_feat,
                    ref_hist=ref_hist,
                    owl_threshold=owl_threshold,
                    top_k=top_k,
                    min_area_frac=min_area_frac,
                    max_area_frac=max_area_frac,
                    pixel_size=pixel_size,
                    use_pixel_in_detail=use_pixel_in_detail,
                    class_threshold=class_threshold,
                    detail_threshold=detail_threshold,
                )
            row = {
                "selected_case_id": ann.selected_case_id,
                "output_case_id": job.output_case_id,
                "prompt_variant": job.prompt_variant,
                "profile": job.profile,
                "method": job.method,
                "ref_id": ref.ref_id,
                "ref_role": ref.ref_role,
                "ref_frame": ref_frame_index,
                "frame_index": frame_index,
                "target_label": ann.target_label,
                "domain_category": ann.domain_category,
                "memory_level": ann.memory_level,
                "object_subtype": ann.object_subtype,
                "validation": ann.validation,
                "candidate_label": scored.candidate.label if scored.candidate else "",
                "class_score": scored.class_score,
                "dino_score": scored.dino_score,
                "color_score": scored.color_score,
                "pixel_score": scored.pixel_score,
                "coverage_hit": "",
                "sam_iou": scored.sam_iou,
                "x": scored.candidate.box[0] if scored.candidate else 0,
                "y": scored.candidate.box[1] if scored.candidate else 0,
                "w": scored.candidate.box[2] if scored.candidate else 0,
                "h": scored.candidate.box[3] if scored.candidate else 0,
                "num_texts": len(ann.texts),
            }
            frame_rows.append(row)
            if save_overlay:
                overlay_by_frame[frame_index].append((ref.ref_id, scored))

    summary_scores = summarize_scores(frame_rows, coverage_threshold)
    efficiency = load_efficiency(job)
    runtime_metrics = load_prune_and_mask_metrics(
        job.method, job.output_dir
    )
    q0_prune: dict[str, Any] | None = None
    snapshot_path = job.output_dir / "q0_cache_snapshot.json"
    if q0_case_row is not None and snapshot_path.exists():
        try:
            q0_prune = q0_case_row(snapshot_path, default_heads=12)
        except Exception as exc:
            q0_prune = {"accounting_error": repr(exc)}
    # Keep the old score-mask number visibly separate from the benchmark PR.
    # It is useful only when auditing historical Ours outputs and must never
    # be copied into the common metric fields.
    legacy_prune = {
        "prune_ratio": q0_prune.get("legacy_zero_mask_score_prune_ratio")
        if q0_prune else None,
        "keep_ratio": None,
        "prune_source": "legacy_zero_mask_score_statistic" if q0_prune and q0_prune.get("legacy_zero_mask_score_prune_ratio") is not None else "none",
        "prune_scope": "legacy_score_mask_only",
        "prune_note": "diagnostic only; not physical cache reduction",
    }
    summary = {
        "selected_case_id": ann.selected_case_id,
        "output_case_id": job.output_case_id,
        "prompt_variant": job.prompt_variant,
        "profile": job.profile,
        "method": job.method,
        "target_label": ann.target_label,
        "target_aliases": ",".join(ann.target_aliases),
        "domain_category": ann.domain_category,
        "memory_level": ann.memory_level,
        "object_subtype": ann.object_subtype,
        "validation": ann.validation,
        "background_motion": ann.background_motion,
        "length_bucket": ann.length_bucket,
        "video_path": str(job.video_path),
        "references": reference_summaries,
        "ref_frame": reference_summaries[0]["ref_frame"] if reference_summaries else None,
        "ref_box_xyxy": reference_summaries[0]["ref_box_xyxy"] if reference_summaries else None,
        "ref_sam_iou": reference_summaries[0]["ref_sam_iou"] if reference_summaries else None,
        "coverage_threshold": coverage_threshold,
        "scoring_path": scoring_path,
        **summary_scores,
        # PR counts the KVs actually read by the first generated q0. CDMR is
        # computed separately from method-native candidate-decision logs.
        "prune_ratio": runtime_metrics.get("prune_ratio"),
        "prune_keep_ratio": runtime_metrics.get("prune_keep_ratio"),
        "prune_source": runtime_metrics.get("prune_source"),
        "prune_scope": runtime_metrics.get("prune_scope"),
        "active_q0_kv_units": runtime_metrics.get("active_q0_kv_units"),
        "full_q0_kv_units": runtime_metrics.get("full_q0_kv_units"),
        "pruned_q0_kv_units": runtime_metrics.get("pruned_q0_kv_units"),
        "prune_note": runtime_metrics.get("prune_note"),
        "legacy_prune_ratio": legacy_prune.get("prune_ratio"),
        "legacy_prune_keep_ratio": legacy_prune.get("keep_ratio"),
        "legacy_prune_source": legacy_prune.get("prune_source"),
        "legacy_prune_scope": legacy_prune.get("prune_scope"),
        "legacy_prune_note": legacy_prune.get("prune_note"),
        **efficiency,
    }

    # A method can deliberately appear at multiple operating points (for
    # example ours_new_prune20 and ours_new_prune80).  Profile is therefore a
    # first-class part of the evaluation identity; writing only by `method`
    # used to overwrite/incorrectly reuse one operating point's summary.
    job_eval_dir = eval_root / "per_video" / job.prompt_variant / ann.selected_case_id / job.profile / job.method
    write_csv(job_eval_dir / "frame_scores.csv", frame_rows)
    write_jsonl(job_eval_dir / "frame_scores.jsonl", frame_rows)
    (job_eval_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for idx, (ref_id, ref_frame, ref_mask, ref_box) in enumerate(reference_overlays):
        ref_overlay = draw_reference_overlay(ref_frame, ref_mask, ref_box, ann)
        name = "reference_overlay.png" if idx == 0 else f"reference_overlay_{ref_id}.png"
        cv2.imwrite(str(job_eval_dir / name), ref_overlay)

    overlay_path: Path | None = None
    if save_overlay and overlay_by_frame:
        overlay_frames = [draw_multi_score_overlay(frames[idx], overlay_by_frame[idx]) for idx in eval_indices]
        overlay_path = eval_root / "overlays" / job.prompt_variant / ann.selected_case_id / f"{job.profile}__{job.method}_scores.mp4"
        write_video(overlay_frames, overlay_path, fps / max(1, frame_stride))
    return summary, overlay_path


def draw_reference_overlay(frame: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], ann: Annotation) -> np.ndarray:
    vis = frame.copy()
    m = mask > 0
    vis[m] = (0.50 * vis[m] + 0.50 * np.asarray((0, 180, 220), dtype=np.float32)).astype(np.uint8)
    x, y, w, h = box
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 220), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 48), (255, 255, 255), -1)
    cv2.putText(vis, f"reference {ann.selected_case_id} target={ann.target_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(vis, f"domain={ann.domain_category} level={ann.memory_level}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 80, 80), 1, cv2.LINE_AA)
    return vis


def make_comparison_videos(
    eval_root: Path,
    overlay_by_key: dict[tuple[str, str, str, str], Path],
    summaries: list[dict[str, Any]],
    methods: list[str],
    profile_specs: list[dict[str, Any]] | None,
    tile_width: int,
    image_height: int,
    cols: int,
) -> list[dict[str, str]]:
    summary_lookup = {
        (row["prompt_variant"], row["selected_case_id"], str(row.get("profile") or ""), row["method"]): row
        for row in summaries
    }
    groups: dict[tuple[str, str], list[str]] = {}
    for row in summaries:
        groups.setdefault((row["prompt_variant"], row["selected_case_id"]), []).append(row["method"])
    manifest: list[dict[str, str]] = []
    for (variant, case_id), available_methods in sorted(groups.items()):
        items: list[tuple[str, Path]] = []
        # A supplied spec is the canonical ordering and, critically, retains
        # both 20% and 80% profiles of the same method.  The old fallback is
        # kept only for historical one-profile-per-method folders.
        candidates = profile_specs or [
            {"profile": str(row.get("profile") or ""), "method": method, "label": METHOD_LABELS.get(method, method)}
            for method in methods
            for row in summaries
            if row["method"] == method and row["prompt_variant"] == variant and row["selected_case_id"] == case_id
        ]
        seen_specs: set[tuple[str, str]] = set()
        for spec in candidates:
            profile = str(spec.get("profile") or "")
            method = str(spec["method"])
            identity = (profile, method)
            if identity in seen_specs:
                continue
            seen_specs.add(identity)
            if method not in available_methods:
                continue
            path = overlay_by_key.get((variant, case_id, profile, method))
            if path is None:
                continue
            row = summary_lookup.get((variant, case_id, profile, method))
            if row is None:
                continue
            prune = row.get("prune_ratio")
            prune_label = "PrNA" if prune is None else f"Pr{float(prune) * 100:.0f}%"
            fps = row.get("generation_fps")
            fps_label = "FPSNA" if fps is None else f"FPS{float(fps):.2f}"
            label = (
                f"{str(spec.get('label') or METHOD_LABELS.get(method, method))}"
                f" DINO{float(row.get('dino_score') or 0):.2f}"
                f" Color{float(row.get('color_score') or row.get('color_ref_best_mean') or 0):.2f}"
                f" Pixel{float(row.get('pixel_score') or row.get('pixel_ref_best_mean') or 0):.2f}"
                f" {prune_label}"
                f" {fps_label}"
            )
            items.append((label, path))
        if not items:
            continue
        out_path = eval_root / "score_comparison_videos" / variant / f"{case_id}_score_comparison.mp4"
        ok = make_grid_video(items, out_path, tile_width, image_height, cols)
        if ok:
            manifest.append({"prompt_variant": variant, "case_id": case_id, "comparison_video": str(out_path)})
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=Path("data/CMBench"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--profile-spec",
        type=Path,
        default=None,
        help="Optional YAML list of {profile, method, label}; required for a multi-operating-point score grid.",
    )
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--video-name", default="continuation.mp4")
    parser.add_argument(
        "--owl-model",
        default=str(_LOCAL_MODEL_ROOT / "google" / "owlvit-base-patch32"),
    )
    parser.add_argument(
        "--sam-model",
        default=str(_LOCAL_MODEL_ROOT / "facebook" / "sam2.1-hiera-large"),
    )
    parser.add_argument(
        "--dino-model",
        default=str(_LOCAL_MODEL_ROOT / "facebook" / "dinov2-base"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-mode", choices=["sam", "box"], default="sam")
    parser.add_argument("--owl-threshold", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-area-frac", type=float, default=0.0005)
    parser.add_argument("--max-area-frac", type=float, default=0.65)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pixel-size", type=int, default=128)
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    parser.add_argument("--full-frame-area-threshold", type=float, default=0.80)
    parser.add_argument(
        "--scoring-path",
        choices=["auto", "owl_sam_dino", "direct_dino"],
        default="auto",
        help="auto and owl_sam_dino: bbox-seeded SAM/SAM2 for every reference; direct_dino is an explicit ablation.",
    )
    parser.add_argument("--class-threshold", type=float, default=0.05)
    parser.add_argument("--detail-threshold", type=float, default=0.50)
    parser.add_argument("--use-pixel-in-detail", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--no-comparison-videos", action="store_true")
    parser.add_argument("--tile-width", type=int, default=420)
    parser.add_argument("--image-height", type=int, default=236)
    parser.add_argument("--cols", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.methods = [canonical_method(method) for method in (args.methods or [])]
    metadata_path = args.metadata or args.benchmark_root / "metadata" / "benchmark_cases.jsonl"
    annotation_path = args.annotations or args.benchmark_root / "metadata" / "eval_annotations.jsonl"
    metadata_cases = load_metadata_cases(metadata_path)
    annotations_by_key = load_latest_annotations(annotation_path, metadata_cases)
    profile_specs = None
    if args.profile_spec is not None:
        spec_document = yaml.safe_load(args.profile_spec.read_text(encoding="utf-8")) or {}
        profile_specs = [
            {**item, "method": canonical_method(item.get("method"))}
            for item in (spec_document.get("profiles") or [])
        ]
        if not all(isinstance(item, dict) and item.get("profile") and item.get("method") for item in profile_specs):
            raise SystemExit("--profile-spec entries must each contain profile and method")
    allowed_profiles = {str(item["profile"]) for item in profile_specs} if profile_specs else None
    selected_filter = set(args.cases or [])
    methods_filter = {canonical_method(method) for method in args.methods} if args.methods else None
    variants_filter = set(args.variants) if args.variants else None
    jobs = discover_jobs(args.output_roots, annotations_by_key, methods_filter, variants_filter, args.video_name)
    if allowed_profiles is not None:
        jobs = [job for job in jobs if job.profile in allowed_profiles]
    if selected_filter:
        jobs = [job for job in jobs if job.selected_case_id in selected_filter or job.output_case_id in selected_filter]
    if not jobs:
        raise SystemExit("no generated videos matched annotations")

    import_runtime_deps()
    args.eval_root.mkdir(parents=True, exist_ok=True)
    scorer = VisualScorer(args.owl_model, args.sam_model, args.dino_model, args.device, args.mask_mode)
    summaries: list[dict[str, Any]] = []
    overlay_by_key: dict[tuple[str, str, str, str], Path] = {}
    for idx, job in enumerate(jobs, start=1):
        ann = annotations_by_key[job.output_case_id]
        job_eval_dir = args.eval_root / "per_video" / job.prompt_variant / ann.selected_case_id / job.profile / job.method
        existing_summary_path = job_eval_dir / "summary.json"
        if existing_summary_path.exists():
            summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
            # Do not reuse an older `auto`/direct-DINO/OWL-SAM-DINO result
            # merely because it happens to share a directory.
            if summary.get("scoring_path") == args.scoring_path and summary.get("profile") == job.profile:
                summaries.append(summary)
                existing_overlay = args.eval_root / "overlays" / job.prompt_variant / ann.selected_case_id / f"{job.profile}__{job.method}_scores.mp4"
                if existing_overlay.exists():
                    overlay_by_key[(job.prompt_variant, ann.selected_case_id, job.profile, job.method)] = existing_overlay
                print(
                    f"[eval] {idx}/{len(jobs)} reuse variant={job.prompt_variant} "
                    f"case={ann.selected_case_id}/{job.output_case_id} profile={job.profile} method={job.method}",
                    flush=True,
                )
                continue
        print(
            f"[eval] {idx}/{len(jobs)} variant={job.prompt_variant} "
            f"case={ann.selected_case_id}/{job.output_case_id} profile={job.profile} method={job.method}",
            flush=True,
        )
        summary, overlay_path = evaluate_job(
            scorer=scorer,
            ann=ann,
            job=job,
            benchmark_root=args.benchmark_root,
            eval_root=args.eval_root,
            frame_stride=args.frame_stride,
            owl_threshold=args.owl_threshold,
            top_k=args.top_k,
            min_area_frac=args.min_area_frac,
            max_area_frac=args.max_area_frac,
            pixel_size=args.pixel_size,
            use_pixel_in_detail=args.use_pixel_in_detail,
            class_threshold=args.class_threshold,
            detail_threshold=args.detail_threshold,
            coverage_threshold=args.coverage_threshold,
            full_frame_area_threshold=args.full_frame_area_threshold,
            scoring_path=args.scoring_path,
            max_frames=args.max_frames,
            save_overlay=not args.no_overlays,
        )
        summaries.append(summary)
        if overlay_path is not None:
            overlay_by_key[(job.prompt_variant, job.selected_case_id, job.profile, job.method)] = overlay_path

    write_csv(args.eval_root / "video_summary.csv", summaries)
    write_jsonl(args.eval_root / "video_summary.jsonl", summaries)
    if not args.no_comparison_videos and not args.no_overlays:
        manifest = make_comparison_videos(
            eval_root=args.eval_root,
            overlay_by_key=overlay_by_key,
            summaries=summaries,
            methods=args.methods,
            profile_specs=profile_specs,
            tile_width=args.tile_width,
            image_height=args.image_height,
            cols=args.cols,
        )
        (args.eval_root / "score_comparison_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"[eval] summary={args.eval_root / 'video_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
