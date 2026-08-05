#!/usr/bin/env python3
"""
BOP to LEPAUTE Monocular Dataset Converter
Production-ready version with progress monitoring, robust I/O, split-aware tree traversal,
and explicit train/test dataset partitioning.
"""

import argparse
import json
import logging
import shutil
import uuid
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] BOP_CONVERTER: %(message)s"
)
logger = logging.getLogger("BOP_CONVERTER")

def calculate_relative_pose(R_A: np.ndarray, t_A: np.ndarray, R_B: np.ndarray, t_B: np.ndarray) -> List[float]:
    """Computes exact SE(3) Lie Algebra twist (xi) linking Pose A to Pose B."""
    R_A_T = R_A.T
    R_rel = R_B @ R_A_T
    t_rel = t_B - (R_rel @ t_A)
    
    trace_R = np.trace(R_rel)
    cos_theta = np.clip((trace_R - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_theta)
    
    phi_raw = np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1]
    ])
    
    if theta > 1e-4:
        phi = (theta / (2.0 * np.sin(theta))) * phi_raw
        K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
        I = np.eye(3)
        half_theta = theta / 2.0
        V_inv = I - 0.5 * K + (1.0 - (theta * np.cos(half_theta)) / (2.0 * np.sin(half_theta))) * (K @ K) / (theta ** 2)
        rho = V_inv @ t_rel.flatten()
    else:
        phi = 0.5 * phi_raw
        K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
        I = np.eye(3)
        V_inv = I - 0.5 * K + (1.0 / 12.0) * (K @ K)
        rho = V_inv @ t_rel.flatten()

    return np.concatenate([rho, phi]).astype(float).tolist()

def process_scene_pair(
    frame_a_id: str, 
    frame_b_id: str, 
    scene_path: Path, 
    out_dir: Path, 
    scene_gt: Dict, 
    target_obj_ids: Optional[List[int]],
    translation_scale: float
) -> List[Dict[str, Any]]:
    
    gt_a = scene_gt.get(frame_a_id)
    gt_b = scene_gt.get(frame_b_id)
    
    if gt_a is None or gt_b is None:
        return []

    obj_dict_a = {obj["obj_id"]: obj for obj in gt_a}
    obj_dict_b = {obj["obj_id"]: obj for obj in gt_b}
    common_objs = set(obj_dict_a.keys()).intersection(set(obj_dict_b.keys()))

    if target_obj_ids:
        common_objs = common_objs.intersection(set(target_obj_ids))

    if not common_objs:
        return []

    rgb_dir = scene_path / "rgb"
    rgb_a_path = next(rgb_dir.glob(f"{int(frame_a_id):06d}.*"), None)
    rgb_b_path = next(rgb_dir.glob(f"{int(frame_b_id):06d}.*"), None)

    if not rgb_a_path or not rgb_b_path:
        return []

    uuid_a, uuid_b = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
    # Enforce strictly .jpg extension to match the target directory naming convention
    out_a, out_b = f"frame_{uuid_a}.jpg", f"frame_{uuid_b}.jpg"
    
    # Safely transcode any image format (like BOP's native PNGs) into actual high-quality JPEGs
    img_a = cv2.imread(str(rgb_a_path))
    img_b = cv2.imread(str(rgb_b_path))
    
    if img_a is None or img_b is None:
        return []
        
    cv2.imwrite(str(out_dir / out_a), img_a, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_dir / out_b), img_b, [cv2.IMWRITE_JPEG_QUALITY, 95])

    records = []
    # Safeguard against zero division if translation_scale is improperly configured
    safe_scale = translation_scale if translation_scale != 0 else 1.0

    for obj_id in common_objs:
        R_A = np.array(obj_dict_a[obj_id]["cam_R_m2c"]).reshape(3, 3)
        t_A = np.array(obj_dict_a[obj_id]["cam_t_m2c"]).reshape(3, 1) / safe_scale
        R_B = np.array(obj_dict_b[obj_id]["cam_R_m2c"]).reshape(3, 3)
        t_B = np.array(obj_dict_b[obj_id]["cam_t_m2c"]).reshape(3, 1) / safe_scale

        records.append({
            "timestamp": 0.0,
            "lie_params": calculate_relative_pose(R_A, t_A, R_B, t_B),
            "loss": 0.0,
            "detected_object": f"obj_{obj_id:06d}",
            "confidence": 1.0,
            "frame_a": out_a,
            "frame_b": out_b,
            "origin_scene": str(scene_path.name)
        })
    return records


def process_bop_scene(scene_path: Path, out_dir: Path, stride: int, target_obj_ids: Optional[List[int]], translation_scale: float) -> List[Dict]:
    scene_gt_path = scene_path / "scene_gt.json"
    if not scene_gt_path.exists():
        return []

    try:
        with open(scene_gt_path, "r") as f:
            scene_gt = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read GT file {scene_gt_path}: {e}")
        return []

    frame_ids = sorted(scene_gt.keys(), key=int)
    scene_records = []
    for i in range(0, len(frame_ids) - stride, stride):
        scene_records.extend(
            process_scene_pair(
                frame_ids[i], frame_ids[i + stride], scene_path, out_dir, scene_gt, target_obj_ids, translation_scale
            )
        )
    return scene_records


def run_conversion(
    bop_dir: str, 
    output_dir: str, 
    stride: int, 
    target_obj_ids: Optional[List[int]], 
    workers: int, 
    translation_scale: float,
    splits: Optional[List[str]]
):
    bop_path, out_path = Path(bop_dir), Path(output_dir)

    # Robust multi-level scene discovery targeting scene_gt.json boundaries
    scene_dirs = []
    if splits:
        for split_name in splits:
            split_path = bop_path / split_name
            if split_path.is_dir():
                logger.info(f"Scanning split directory: {split_path}")
                scene_dirs.extend([p.parent for p in split_path.rglob("scene_gt.json")])
            else:
                logger.warning(f"Target split directory missing, skipping: {split_path}")
    else:
        logger.info(f"No specific splits provided. Deep scanning entire structure: {bop_path}")
        scene_dirs = [p.parent for p in bop_path.rglob("scene_gt.json")]

    # Deduplicate in case of symlinks
    scene_dirs = sorted(list(set(scene_dirs)))
    logger.info(f"Discovered {len(scene_dirs)} valid BOP scenes for processing.")

    if not scene_dirs:
        logger.error("No valid BOP scenes found. Please verify the dataset path and splits.")
        sys.exit(1)

    # Categorize scenes into Train and Test splits based on directory nomenclature
    categorized_scenes = {"train": [], "test": []}
    for scene in scene_dirs:
        # Check parent directories for testing/validation keywords
        path_parts = [p.name.lower() for p in scene.parents] + [scene.name.lower()]
        if any("test" in part or "val" in part for part in path_parts):
            categorized_scenes["test"].append(scene)
        else:
            categorized_scenes["train"].append(scene)

    for category, scenes in categorized_scenes.items():
        if not scenes:
            logger.info(f"No scenes discovered for the '{category}' split. Skipping.")
            continue
            
        logger.info(f"Initiating processing for '{category}' split ({len(scenes)} scenes)...")
        
        # Output directory is now directly the category (train/ or test/) with no sub-folders
        category_out_path = out_path / category
        category_out_path.mkdir(parents=True, exist_ok=True)
        
        all_records = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_bop_scene, s, category_out_path, stride, target_obj_ids, translation_scale
                ): s for s in scenes
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Converting {category.capitalize()} Data"):
                try:
                    all_records.extend(future.result())
                except Exception as e:
                    logger.error(f"Process pipeline collapsed for a scene in '{category}' split: {e}")

        # DECENTRALIZED JSON ARCHITECTURE: Save 1 JSON per image/record directly next to the images
        for record in tqdm(all_records, desc=f"Generating JSON metadata for {category.capitalize()}"):
            frame_a_stem = Path(record["frame_a"]).stem
            obj_str = record["detected_object"]
            json_filename = f"{frame_a_stem}_{obj_str}.json"
            
            json_path = category_out_path / json_filename
            with open(json_path, "w") as f:
                json.dump(record, f, indent=4)
            
        logger.info(f"[{category.capitalize()} Pipeline Complete] Generated {len(all_records)} individual JSON records saved directly to {category_out_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively convert nested BOP datasets to LEPAUTE monocular format.")
    parser.add_argument("--bop_dir", required=True, help="Root path of the BOP dataset (e.g., path/to/ycbv)")
    parser.add_argument("--output_dir", default="./lepaute_dataset", help="Target output directory")
    parser.add_argument("--stride", type=int, default=1, help="Frame skipping stride")
    parser.add_argument("--obj_ids", type=int, nargs="+", help="Specific BOP Object IDs to filter by")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel process workers")
    parser.add_argument("--scale", type=float, default=1000.0, help="Translation normalization scale")
    parser.add_argument("--splits", type=str, nargs="+", help="Specific dataset splits to process (e.g., train_pbr train_real test_all)")
    
    args = parser.parse_args()
    run_conversion(args.bop_dir, args.output_dir, args.stride, args.obj_ids, args.workers, args.scale, args.splits)