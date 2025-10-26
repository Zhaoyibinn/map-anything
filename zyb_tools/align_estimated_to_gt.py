#!/usr/bin/env python3
"""Aligns predicted camera poses to DTU ground-truth poses and merges point clouds.

This script reads the predicted camera poses and merged point cloud produced by
``demo.py`` (stored in the ``output/`` directory by default), aligns the
predictions to the DTU ground-truth COLMAP reconstruction, and writes:

1. Aligned camera pose matrices for each predicted view.
2. The aligned predicted point cloud (voxel-downsampled, colored green).
3. A merged point cloud combining the aligned predictions with the ground truth
    (both voxel-downsampled; GT colored red).

Example usage (all paths are configurable with command-line flags)::

    python scripts/align_estimated_to_gt.py \
        --estimated-root output \
        --gt-colmap-dir DTU/set_23_24_33/scan24/sparse/0 \
        --gt-ply DTU/gt/Points/stl/stl024_total.ply \
        --output-dir output/aligned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open3d as o3d
import roma
import torch

from mapanything.utils.colmap import Image as ColmapImage
from mapanything.utils.colmap import (
    qvec2rotmat,
    read_cameras_text,
    read_images_text,
    rotmat2qvec,
    write_cameras_text,
    write_images_text,
)


def make_colored_point_cloud(points: np.ndarray, color: np.ndarray) -> o3d.geometry.PointCloud:
    """Create a point cloud with a uniform RGB color."""

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = np.tile(color.astype(np.float64), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def voxel_down_sample_safe(
    pcd: o3d.geometry.PointCloud, voxel_size: float
) -> o3d.geometry.PointCloud:
    """Downsample ``pcd`` if ``voxel_size`` is positive, otherwise return original."""

    if voxel_size <= 0:
        return pcd
    downsampled = pcd.voxel_down_sample(voxel_size)
    return downsampled if len(downsampled.points) > 0 else pcd


def get_med_dist_between_poses(poses: torch.Tensor) -> float:
    """Median pairwise distance between camera centers."""
    centers = poses[:, :3, 3]
    dists = torch.cdist(centers, centers, p=2)
    # Only consider the upper triangle without the diagonal
    triu = dists[torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)]
    return torch.median(triu).item()


def align_multiple_poses(
    src_poses: torch.Tensor, target_poses: torch.Tensor
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Compute similarity transform aligning ``src_poses`` to ``target_poses``.

    Returns scale ``s``, rotation matrix ``R`` and translation ``T`` such that::

        aligned_center = s * (R @ src_center) + T

    """

    assert src_poses.shape == target_poses.shape
    assert src_poses.ndim == 3 and src_poses.shape[-2:] == (4, 4)

    eps = get_med_dist_between_poses(src_poses) / 5

    def center_and_z(poses: torch.Tensor) -> torch.Tensor:
        centers = poses[:, :3, 3]
        z_dirs = poses[:, :3, 2]
        return torch.cat([centers, centers + eps * z_dirs], dim=0)

    R, T, s = roma.rigid_points_registration(
        center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True
    )
    return s.item(), R, T


def load_estimated_predictions(
    estimated_root: Path,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, Dict[int, np.ndarray]]:
    images_path = estimated_root / "images.txt"
    cameras_path = estimated_root / "cameras.txt"

    if not images_path.exists():
        raise FileNotFoundError(
            f"Predicted COLMAP images file not found: {images_path}. "
            "Ensure demo.py generated COLMAP-formatted outputs."
        )

    entries: List[Dict[str, np.ndarray]] = []
    with images_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                raise ValueError(f"Malformed image entry in {images_path}: '{line}'")
            image_id = int(parts[0])
            qvec = np.array(tuple(map(float, parts[1:5])), dtype=np.float64)
            tvec = np.array(tuple(map(float, parts[5:8])), dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]
            entries.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "name": name,
                    "qvec": qvec,
                    "tvec": tvec,
                }
            )
            _ = next(f, "")

    if not entries:
        raise RuntimeError(f"No image entries found in {images_path}")

    poses = []
    for entry in entries:
        R_cw = qvec2rotmat(entry["qvec"])
        t_cw = entry["tvec"]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_wc
        pose[:3, 3] = t_wc
        poses.append(pose)

    cameras = {}
    if cameras_path.exists():
        cameras = read_cameras_text(str(cameras_path))

    return entries, np.stack(poses, axis=0), cameras


def load_gt_cam2world(colmap_dir: Path, unit_scale: float) -> Dict[str, np.ndarray]:
    images_txt = colmap_dir / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(f"Missing COLMAP images file: {images_txt}")

    images: Dict[int, ColmapImage] = read_images_text(str(images_txt))
    pose_dict: Dict[str, np.ndarray] = {}
    for img in images.values():
        R_cw = qvec2rotmat(img.qvec)
        t_cw = img.tvec
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R_wc
        pose[:3, 3] = t_wc * unit_scale
        pose_dict[img.name] = pose
    return pose_dict


def build_pose_tensors(
    estimated: np.ndarray, gt: Sequence[np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    src = torch.from_numpy(estimated).float()
    tgt = torch.from_numpy(np.stack(gt, axis=0)).float()
    return src, tgt


def apply_alignment_to_pose(
    pose: np.ndarray, scale: float, rot: torch.Tensor, trans: torch.Tensor
) -> np.ndarray:
    pose_t = torch.from_numpy(pose).float()
    aligned = pose_t.clone()
    aligned[:3, :3] = rot @ pose_t[:3, :3]
    aligned[:3, 3] = scale * (rot @ pose_t[:3, 3]) + trans
    return aligned.numpy()


def apply_alignment_to_points(
    points: np.ndarray, scale: float, rot: torch.Tensor, trans: torch.Tensor
) -> np.ndarray:
    pts = torch.from_numpy(points).float().t()
    aligned = scale * (rot @ pts) + trans[:, None]
    return aligned.t().numpy()


def align_predictions(
    estimated_root: Path | str,
    gt_colmap_dir: Path | str,
    gt_ply: Path | str,
    output_dir: Path | str,
    voxel_size: float = 0.04,
    unit_scale: float = 1.0,
    scale_trans: bool = True,
):
    estimated_root = Path(estimated_root)
    gt_colmap_dir = Path(gt_colmap_dir)
    gt_ply = Path(gt_ply)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_entries, est_pose_np, cameras_dict = load_estimated_predictions(estimated_root)
    est_pose_names = [entry["name"] for entry in pred_entries]

    gt_pose_map = load_gt_cam2world(gt_colmap_dir, unit_scale)


    gt_pose_np = [gt_pose_map[name] for name in est_pose_names]

    est_pose_t, gt_pose_t = build_pose_tensors(est_pose_np, gt_pose_np)
    scale, rot, trans = align_multiple_poses(est_pose_t, gt_pose_t)

    applied_scale = scale if scale_trans else 1.0

    aligned_pose_np = np.stack(
        [apply_alignment_to_pose(pose, applied_scale, rot, trans) for pose in est_pose_np],
        axis=0,
    )

    rot_np = rot.numpy()
    trans_np = trans.numpy()

    aligned_pose_dir = output_dir / "poses"
    aligned_pose_dir.mkdir(parents=True, exist_ok=True)

    aligned_images: Dict[int, ColmapImage] = {}

    for entry, aligned in zip(pred_entries, aligned_pose_np):
        # np.savetxt(
        #     aligned_pose_dir / f"{Path(entry['name']).stem}_aligned_pose.txt",
        #     aligned,
        #     fmt="%.8f",
        #     header="Aligned camera pose (4x4 cam2world)",
        #     comments="# ",
        # )

        R_wc_aligned = aligned[:3, :3]
        t_wc_aligned = aligned[:3, 3]
        R_cw_aligned = R_wc_aligned.T
        t_cw_aligned = -R_cw_aligned @ t_wc_aligned
        qvec_aligned = rotmat2qvec(R_cw_aligned)

        aligned_images[entry["image_id"]] = ColmapImage(
            id=entry["image_id"],
            qvec=qvec_aligned.astype(np.float64),
            tvec=t_cw_aligned.astype(np.float64),
            camera_id=entry["camera_id"],
            name=entry["name"],
            xys=np.empty((0, 2), dtype=np.float64),
            point3D_ids=np.empty((0,), dtype=np.int64),
        )

    transform_info = {
        "scale": scale,
        "scale_applied": applied_scale,
        "rotation": rot_np.tolist(),
        "translation": trans_np.tolist(),
        "view_to_image": {name: name for name in est_pose_names},
    }
    # with open(output_dir / "alignment_transform.json", "w", encoding="utf-8") as f:
    #     json.dump(transform_info, f, indent=2)

    combined_estimated_path = estimated_root / "points3D_MA.ply"
    if not combined_estimated_path.exists():
        raise FileNotFoundError(
            f"没有找到之前已经合起来的est点云{combined_estimated_path}"
        )

    est_pcd_orig = o3d.io.read_point_cloud(str(combined_estimated_path))
    est_points = np.asarray(est_pcd_orig.points)


    aligned_points = apply_alignment_to_points(est_points, applied_scale, rot, trans)
    # 变换点云 对齐
    pred_color = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    est_pcd_aligned = make_colored_point_cloud(aligned_points, pred_color)
    est_pcd_aligned_down = voxel_down_sample_safe(est_pcd_aligned, voxel_size)

    aligned_pred_path = output_dir / "predicted_points_aligned_down.ply"
    o3d.io.write_point_cloud(str(aligned_pred_path), est_pcd_aligned_down)



    gt_pcd_raw = o3d.io.read_point_cloud(str(gt_ply))
    gt_points = np.asarray(gt_pcd_raw.points) * unit_scale
    gt_color = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    gt_pcd_colored = make_colored_point_cloud(gt_points, gt_color)
    gt_pcd_down = voxel_down_sample_safe(gt_pcd_colored, voxel_size)

    est_down_points = np.asarray(est_pcd_aligned_down.points)
    gt_down_points = np.asarray(gt_pcd_down.points)
    merged_points = np.vstack([est_down_points, gt_down_points])
    est_down_colors = np.asarray(est_pcd_aligned_down.colors)
    gt_down_colors = np.asarray(gt_pcd_down.colors)
    merged_colors = np.vstack([est_down_colors, gt_down_colors])

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    merged_path = output_dir / "predicted_plus_gt_points.ply"
    o3d.io.write_point_cloud(str(merged_path), merged_pcd)



    # print("Alignment complete.")
    # print(f" - Saved alignment data to {output_dir}")
    # print(f" - Aligned poses in {aligned_pose_dir}")
    # print(f" - Aligned predicted point cloud: {aligned_pred_path}")
    # print(f" - Combined point cloud: {merged_path}")

    return scale, rot, trans, aligned_pose_np, gt_pose_t,cameras_dict



def main(args: argparse.Namespace) -> None:
    align_predictions(
        estimated_root=args.estimated_root,
        gt_colmap_dir=args.gt_colmap_dir,
        gt_ply=args.gt_ply,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimated-root",
        type=str,
        default="output",
        help="Directory containing predicted view_* folders and combined_points.ply",
    )
    parser.add_argument(
        "--gt-colmap-dir",
        type=str,
        default="DTU/set_23_24_33/scan24/sparse/0",
        help="Directory holding COLMAP images.txt (and cameras.txt).",
    )
    parser.add_argument(
        "--gt-ply",
        type=str,
        default="DTU/set_23_24_33/scan24/dense/fused.ply",
        help="Ground-truth point cloud to merge with predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/aligned",
        help="Directory where aligned artifacts should be written.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help=(
            "Voxel size (in scene units) for downsampling predicted and ground-truth "
            "point clouds. Set to <=0 to disable downsampling."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
