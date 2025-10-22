# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Required imports
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from mapanything.models import MapAnything
from mapanything.utils.colmap import qvec2rotmat, read_images_text
from mapanything.utils.image import load_images
import open3d as o3d

from zyb_tools.align_estimated_to_gt import align_predictions
import copy


SCAN_ID = 55
SCAN_NAME = f"scan{int(SCAN_ID):02d}"

DTU_SET_ROOT = Path("DTU/set_23_24_33")
DTU_SCAN_ROOT = DTU_SET_ROOT / SCAN_NAME
IMAGES_DIR = DTU_SCAN_ROOT / "images"
GT_COLMAP_DIR = DTU_SCAN_ROOT / "sparse" / "0"
GT_DENSE_PLY = DTU_SCAN_ROOT / "dense" / "fused.ply"
output_root = Path("output") / "DTU" / SCAN_NAME


gt_images = read_images_text(str(GT_COLMAP_DIR / "images.txt"))
gt_images_by_name = {img.name: img for img in gt_images.values()}

# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requires internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

# COLMAP helpers ------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")


def list_image_paths(images_source: str | Sequence[str]) -> List[Path]:
    if isinstance(images_source, (str, Path)):
        image_dir = Path(images_source)
        paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        return paths


def rotation_matrix_to_qvec(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    return q


def write_colmap_cameras(path: Path, entries: Sequence[dict]) -> None:
    header = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"# Number of cameras: {len(entries)}\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for entry in entries:
            line = (
                f"{entry['camera_id']} PINHOLE {entry['width']} {entry['height']} "
                f"{entry['fx']:.10f} {entry['fy']:.10f} {entry['cx']:.10f} {entry['cy']:.10f}\n"
            )
            f.write(line)


def write_colmap_images(path: Path, entries: Sequence[dict]) -> None:
    header = (
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        f"# Number of images: {len(entries)}, mean observations per image: 0\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for entry in entries:
            q = entry["qvec"]
            t = entry["tvec"]
            f.write(
                f"{entry['image_id']} "
                f"{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} "
                f"{entry['camera_id']} {entry['name']}\n"
            )
            f.write("\n")


# Load and preprocess images from a folder or list of paths
images = IMAGES_DIR
image_paths = list_image_paths(images)
views = load_images([str(p) for p in image_paths])

# Manually specified camera intrinsics in the original image resolution (fx, fy, cx, cy)
MANUAL_FX = 2892.3299999999999
MANUAL_FY = 2883.1799999999998
MANUAL_CX = 777.0
MANUAL_CY = 581.0

scale_factors = []
original_shapes: List[Tuple[int, int]] = []
for idx, view in enumerate(views):
    resized_h, resized_w = map(int, np.squeeze(view["true_shape"]))
    original_shape = view.get("original_shape")
    if original_shape is not None:
        orig_h, orig_w = map(int, np.squeeze(original_shape))
    else:
        orig_h, orig_w = resized_h, resized_w

    original_shapes.append((orig_h, orig_w))

    if resized_w == 0 or resized_h == 0:
        scale_x = scale_y = 1.0
    else:
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

    scale_factors.append((scale_x, scale_y))

    fx_resized = MANUAL_FX / scale_x if scale_x != 0 else MANUAL_FX
    fy_resized = MANUAL_FY / scale_y if scale_y != 0 else MANUAL_FY
    cx_resized = MANUAL_CX / scale_x if scale_x != 0 else MANUAL_CX
    cy_resized = MANUAL_CY / scale_y if scale_y != 0 else MANUAL_CY

    intrinsics = torch.tensor(
        [
            [fx_resized, 0.0, cx_resized],
            [0.0, fy_resized, cy_resized],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    view["intrinsics"] = intrinsics

    image_name = image_paths[idx].name if idx < len(image_paths) else f"view_{idx:03d}.png"
    gt_image = gt_images_by_name.get(image_name)
    gt_qvec = gt_image.qvec.astype(np.float64)
    gt_tvec = gt_image.tvec.astype(np.float64)
    gt_R_cw = qvec2rotmat(gt_qvec)
    gt_R_wc = gt_R_cw.T
    gt_t_wc = -gt_R_wc @ gt_tvec

    gt_pose_cam2world = np.eye(4, dtype=np.float32)
    gt_pose_cam2world[:3, :3] = gt_R_wc.astype(np.float32)
    gt_pose_cam2world[:3, 3] = gt_t_wc.astype(np.float32)

    view["camera_poses"] = torch.from_numpy(gt_pose_cam2world[None, ...])

# Run inference
predictions = model.infer(
    views,                            # Input views
    memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
    use_amp=True,                     # Use mixed precision inference (recommended)
    amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
    apply_mask=True,                  # Apply masking to dense geometry outputs
    mask_edges=True,                  # Remove edge artifacts by using normals and depth
    apply_confidence_mask=False,      # Filter low-confidence regions
    confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
)


output_root.mkdir(parents=True, exist_ok=True)

sparse_root = output_root / "sparse"
ma_sparse_dir = sparse_root / "MA"
ma_sparse_dir.mkdir(parents=True, exist_ok=True)
align_sparse_dir = sparse_root / "align"
align_sparse_dir.mkdir(parents=True, exist_ok=True)

gt_sparse_dir = sparse_root / "gt"
gt_sparse_dir.mkdir(parents=True, exist_ok=True)

DEPTH_SCALE_MM = 1000.0


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


saved_view_idx = 0
all_points = []
all_colors = []
camera_entries: List[dict] = []
image_entries: List[dict] = []
gt_camera_entries: List[dict] = []
gt_image_entries: List[dict] = []

# Access results for each view - Complete list of metric outputs
for i, pred in enumerate(predictions):
    # Geometry outputs
    pts3d = pred["pts3d"]                     # 3D points in world coordinates (B, H, W, 3)
    pts3d_cam = pred["pts3d_cam"]             # 3D points in camera coordinates (B, H, W, 3)
    depth_z = pred["depth_z"]                 # Z-depth in camera frame (B, H, W, 1)
    depth_along_ray = pred["depth_along_ray"] # Depth along ray in camera frame (B, H, W, 1)

    # Camera outputs
    ray_directions = pred["ray_directions"]   # Ray directions in camera frame (B, H, W, 3)
    intrinsics = pred["intrinsics"]           # Recovered pinhole camera intrinsics (B, 3, 3)
    camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
    cam_trans = pred["cam_trans"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world translation in world frame (B, 3)
    cam_quats = pred["cam_quats"]             # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world quaternion in world frame (B, 4)

    # Quality and masking
    confidence = pred["conf"]                 # Per-pixel confidence scores (B, H, W)
    mask = pred["mask"]                       # Combined validity mask (B, H, W, 1)
    non_ambiguous_mask = pred["non_ambiguous_mask"]                # Non-ambiguous regions (B, H, W)
    non_ambiguous_mask_logits = pred["non_ambiguous_mask_logits"]  # Mask logits (B, H, W)

    # Scaling
    metric_scaling_factor = pred["metric_scaling_factor"]  # Applied metric scaling (B,)

    # Original input
    img_no_norm = pred["img_no_norm"]         # Denormalized input images for visualization (B, H, W, 3)

    # Save key outputs to disk for each view in the batch
    pts3d_np = to_numpy(pts3d)
    depth_np = to_numpy(depth_z)
    intrinsics_np = to_numpy(intrinsics)
    poses_np = to_numpy(camera_poses)
    mask_np = to_numpy(mask) if mask is not None else None
    img_np = to_numpy(img_no_norm) if img_no_norm is not None else None

    depth_np = depth_np[..., 0]
    mask_np = mask_np[..., 0]

    batch_size = pts3d_np.shape[0]

    for b in range(batch_size):
        view_dir = output_root / f"view_{saved_view_idx:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)

        points = pts3d_np[b]
        depth_map = np.squeeze(depth_np[b]).astype(np.float32)
        mask_view = mask_np[b] if mask_np is not None else None
        image_view = img_np[b] if img_np is not None else None
        scale_x, scale_y = (
            scale_factors[saved_view_idx] if saved_view_idx < len(scale_factors) else (1.0, 1.0)
        )

        points_flat = points.reshape(-1, 3).astype(np.float32)
        valid_flat = np.isfinite(points_flat).all(axis=1)
        if mask_view is not None:
            valid_flat &= mask_view.reshape(-1) > 0.5

        valid_points = points_flat[valid_flat]


        colors_flat = image_view.reshape(-1, 3)
        colors_float = np.clip(colors_flat, 0.0, 1.0).astype(np.float32)
        valid_colors = colors_float[valid_flat]
        all_points.append(valid_points.astype(np.float32))
        all_colors.append(valid_colors.astype(np.float32))
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(valid_points.astype(np.float64))
        point_cloud.colors = o3d.utility.Vector3dVector(valid_colors.astype(np.float64))
        o3d.io.write_point_cloud(str(view_dir / "points.ply"), point_cloud)
        # 保存点云 并且加入全部点云合并列表



        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(depth_map * DEPTH_SCALE_MM, 0.0, float(np.iinfo(np.uint16).max))
        depth_mm = depth_mm.astype(np.uint16)
        cv2.imwrite(str(view_dir / "depth.png"), depth_mm)
        cv2.imwrite(str(view_dir / "depth.tiff"), depth_mm)
        # 保存深度图
        # Also save as 16-bit TIFF

        intrinsics_rescaled = intrinsics_np[b].astype(np.float64).copy()
        intrinsics_rescaled[0, 0] *= scale_x
        intrinsics_rescaled[0, 2] *= scale_x
        intrinsics_rescaled[1, 1] *= scale_y
        intrinsics_rescaled[1, 2] *= scale_y

        np.savetxt(
            view_dir / "intrinsics.txt",
            intrinsics_rescaled,
            fmt="%.8f",
            header="Camera intrinsics (3x3)",
        )
        pose_cam2world = poses_np[b].astype(np.float64)
        np.savetxt(
            view_dir / "camera_pose.txt",
            pose_cam2world,
            fmt="%.8f",
            header="Camera pose (4x4 cam2world)",
        )
        # 保存内参外参(单个图片)

        camera_id = saved_view_idx + 1
        image_id = saved_view_idx + 1
        orig_h, orig_w = (
            original_shapes[saved_view_idx]
            if saved_view_idx < len(original_shapes)
            else (int(scale_y * resized_h), int(scale_x * resized_w))
        )
        image_name = (
            image_paths[saved_view_idx].name
            if saved_view_idx < len(image_paths)
            else f"view_{saved_view_idx:03d}.png"
        )

        camera_entries.append(
            dict(
                camera_id=camera_id,
                width=int(orig_w),
                height=int(orig_h),
                fx=float(intrinsics_rescaled[0, 0]),
                fy=float(intrinsics_rescaled[1, 1]),
                cx=float(intrinsics_rescaled[0, 2]),
                cy=float(intrinsics_rescaled[1, 2]),
            )
        )
        # 保存内参信息到colmap格式

        R_wc = pose_cam2world[:3, :3]
        t_wc = pose_cam2world[:3, 3]
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        qvec = rotation_matrix_to_qvec(R_cw)

        image_entries.append(
            dict(
                image_id=image_id,
                camera_id=camera_id,
                name=image_name,
                qvec=qvec,
                tvec=t_cw.astype(np.float64),
            )
        )
        # 保存外参信息到colmap格式

        saved_view_idx += 1


merged_points = np.concatenate(all_points, axis=0)
merged_colors = np.concatenate(all_colors, axis=0)
merged_pc = o3d.geometry.PointCloud()
merged_pc.points = o3d.utility.Vector3dVector(merged_points.astype(np.float64))
merged_pc.colors = o3d.utility.Vector3dVector(np.clip(merged_colors, 0.0, 1.0).astype(np.float64))
ma_combined_path = ma_sparse_dir / "points3D_MA.ply"
o3d.io.write_point_cloud(str(ma_combined_path), merged_pc)
print(
    f"Saved merged point cloud with {len(merged_points)} points to '{ma_combined_path}'"
)




write_colmap_cameras(ma_sparse_dir / "cameras.txt", camera_entries)
write_colmap_images(ma_sparse_dir / "images.txt", image_entries)
print(f"Saved outputs for {saved_view_idx} views to '{ma_sparse_dir.resolve()}'")

align_dir = output_root / "align"

scale, rot, trans, aligned_pose_np, gt_pose_t, cameras_dict = align_predictions(
    estimated_root=ma_sparse_dir,
    gt_colmap_dir=str(GT_COLMAP_DIR),
    gt_ply=str(GT_DENSE_PLY),
    output_dir=align_dir,
    voxel_size=0.04,
)

camera_entries_align = copy.deepcopy(camera_entries)
image_entries_align = copy.deepcopy(image_entries)



for cam_entry, img_entry, aligned_pose in zip(
    camera_entries_align, image_entries_align, aligned_pose_np
):
    cam_entry["pose_cam2world_aligned"] = aligned_pose

    R_wc_aligned = aligned_pose[:3, :3]
    t_wc_aligned = aligned_pose[:3, 3]
    R_cw_aligned = R_wc_aligned.T
    t_cw_aligned = -R_cw_aligned @ t_wc_aligned
    qvec_aligned = rotation_matrix_to_qvec(R_cw_aligned)

    img_entry["qvec"] = qvec_aligned.astype(np.float64)
    img_entry["tvec"] = t_cw_aligned.astype(np.float64)

write_colmap_cameras(align_sparse_dir / "cameras.txt", camera_entries_align)
write_colmap_images(align_sparse_dir / "images.txt", image_entries_align)


camera_entries_gt = copy.deepcopy(camera_entries)
image_entries_gt = copy.deepcopy(image_entries)



for cam_entry, img_entry, gt_pose in zip(
    camera_entries_gt, image_entries_gt, np.array(gt_pose_t)
):
    cam_entry["pose_cam2world_aligned"] = gt_pose

    R_wc_aligned = gt_pose[:3, :3]
    t_wc_aligned = gt_pose[:3, 3]
    R_cw_aligned = R_wc_aligned.T
    t_cw_aligned = -R_cw_aligned @ t_wc_aligned
    qvec_aligned = rotation_matrix_to_qvec(R_cw_aligned)

    img_entry["qvec"] = qvec_aligned.astype(np.float64)
    img_entry["tvec"] = t_cw_aligned.astype(np.float64)

write_colmap_cameras(gt_sparse_dir / "cameras.txt", camera_entries_gt)
write_colmap_images(gt_sparse_dir / "images.txt", image_entries_gt)
