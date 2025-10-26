# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Required imports
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from mapanything.models import MapAnything
from mapanything.utils.colmap import (
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_text,
)
from mapanything.utils.image import load_images
from mapanything.utils.geometry import colmap_to_opencv_intrinsics
import open3d as o3d

from zyb_tools.align_estimated_to_gt import align_predictions
import copy

REPLICA_ROOT = Path("Replica")
REPLICA_GT_ROOT = REPLICA_ROOT / "replica_gt"
SCENE_NAME = "office0_sparse"
input_camera = True
input_pose = False
input_depth = False

input_tags: List[str] = []
if input_camera:
    input_tags.append("camera")
if input_pose:
    input_tags.append("pose")
if input_depth:
    input_tags.append("depth")
input_suffix = "+".join(input_tags) if input_tags else "no_inputs"

output_root = Path("output") / "Replica" / SCENE_NAME / input_suffix



SCENE_ROOT = REPLICA_GT_ROOT / f"{SCENE_NAME}"
SPARSE_SUBDIR = Path(os.environ.get("REPLICA_SPARSE_SUBDIR", "sparse/0"))
IMAGES_DIR = SCENE_ROOT / "images"
DEPTH_DIR = SCENE_ROOT / "depth_images"
GT_COLMAP_DIR = SCENE_ROOT / SPARSE_SUBDIR



def resolve_gt_dense_ply(scene_root: Path) -> Optional[Path]:
    env_path = os.environ.get("REPLICA_GT_PLY")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    candidates = [
        scene_root / "gt_pcd_0.ply",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


GT_DENSE_PLY = resolve_gt_dense_ply(SCENE_ROOT)


# Get inference device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init model - This requires internet access or the huggingface hub cache to be pre-downloaded
# For Apache 2.0 license model, use "facebook/map-anything-apache"
model = MapAnything.from_pretrained("facebook/map-anything").to(device)

# COLMAP helpers ------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
SUPPORTED_DEPTH_EXTENSIONS: Tuple[str, ...] = (".png", ".tif", ".tiff", ".exr", ".npy", ".npz")


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


def camera_to_intrinsics(camera) -> np.ndarray:
    model = camera.model.upper()
    params = camera.params

    if model in {"PINHOLE", "OPENCV", "FULL_OPENCV", "RADIAL", "RADIAL_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
    elif model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        raise ValueError(f"Unsupported COLMAP camera model '{camera.model}'")

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return colmap_to_opencv_intrinsics(K)


def build_depth_index(depth_dir: Path) -> Dict[str, Path]:
    if not depth_dir.exists():
        return {}

    index: Dict[str, Path] = {}
    for path in depth_dir.glob("**/*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_DEPTH_EXTENSIONS:
            continue
        stem = path.stem
        index.setdefault(stem, path)
        if stem.endswith("_depth"):
            index.setdefault(stem[:-6], path)
        if stem.endswith("Depth"):
            index.setdefault(stem[:-5], path)
        if stem.endswith("_d"):
            index.setdefault(stem[:-2], path)
    return index


def find_depth_path(image_path: Path, depth_index: Dict[str, Path]) -> Optional[Path]:
    stem = image_path.stem
    candidates = [stem]
    if stem.endswith("_rgb"):
        candidates.append(stem[:-4])
        candidates.append(f"{stem[:-4]}_depth")
    candidates.append(f"{stem}_depth")
    candidates.append(stem.replace("_rgb", "_depth"))

    for candidate in candidates:
        path = depth_index.get(candidate)
        if path is not None:
            return path
    return None


def load_depth_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".png", ".tiff", ".tif", ".exr"}:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    elif suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if "depth" in data:
                depth = data["depth"]
            else:
                depth = data[data.files[0]]
    else:
        raise ValueError(f"Unsupported depth file extension: {path.suffix}")

    if depth is None:
        raise ValueError(f"Failed to load depth file: {path}")

    return depth


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
if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Replica images directory not found: {IMAGES_DIR}")

if not GT_COLMAP_DIR.exists():
    raise FileNotFoundError(f"Replica COLMAP sparse directory not found: {GT_COLMAP_DIR}")

gt_images = read_images_text(str(GT_COLMAP_DIR / "images.txt"))
gt_images_by_name = {img.name: img for img in gt_images.values()}

cameras_txt = GT_COLMAP_DIR / "cameras.txt"
cameras_bin = GT_COLMAP_DIR / "cameras.bin"
if cameras_txt.exists():
    gt_cameras = read_cameras_text(str(cameras_txt))
elif cameras_bin.exists():
    gt_cameras = read_cameras_binary(str(cameras_bin))
else:
    raise FileNotFoundError(
        f"Neither cameras.txt nor cameras.bin found in {GT_COLMAP_DIR}"
    )

depth_index = build_depth_index(DEPTH_DIR)
if not depth_index:
    print(f"Warning: no depth files found under {DEPTH_DIR}")

images = IMAGES_DIR
image_paths = list_image_paths(images)
views = load_images([str(p) for p in image_paths])

view_transforms: List[dict] = []
original_shapes: List[Tuple[int, int]] = []
gt_depths_resized: List[Optional[np.ndarray]] = []
gt_depths_raw: List[Optional[np.ndarray]] = []
gt_depth_paths: List[Optional[Path]] = []
missing_depth_images: List[str] = []
for idx, view in enumerate(views):
    resized_h, resized_w = map(int, np.squeeze(view["true_shape"]))
    original_shape = view.get("original_shape")
    if original_shape is not None:
        orig_h, orig_w = map(int, np.squeeze(original_shape))
    else:
        orig_h, orig_w = resized_h, resized_w

    original_shapes.append((orig_h, orig_w))

    image_path = image_paths[idx] if idx < len(image_paths) else Path(f"view_{idx:03d}.png")
    image_name = image_path.name

    gt_image = gt_images_by_name.get(image_name)
    if gt_image is None:
        raise KeyError(f"No COLMAP entry found for image '{image_name}'")

    gt_camera = gt_cameras.get(gt_image.camera_id)
    if gt_camera is None:
        raise KeyError(
            f"No camera parameters for camera_id={gt_image.camera_id} (image '{image_name}')"
        )

    intrinsics_full = camera_to_intrinsics(gt_camera)

    # Mirror the uniform resize + center crop done in load_images for consistent intrinsics.
    if orig_w <= 0 or orig_h <= 0:
        scale = 1.0
    else:
        scale = max(resized_w / orig_w, resized_h / orig_h)

    scale = float(scale) if scale > 0.0 else 1.0

    scaled_w = max(resized_w, int(round(orig_w * scale)))
    scaled_h = max(resized_h, int(round(orig_h * scale)))
    crop_left = max((scaled_w - resized_w) // 2, 0)
    crop_top = max((scaled_h - resized_h) // 2, 0)

    intrinsics_resized = intrinsics_full.copy()
    intrinsics_resized[0, 0] *= scale
    intrinsics_resized[1, 1] *= scale
    intrinsics_resized[0, 2] = intrinsics_resized[0, 2] * scale - crop_left
    intrinsics_resized[1, 2] = intrinsics_resized[1, 2] * scale - crop_top

    if input_camera:
        view["intrinsics"] = (
            torch.from_numpy(intrinsics_resized.astype(np.float32)).unsqueeze(0)
        )

    view_transforms.append(
        dict(
            scale=scale,
            crop_left=float(crop_left),
            crop_top=float(crop_top),
        )
    )

    depth_path = find_depth_path(image_path, depth_index) if depth_index else None
    if depth_path is not None:
        depth_raw = load_depth_file(depth_path)
        depth_raw_float = depth_raw.astype(np.float32)

        if (depth_raw.shape[1], depth_raw.shape[0]) != (scaled_w, scaled_h):
            depth_scaled = cv2.resize(
                depth_raw_float,
                (scaled_w, scaled_h),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            depth_scaled = depth_raw_float

        depth_cropped = depth_scaled[
            crop_top : crop_top + resized_h,
            crop_left : crop_left + resized_w,
        ]

        if depth_cropped.shape != (resized_h, resized_w):
            center_top = max((depth_scaled.shape[0] - resized_h) // 2, 0)
            center_left = max((depth_scaled.shape[1] - resized_w) // 2, 0)
            depth_cropped = depth_scaled[
                center_top : center_top + resized_h,
                center_left : center_left + resized_w,
            ]

        depth_cropped = np.ascontiguousarray(depth_cropped.astype(np.float32))
        depth_tensor = torch.from_numpy(depth_cropped).unsqueeze(0)
        if input_depth:
            view["depth_z"] = depth_tensor / 6553.5
            view["is_metric_scale"] = torch.tensor([True], dtype=torch.bool)
        # view["gt_depth_resized"] = depth_tensor.unsqueeze(-1)
        gt_depths_resized.append(depth_cropped.copy())
        gt_depths_raw.append(np.ascontiguousarray(depth_raw))
        gt_depth_paths.append(depth_path)
    else:
        missing_depth_images.append(image_name)
        gt_depths_resized.append(None)
        gt_depths_raw.append(None)
        gt_depth_paths.append(None)

    gt_qvec = gt_image.qvec.astype(np.float64)
    gt_tvec = gt_image.tvec.astype(np.float64)
    gt_R_cw = qvec2rotmat(gt_qvec)
    gt_R_wc = gt_R_cw.T
    gt_t_wc = -gt_R_wc @ gt_tvec

    gt_pose_cam2world = np.eye(4, dtype=np.float32)
    gt_pose_cam2world[:3, :3] = gt_R_wc.astype(np.float32)
    gt_pose_cam2world[:3, 3] = gt_t_wc.astype(np.float32)

    if input_pose:
        view["camera_poses"] = torch.from_numpy(gt_pose_cam2world[None, ...])

if missing_depth_images:
    preview = ", ".join(missing_depth_images[:5])
    print(
        f"Warning: missing depth maps for {len(missing_depth_images)} images. "
        f"Examples: {preview}"
    )

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
        transform = (
            view_transforms[saved_view_idx]
            if saved_view_idx < len(view_transforms)
            else None
        )
        if transform is not None:
            scale = transform.get("scale", 1.0)
            crop_left = transform.get("crop_left", 0.0)
            crop_top = transform.get("crop_top", 0.0)
        else:
            scale = 1.0
            crop_left = 0.0
            crop_top = 0.0

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

        gt_depth_resized = (
            gt_depths_resized[saved_view_idx]
            if saved_view_idx < len(gt_depths_resized)
            else None
        )
        if gt_depth_resized is not None:
            np.save(view_dir / "gt_depth.npy", gt_depth_resized.astype(np.float32))

            raw_depth = (
                gt_depths_raw[saved_view_idx]
                if saved_view_idx < len(gt_depths_raw)
                else None
            )
            if raw_depth is not None:
                np.save(view_dir / "gt_depth_raw.npy", raw_depth)
                if raw_depth.dtype in (np.uint8, np.uint16, np.uint32):
                    cv2.imwrite(str(view_dir / "gt_depth_raw.png"), raw_depth)

            depth_src_path = (
                gt_depth_paths[saved_view_idx]
                if saved_view_idx < len(gt_depth_paths)
                else None
            )
            if depth_src_path is not None:
                (view_dir / "gt_depth_source.txt").write_text(
                    str(depth_src_path), encoding="utf-8"
                )

            if gt_depth_resized.dtype in (np.float32, np.float64):
                depth_mm_gt = np.clip(
                    gt_depth_resized * DEPTH_SCALE_MM,
                    0.0,
                    float(np.iinfo(np.uint16).max),
                ).astype(np.uint16)
                cv2.imwrite(str(view_dir / "gt_depth_mm.png"), depth_mm_gt)

        intrinsics_rescaled = intrinsics_np[b].astype(np.float64).copy()
        # Undo the preprocessing crop/scale to express intrinsics in the original resolution.
        intrinsics_rescaled[0, 2] += crop_left
        intrinsics_rescaled[1, 2] += crop_top
        if scale > 0.0:
            inv_scale = 1.0 / scale
        else:
            inv_scale = 1.0
        intrinsics_rescaled[0, 0] *= inv_scale
        intrinsics_rescaled[1, 1] *= inv_scale
        intrinsics_rescaled[0, 2] *= inv_scale
        intrinsics_rescaled[1, 2] *= inv_scale

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
        if saved_view_idx < len(original_shapes):
            orig_h, orig_w = original_shapes[saved_view_idx]
        else:
            if image_view is not None:
                orig_h, orig_w = image_view.shape[:2]
            else:
                orig_h = depth_map.shape[0]
                orig_w = depth_map.shape[1]
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

if GT_DENSE_PLY is not None and GT_DENSE_PLY.exists():
    scale, rot, trans, aligned_pose_np, gt_pose_t, cameras_dict = align_predictions(
        estimated_root=ma_sparse_dir,
        gt_colmap_dir=str(GT_COLMAP_DIR),
        gt_ply=str(GT_DENSE_PLY),
        output_dir=align_dir,
        voxel_size=0.04,
        scale_trans = False
    )
    print(f"Alignment scale: {scale}")

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
else:
    print(
        "Skipping alignment step: ground-truth point cloud not found. "
        f"Checked path: {GT_DENSE_PLY}"
    )
