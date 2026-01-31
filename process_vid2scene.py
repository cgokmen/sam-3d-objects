import json
import os
import pathlib
import random
import sys
import traceback
import hashlib

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

sys.path.append("notebook")
from inference import Inference

STRUCTURE_CATEGORIES = {
    "wall",
    "floor",
    "ceiling",
    "window",
    "door",
    "doorway",
    "stairs",
}


OPENCV_TO_PYTORCH = R.from_euler("z", [180], degrees=True)


def process_object(scene_dir, segment, inference, use_gt_pointmap=False):
    print(f"Processing {scene_dir} segment {segment}")

    # Find where to put the mesh
    mesh_dir = scene_dir / (
        "obj_meshes_v9_pointmap" if use_gt_pointmap else "obj_meshes_v9"
    )
    mesh_path = mesh_dir / f"{segment}.glb"
    transform_path = mesh_dir / f"{segment}.json"
    if mesh_path.exists() and transform_path.exists():
        return

    # Get all the image files
    image_dir = scene_dir / "images"
    assert image_dir.exists(), f"Image directory {image_dir} does not exist."

    # Load the appearances
    segmentation_file = scene_dir / "segmentation3d" / f"objects_to_segmentation_maps.json"
    with open(segmentation_file, "r") as f:
        appearances_by_segment = json.load(f)
    appearances = appearances_by_segment[segment]

    images = []
    filenames = []
    pointmaps = []

    cameras_and_points = np.load(scene_dir / "sparse_pi3x/0/cameras_and_points.npz")
    cameras_and_points_filenames = list(cameras_and_points["filenames"])

    for img_fn, mask_indices in appearances.items():
        # Load the image
        img_fn = pathlib.Path(img_fn)
        img_fullpath = image_dir / img_fn.name
        img = np.array(Image.open(img_fullpath).convert("RGBA"))

        seg_fn = scene_dir / "segmentations" / f"{img_fn.stem}.npz"
        segmentation_data = np.load(seg_fn)["masks"]
        mask = np.sum(segmentation_data[mask_indices], axis=0, dtype=bool)

        pointmap = cameras_and_points["local_points"][
            cameras_and_points_filenames.index(img_fn.name)
        ]
        pointmap = OPENCV_TO_PYTORCH.apply(pointmap.reshape(-1, 3)).reshape(
            pointmap.shape
        )

        img[~mask, 3] = 0  # Set alpha to 0 where mask is False
        images.append(img)
        filenames.append(img_fn.name)
        pointmaps.append(pointmap)

    # Do a filter and shuffle.
    indices = list(range(len(images)))
    if len(images) > 8:
        # Get 20 equally spaced images by index
        indices = list(np.linspace(0, len(images) - 1, 8, dtype=int))
    # random.shuffle(indices)
    images = [images[i] for i in indices]
    filenames = [filenames[i] for i in indices]
    pointmaps = [pointmaps[i] for i in indices] if use_gt_pointmap else None

    # Process the object.
    print(f"Generating object from {len(images)} images.")

    # Do inference.
    output = inference(
        images,
        mask=None,
        pointmap=pointmaps,
        # stage1_inference_steps=100,
        # stage2_inference_steps=100,
    )

    # Save the mesh.
    mesh_dir.mkdir(parents=True, exist_ok=True)
    output["glb"].export(mesh_path)

    # Save the transform
    pre_pose = output.get("pre_postprocess_pose", None)
    if pre_pose is None:
        pre_translation = output["translation"]
        pre_rotation = output["rotation"]
        pre_scale = output["scale"]
    else:
        pre_translation = pre_pose["translation"]
        pre_rotation = pre_pose["rotation"]
        pre_scale = pre_pose["scale"]

    transform = {
        filenames[i]: {
            "pre_postprocess": {
                "translation": pre_translation[i].cpu().numpy().tolist(),
                "rotation": pre_rotation[i].cpu().numpy().tolist(),
                "scale": pre_scale[i].cpu().numpy().tolist(),
            },
            "post_postprocess": {
                "translation": output["translation"][i].cpu().numpy().tolist(),
                "rotation": output["rotation"][i].cpu().numpy().tolist(),
                "scale": output["scale"][i].cpu().numpy().tolist(),
            },
        }
        for i in range(len(filenames))
    }
    transform_path.write_text(json.dumps(transform))

def main():
    config_path = f"checkpoints/hf/pipeline.yaml"
    inference = Inference(config_path, compile=True)
    dataset_root = sys.argv[1]
    task_id = int(sys.argv[2])  # 0-indexed
    total_jobs = int(sys.argv[3])
    use_gt_pointmap = True

    print(f"Finding rooms...")
    dataset_root = pathlib.Path(dataset_root)
    assert dataset_root.is_dir(), f"Error: {dataset_root} is not a directory"
    room_roots = [
        success.parent
        for success in dataset_root.glob("*/rooms/*/association.success")
    ]
    room_roots = sorted(room_roots)
    num_rooms = len(room_roots)
    print(f"Found {num_rooms} rooms")

    all_objects = []
    for i, room_root in enumerate(room_roots):
        # Get all segments
        segmentation_file = room_root / "segmentation3d" / f"objects_to_segmentation_maps.json"
        with open(segmentation_file, "r") as f:
            appearances_by_segment = json.load(f)
        segments = {
            k
            for k in appearances_by_segment
            if k.rsplit("-", 2)[0] not in STRUCTURE_CATEGORIES #  and k == "sofa-segment27-2"
        }

        all_objects.extend([(room_root, segment) for segment in segments])

    this_job_objects = [
        (room_root, segment)
        for room_root, segment in all_objects
        if int(
            hashlib.md5((str(room_root) + segment + "potato").encode()).hexdigest(), 16
        )
        % total_jobs
        == task_id
    ]
    num_objects = len(this_job_objects)
    print(f"Processing {num_objects} objects in job {task_id}/{total_jobs}")
    for scene_dir, segment in tqdm(this_job_objects):
        try:
            process_object(
                scene_dir,
                segment,
                inference,
                use_gt_pointmap=use_gt_pointmap,
            )
        except Exception as e:
            print(f"Error processing {scene_dir}, {segment}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()

