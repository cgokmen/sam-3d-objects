import json
import pathlib
import random
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

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

    cameras_and_points = np.load(scene_dir / "sparse/0/cameras_and_points.npz")
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
    if len(images) > 20:
        # Get 20 equally spaced images by index
        indices = list(np.linspace(0, len(images) - 1, 20, dtype=int))
    random.shuffle(indices)
    images = [images[i] for i in indices]
    filenames = [filenames[i] for i in indices]
    pointmaps = [pointmaps[i] for i in indices]

    # Process the object.
    print(f"Generating object from {len(images)} images.")

    # Find where to put the mesh
    mesh_dir = scene_dir / "obj_meshes2"
    mesh_path = mesh_dir / f"{segment}.glb"
    transform_path = mesh_dir / f"{segment}.json"
    if mesh_path.exists() and transform_path.exists():
        return

    # Do inference.
    output = inference(
        images,
        mask=None,
        stage1_inference_steps=25,
        stage2_inference_steps=25,
    )

    breakpoint()

    # Save the mesh.
    mesh_dir.mkdir(parents=True, exist_ok=True)
    output["glb"].export(mesh_path)

    # Save the transform
    transform = {
        filenames[i]: {
            "translation": output["translation"][i].cpu().numpy().tolist(),
            "rotation": output["rotation"][i].cpu().numpy().tolist(),
            "scale": output["scale"][i].cpu().numpy().tolist(),
        }
        for i in range(len(filenames))
    }
    transform_path.write_text(json.dumps(transform))


def main():
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)
    scene_dir = pathlib.Path(
        "/fsx-siro/cgokmen/vid2scene/val/41-Blackwell-Crt-179017109/rooms/living_room_0"
    )
    segment_name = "segment17-lamp-0"

    process_object(scene_dir, segment_name, inference)


if __name__ == "__main__":
    main()

