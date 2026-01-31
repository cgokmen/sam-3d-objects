import argparse
import json
import pathlib
import random
import sys

import numpy as np
from PIL import Image

sys.path.append("notebook")
from inference import Inference


def load_segment_images(room_dir: pathlib.Path, segment_name: str):
    image_dir = room_dir / "images"
    segmentation_file = room_dir / "segmentation3d" / "objects_to_segmentation_maps.json"
    if not segmentation_file.exists():
        raise FileNotFoundError(f"Missing association file: {segmentation_file}")

    with open(segmentation_file, "r") as f:
        appearances_by_segment = json.load(f)
    if segment_name not in appearances_by_segment:
        raise KeyError(f"Segment {segment_name} not found in {segmentation_file}")

    appearances = appearances_by_segment[segment_name]
    images = []
    filenames = []
    for img_fn, mask_indices in appearances.items():
        img_fn = pathlib.Path(img_fn)
        img_fullpath = image_dir / img_fn.name
        if not img_fullpath.exists():
            raise FileNotFoundError(f"Missing image file: {img_fullpath}")

        seg_fn = room_dir / "segmentations" / f"{img_fn.stem}.npz"
        segmentation_data = np.load(seg_fn)
        seg_masks = segmentation_data["masks"].astype(bool)
        if len(seg_masks) == 0:
            continue

        mask = np.sum(seg_masks[mask_indices], axis=0, dtype=bool)
        img = np.array(Image.open(img_fullpath).convert("RGBA"))
        img[~mask, 3] = 0
        images.append(img)
        filenames.append(img_fn.name)

    return images, filenames


def main():
    parser = argparse.ArgumentParser(description="Generate a 3D object from a room segment.")
    parser.add_argument("--room-dir", type=pathlib.Path, default="/checkpoint/clear/cgokmen/vid2room/RealEstate10K/vid_1vdXN7X4Af4/rooms/living_room_0", help="Room directory path")
    parser.add_argument("--segment", type=str, default="sofa-segment27-2", help="Segment name, e.g. bed-segment17-0")
    parser.add_argument("--config-tag", type=str, default="hf", help="Checkpoint tag")
    parser.add_argument("--max-images", type=int, default=8, help="Max number of images to use")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("sample.glb"))
    args = parser.parse_args()

    config_path = f"checkpoints/{args.config_tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    images, filenames = load_segment_images(args.room_dir, args.segment)
    if not images:
        raise RuntimeError(f"No images found for segment {args.segment} in {args.room_dir}")

    indices = list(range(len(images)))
    if len(images) > args.max_images:
        indices = list(np.linspace(0, len(images) - 1, args.max_images, dtype=int))
    images = [images[i] for i in indices]
    filenames = [filenames[i] for i in indices]

    output = inference(
        images,
        mask=None,
        # stage1_inference_steps=100,
        # stage2_inference_steps=100,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output["glb"].export(args.output)
    np.save(args.output.with_suffix(".npy"), output["coords"].cpu().numpy())


if __name__ == "__main__":
    main()
