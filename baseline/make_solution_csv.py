import argparse
import csv
from pathlib import Path

import numpy as np


def mask_to_rle(mask):
    rles = []
    for channel in range(mask.shape[2]):
        pixels = mask[:, :, channel].T.reshape(-1)
        pixels = np.concatenate(([0], pixels, [0]))
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        rles.append(" ".join(str(x) for x in runs))
    return ";".join(rles)


def parse_args():
    base = Path(__file__).resolve().parent / "outputs" / "simple"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--masks",
        type=str,
        default=str(base / "test_predictions_original.npy"),
    )
    parser.add_argument(
        "--names",
        type=str,
        default=str(base / "test_image_names.npy"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(base / "solution.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    masks_path = Path(args.masks)
    names_path = Path(args.names)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    masks = np.load(masks_path, allow_pickle=True)
    names = np.load(names_path, allow_pickle=True)

    if len(masks) != len(names):
        raise RuntimeError(
            f"Mismatch: {len(masks)} masks vs {len(names)} names ({masks_path}, {names_path})"
        )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "rle", "batch_number"])

        for i, (name, mask) in enumerate(zip(names, masks)):
            mask = np.asarray(mask)
            if mask.ndim != 3 or mask.shape[2] != 4:
                raise RuntimeError(
                    f"Mask at index {i} has shape {mask.shape}, expected [H, W, 4]"
                )
            mask = (mask > 0).astype(np.uint8)
            writer.writerow([str(name), mask_to_rle(mask), i])

    print(f"Saved {len(names)} rows to: {output_path}")


if __name__ == "__main__":
    main()
