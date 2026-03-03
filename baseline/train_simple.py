#!/usr/bin/env python3

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


NUM_CLASSES = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_image(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def read_mask(path):
    mask = np.load(path)
    if mask.ndim == 2:
        mask = mask[..., None]
    mask = mask[..., :NUM_CLASSES]
    return (mask > 0.5).astype(np.float32)


def get_train_pairs(data_dir):
    images = sorted((data_dir / "train" / "images").glob("*.jpg"))
    masks = sorted((data_dir / "train" / "masks").glob("*.npy"))

    if len(images) != len(masks):
        raise RuntimeError(
            f"Mismatched train files: {len(images)} images vs {len(masks)} masks"
        )

    mask_map = {p.stem: p for p in masks}
    out_images = []
    out_masks = []

    for image_path in images:
        mask_path = mask_map.get(image_path.stem)
        if mask_path is None:
            raise RuntimeError(f"Mask not found for image: {image_path.name}")
        out_images.append(image_path)
        out_masks.append(mask_path)

    return out_images, out_masks


def train_transforms(image_size):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(transpose_mask=True),
        ],
        is_check_shapes=False,
    )


def eval_transforms(image_size):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(transpose_mask=True),
        ],
        is_check_shapes=False,
    )


class KitchenDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms):
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths) if mask_paths is not None else None
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = read_image(self.image_paths[index])

        if self.mask_paths is None:
            transformed = self.transforms(image=image)
            x = transformed["image"].float()
            return x, str(self.image_paths[index])

        mask = read_mask(self.mask_paths[index])
        transformed = self.transforms(image=image, mask=mask)
        x = transformed["image"].float()
        y = transformed["mask"].float()
        return x, y


def dice_channels(probs, targets, thresholds=0.5):
    eps = 1e-8
    if np.isscalar(thresholds):
        thresholds = [float(thresholds)] * probs.shape[1]

    threshold_tensor = torch.as_tensor(
        thresholds,
        dtype=probs.dtype,
        device=probs.device,
    ).view(1, -1, 1, 1)

    pred = (probs > threshold_tensor).float()
    truth = (targets > 0.5).float()

    intersection = (pred * truth).sum(dim=(0, 2, 3))
    union = pred.sum(dim=(0, 2, 3)) + truth.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().item())


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, -1, 1, 1))
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

    def forward(self, logits, targets):
        return 0.6 * self.bce(logits, targets) + 0.4 * self.dice(logits, targets)


def compute_pos_weight(mask_paths):
    positive = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_pixels = 0.0

    for path in mask_paths:
        mask = read_mask(path)
        positive += mask.reshape(-1, NUM_CLASSES).sum(axis=0)
        total_pixels += mask.shape[0] * mask.shape[1]

    negative = total_pixels - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1.0), 1.0, 30.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_sample_weights(mask_paths):
    presence = []

    for path in mask_paths:
        mask = read_mask(path)
        exists = (mask.reshape(-1, NUM_CLASSES).sum(axis=0) > 0).astype(np.float32)
        presence.append(exists)

    presence = np.asarray(presence, dtype=np.float32)
    freq = presence.sum(axis=0)
    inv = np.where(freq > 0, len(mask_paths) / np.maximum(freq, 1.0), 0.0)

    if np.any(inv > 0):
        inv = inv / np.mean(inv[inv > 0])

    weights = 1.0 + (presence * inv.reshape(1, -1)).sum(axis=1)
    weights = np.clip(weights, 1.0, 6.0)
    return torch.tensor(weights, dtype=torch.double)


def build_stratify_labels(mask_paths):
    labels = []

    for path in mask_paths:
        mask = read_mask(path)
        exists = (mask.reshape(-1, NUM_CLASSES).sum(axis=0) > 0).astype(np.int32)
        label = int((exists * (2 ** np.arange(NUM_CLASSES))).sum())
        labels.append(label)

    return np.asarray(labels, dtype=np.int32)


def optimize_thresholds(probs, targets, t_min, t_max, t_steps):
    grid = np.linspace(t_min, t_max, t_steps)
    best_thresholds = []

    for class_idx in range(NUM_CLASSES):
        class_probs = probs[:, class_idx : class_idx + 1]
        class_targets = targets[:, class_idx : class_idx + 1]

        best_t = 0.5
        best_score = -1.0

        for t in grid:
            score = dice_channels(class_probs, class_targets, [float(t)])
            if score > best_score:
                best_score = score
                best_t = float(t)

        best_thresholds.append(best_t)

    score = dice_channels(probs, targets, best_thresholds)
    return best_thresholds, score


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, amp):
    model.train()
    total_loss = 0.0
    use_autocast = amp and device.type == "cuda"

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        context = (
            torch.autocast(device_type="cuda", enabled=True)
            if use_autocast
            else nullcontext()
        )
        with context:
            logits = model(images)
            loss = loss_fn(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def validate(model, loader, loss_fn, device, amp):
    model.eval()
    losses = []
    all_probs = []
    all_targets = []
    use_autocast = amp and device.type == "cuda"

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            context = (
                torch.autocast(device_type="cuda", enabled=True)
                if use_autocast
                else nullcontext()
            )
            with context:
                logits = model(images)
                loss = loss_fn(logits, targets)
                probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return float(np.mean(losses)), probs, targets


def predict_test(model, loader, device, amp, tta):
    model.eval()
    all_names = []
    all_probs = []
    use_autocast = amp and device.type == "cuda"

    with torch.no_grad():
        for images, names in loader:
            images = images.to(device, non_blocking=True)

            context = (
                torch.autocast(device_type="cuda", enabled=True)
                if use_autocast
                else nullcontext()
            )
            with context:
                logits = model(images)
                probs = torch.sigmoid(logits)
                if tta:
                    flipped_images = torch.flip(images, dims=[3])
                    flipped_probs = torch.sigmoid(model(flipped_images))
                    probs = 0.5 * (probs + torch.flip(flipped_probs, dims=[3]))

            all_names.extend(list(names))
            all_probs.append(probs.cpu().numpy())

    return all_names, np.concatenate(all_probs, axis=0)


def create_original_size_masks(probs, image_paths, thresholds):
    thr = np.asarray(thresholds, dtype=np.float32).reshape(1, NUM_CLASSES, 1, 1)
    binary = (probs > thr).astype(np.float32)
    output = []

    for i, image_path in enumerate(image_paths):
        with Image.open(image_path) as img:
            h = img.height
            w = img.width

        tensor = torch.from_numpy(binary[i : i + 1])
        resized = F.interpolate(tensor, size=(h, w), mode="nearest")[0]
        output.append(resized.permute(1, 2, 0).numpy().astype(np.uint8))

    return np.asarray(output, dtype=object)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Simple and stable segmentation training")

    parser.add_argument("--data-dir", type=str, default=str(script_dir / "data"))
    parser.add_argument("--output-dir", type=str, default=str(script_dir / "outputs" / "simple"))
    parser.add_argument("--arch", type=str, default="unetpp", choices=["unet", "unetpp", "fpn"])
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--threshold-min", type=float, default=0.2)
    parser.add_argument("--threshold-max", type=float, default=0.8)
    parser.add_argument("--threshold-steps", type=int, default=13)

    parser.add_argument("--oversample", action="store_true", default=True)
    parser.add_argument("--no-oversample", action="store_false", dest="oversample")

    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", action="store_false", dest="tta")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")

    if args.amp and device.type != "cuda":
        print("AMP works only on CUDA, so AMP is disabled")
        args.amp = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_images, train_masks = get_train_pairs(data_dir)
    test_images = sorted((data_dir / "test" / "images").glob("*.jpg"))

    if len(test_images) == 0:
        raise RuntimeError(f"No test images found in: {data_dir / 'test' / 'images'}")

    stratify_labels = build_stratify_labels(train_masks)
    _, counts = np.unique(stratify_labels, return_counts=True)
    stratify = stratify_labels if counts.min() >= 2 else None

    train_idx, val_idx = train_test_split(
        np.arange(len(train_images)),
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
        stratify=stratify,
    )

    tr_images = [train_images[i] for i in train_idx]
    tr_masks = [train_masks[i] for i in train_idx]
    va_images = [train_images[i] for i in val_idx]
    va_masks = [train_masks[i] for i in val_idx]

    train_ds = KitchenDataset(tr_images, tr_masks, train_transforms(args.image_size))
    val_ds = KitchenDataset(va_images, va_masks, eval_transforms(args.image_size))
    test_ds = KitchenDataset(test_images, None, eval_transforms(args.image_size))

    sampler = None
    if args.oversample:
        sample_weights = compute_sample_weights(tr_masks)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_map = {
        "unet": smp.Unet,
        "unetpp": smp.UnetPlusPlus,
        "fpn": smp.FPN,
    }
    model = model_map[args.arch](
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    ).to(device)

    pos_weight = compute_pos_weight(tr_masks).to(device)
    loss_fn = BCEDiceLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.2,
    )
    scaler = torch.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_dice = -1.0
    best_epoch = -1
    best_thresholds = [0.5] * NUM_CLASSES
    bad_epochs = 0
    best_model_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            scaler,
            args.amp,
        )
        scheduler.step()

        val_loss, val_probs, val_targets = validate(
            model,
            val_loader,
            loss_fn,
            device,
            args.amp,
        )
        val_thresholds, val_dice = optimize_thresholds(
            val_probs,
            val_targets,
            args.threshold_min,
            args.threshold_max,
            args.threshold_steps,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | thr={np.round(val_thresholds, 2).tolist()}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            best_thresholds = [float(x) for x in val_thresholds]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "thresholds": best_thresholds,
                    "best_dice": best_dice,
                    "best_epoch": best_epoch,
                },
                best_model_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= args.early_stopping:
            print(f"Early stopping at epoch {epoch}. Best dice={best_dice:.4f}")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    best_thresholds = checkpoint["thresholds"]

    print(
        f"Loaded best model from epoch {best_epoch}, "
        f"val_dice={best_dice:.4f}, thresholds={np.round(best_thresholds, 2).tolist()}"
    )

    test_names, test_probs = predict_test(
        model,
        test_loader,
        device,
        args.amp,
        args.tta,
    )

    expected_names = [str(path) for path in test_images]
    if test_names != expected_names:
        raise RuntimeError("Test image order mismatch")

    thresholds = np.asarray(best_thresholds, dtype=np.float32).reshape(1, NUM_CLASSES, 1, 1)
    test_masks_resized = np.transpose((test_probs > thresholds).astype(np.uint8), (0, 2, 3, 1))
    test_masks_original = create_original_size_masks(test_probs, test_images, best_thresholds)

    np.save(output_dir / "test_predictions.npy", test_masks_resized)
    np.save(output_dir / "test_predictions_original.npy", test_masks_original, allow_pickle=True)
    np.save(output_dir / "test_probabilities.npy", test_probs.astype(np.float16))
    np.save(output_dir / "test_image_names.npy", np.array([p.name for p in test_images], dtype=object))

    report = {
        "best_val_dice": float(best_dice),
        "best_epoch": int(best_epoch),
        "thresholds": [float(x) for x in best_thresholds],
        "config": vars(args),
        "outputs": {
            "test_predictions_resized": str(output_dir / "test_predictions.npy"),
            "test_predictions_original": str(output_dir / "test_predictions_original.npy"),
            "test_probabilities": str(output_dir / "test_probabilities.npy"),
            "test_image_names": str(output_dir / "test_image_names.npy"),
        },
    }

    with (output_dir / "training_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
