#!/usr/bin/env python3
"""Train 2D U-Net for BraTS tumor segmentation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset import BraTSDataset, split_train_val
from src.losses import DiceCrossEntropyLoss
from src.metrics import DEFAULT_CLASS_NAMES, compute_dice_summary
from src.model_architecture import build_model, count_trainable_parameters


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a 2D U-Net on BraTS slices.")
	parser.add_argument("--data-dir", type=Path, default=Path("./data"))
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-5)
	parser.add_argument("--val-ratio", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--target-size", type=int, default=128)
	parser.add_argument("--base-channels", type=int, default=32)
	parser.add_argument("--model", type=str, default="unet", choices=["unet", "resunet"])
	parser.add_argument("--save-dir", type=Path, default=Path("./models"))
	parser.add_argument("--patience", type=int, default=12)
	parser.add_argument("--min-delta", type=float, default=1e-4)
	parser.add_argument("--lr-patience", type=int, default=4)
	parser.add_argument("--lr-factor", type=float, default=0.5)
	parser.add_argument("--history-csv", type=Path, default=Path("./models/training_history.csv"))
	return parser.parse_args()


def set_seed(seed: int) -> None:
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def remap_brats_labels(mask: torch.Tensor) -> torch.Tensor:
	"""Remap BraTS mask labels {0,1,2,4} -> {0,1,2,3}."""

	remapped = mask.clone().long()
	remapped[remapped == 4] = 3
	return remapped


def train_one_epoch(
	model: torch.nn.Module,
	loader: DataLoader,
	criterion: DiceCrossEntropyLoss,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
) -> float:
	model.train()
	running_loss = 0.0

	for images, masks in loader:
		images = images.to(device)
		masks = remap_brats_labels(masks.squeeze(1).to(device))

		optimizer.zero_grad()
		logits = model(images)
		loss = criterion(logits, masks)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	return running_loss / max(1, len(loader))


@torch.no_grad()
def validate(
	model: torch.nn.Module,
	loader: DataLoader,
	criterion: DiceCrossEntropyLoss,
	device: torch.device,
	num_classes: int,
) -> tuple[float, float, dict[str, float]]:
	model.eval()
	running_loss = 0.0
	running_dice = 0.0
	running_class_dice = {name: 0.0 for name in DEFAULT_CLASS_NAMES.values()}

	for images, masks in loader:
		images = images.to(device)
		masks = remap_brats_labels(masks.squeeze(1).to(device))

		logits = model(images)
		loss = criterion(logits, masks)
		mean_dice, per_class = compute_dice_summary(logits, masks, num_classes)

		running_loss += loss.item()
		running_dice += mean_dice
		for name, value in per_class.items():
			if name in running_class_dice:
				running_class_dice[name] += value

	n_batches = max(1, len(loader))
	for name in running_class_dice:
		running_class_dice[name] /= n_batches

	return running_loss / n_batches, running_dice / n_batches, running_class_dice


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	dataset_eval = BraTSDataset(
		data_dir=args.data_dir,
		target_size=args.target_size,
		augment=False,
		cache=True,
	)
	train_split_eval, val_split_eval = split_train_val(
		dataset_eval,
		val_ratio=args.val_ratio,
		seed=args.seed,
		split_by_patient=True,
	)

	dataset_train = BraTSDataset(
		data_dir=args.data_dir,
		target_size=args.target_size,
		augment=True,
		cache=False,
	)

	train_set = Subset(dataset_train, train_split_eval.indices)
	val_set = Subset(dataset_eval, val_split_eval.indices)

	train_loader = DataLoader(
		train_set,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	val_loader = DataLoader(
		val_set,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
	)

	num_classes = 4
	model = build_model(
		args.model,
		in_channels=4,
		num_classes=num_classes,
		base_channels=args.base_channels,
	).to(device)
	criterion = DiceCrossEntropyLoss(alpha=0.5, beta=0.5)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="max",
		factor=args.lr_factor,
		patience=args.lr_patience,
	)

	args.save_dir.mkdir(parents=True, exist_ok=True)
	best_path = args.save_dir / f"best_{args.model}.pt"

	print(f"Model: {args.model}")
	print(f"Model parameters: {count_trainable_parameters(model):,}")
	print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

	best_val_dice = -1.0
	epochs_without_improvement = 0
	history_rows: list[dict[str, float | int]] = []
	for epoch in range(1, args.epochs + 1):
		train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_dice, val_class_dice = validate(model, val_loader, criterion, device, num_classes)
		scheduler.step(val_dice)

		current_lr = optimizer.param_groups[0]["lr"]

		class_line = " | ".join(
			f"{k}={v:.4f}" for k, v in val_class_dice.items() if k != "background"
		)

		print(
			f"Epoch {epoch:03d}/{args.epochs} | "
			f"lr={current_lr:.6f} | train_loss={train_loss:.4f} | "
			f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | {class_line}"
		)

		history_rows.append(
			{
				"epoch": epoch,
				"lr": float(current_lr),
				"train_loss": float(train_loss),
				"val_loss": float(val_loss),
				"val_dice_mean": float(val_dice),
				"val_dice_necrotic_non_enhancing": float(val_class_dice.get("necrotic_non_enhancing", 0.0)),
				"val_dice_edema": float(val_class_dice.get("edema", 0.0)),
				"val_dice_enhancing": float(val_class_dice.get("enhancing", 0.0)),
			}
		)

		if val_dice > best_val_dice:
			best_val_dice = val_dice
			epochs_without_improvement = 0
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"val_dice": val_dice,
					"args": vars(args),
				},
				best_path,
			)
			print(f"Saved new best model to: {best_path}")
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= args.patience:
				print(
					"Early stopping triggered: "
					f"no validation Dice improvement in {args.patience} epochs."
				)
				break

	args.history_csv.parent.mkdir(parents=True, exist_ok=True)
	with args.history_csv.open("w", newline="", encoding="utf-8") as f:
		fieldnames = [
			"epoch",
			"lr",
			"train_loss",
			"val_loss",
			"val_dice_mean",
			"val_dice_necrotic_non_enhancing",
			"val_dice_edema",
			"val_dice_enhancing",
		]
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(history_rows)
	print(f"Saved training history to: {args.history_csv}")

	print(f"Training complete. Best validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
	main()
