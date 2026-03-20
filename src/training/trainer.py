"""
==============================================================================
Trainer — Handles the complete training and validation loop.
==============================================================================
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
import logging
import json

from ..evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates training, validation, checkpointing, and early stopping.

    Usage:
        >>> trainer = Trainer(model, criterion, optimizer, scheduler, device)
        >>> trainer.fit(train_loader, test_loader, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "models/checkpoints",
        log_dir: str = "outputs/logs",
        clip_grad_norm: float = 1.0,
        early_stopping_patience: int = 10,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.early_stopping_patience = early_stopping_patience

        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

        # Tracking
        self.best_auc = 0.0
        self.epochs_without_improvement = 0
        self.training_history = []

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {"mil": 0.0, "smoothness": 0.0, "sparsity": 0.0}
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            features = batch["features"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            # Forward pass
            output = self.model(features, mask)
            losses = self.criterion(output["clip_scores"], labels, mask)

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )

            self.optimizer.step()

            # Accumulate losses
            total_loss += losses["total"].item()
            for key in loss_components:
                loss_components[key] += losses[key].item()
            num_batches += 1

            # Log every 20 batches
            if (batch_idx + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f}"
                )

        # Average losses
        avg_losses = {
            "total": total_loss / max(num_batches, 1),
            **{k: v / max(num_batches, 1) for k, v in loss_components.items()},
        }

        return avg_losses

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Run evaluation on test set."""
        self.model.eval()

        all_video_scores = []
        all_labels = []
        all_clip_scores = []
        total_loss = 0.0
        num_batches = 0

        for batch in test_loader:
            features = batch["features"].to(self.device)
            mask = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            output = self.model(features, mask)
            losses = self.criterion(output["clip_scores"], labels, mask)

            total_loss += losses["total"].item()
            num_batches += 1

            all_video_scores.append(output["video_score"].cpu())
            all_labels.append(labels.cpu())
            all_clip_scores.append(output["clip_scores"].cpu())

        # Concatenate all predictions
        video_scores = torch.cat(all_video_scores).numpy()
        labels = torch.cat(all_labels).numpy()

        # Compute metrics
        metrics = compute_all_metrics(labels, video_scores)
        metrics["loss"] = total_loss / max(num_batches, 1)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 50,
    ) -> Dict:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader.
            test_loader:  Test data loader.
            epochs:       Number of training epochs.

        Returns:
            Dict with training history and best metrics.
        """
        logger.info("=" * 60)
        logger.info("Starting CausalVAD Training")
        logger.info(f"  Epochs:     {epochs}")
        logger.info(f"  Device:     {self.device}")
        logger.info(f"  Train size: {len(train_loader.dataset)}")
        logger.info(f"  Test size:  {len(test_loader.dataset)}")
        logger.info("=" * 60)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # ─── Train ───────────────────────────────────────────
            train_losses = self.train_one_epoch(train_loader, epoch)

            # ─── Evaluate ────────────────────────────────────────
            metrics = self.evaluate(test_loader)

            # ─── Learning Rate Step ──────────────────────────────
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            # ─── Logging ─────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Test AUC: {metrics['auc']:.4f} | "
                f"Test AP: {metrics['ap']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # TensorBoard
            self.writer.add_scalar("Loss/train_total", train_losses["total"], epoch)
            self.writer.add_scalar("Loss/mil", train_losses["mil"], epoch)
            self.writer.add_scalar("Loss/smoothness", train_losses["smoothness"], epoch)
            self.writer.add_scalar("Loss/sparsity", train_losses["sparsity"], epoch)
            self.writer.add_scalar("Metrics/AUC", metrics["auc"], epoch)
            self.writer.add_scalar("Metrics/AP", metrics["ap"], epoch)
            self.writer.add_scalar("Metrics/F1", metrics["f1"], epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            # ─── Save history ────────────────────────────────────
            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_losses["total"],
                "test_auc": metrics["auc"],
                "test_ap": metrics["ap"],
                "test_f1": metrics["f1"],
                "lr": current_lr,
            })

            # ─── Checkpointing ───────────────────────────────────
            if metrics["auc"] > self.best_auc:
                self.best_auc = metrics["auc"]
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, metrics, is_best=True)
                logger.info(f"  [BEST] New best AUC: {self.best_auc:.4f} - checkpoint saved")
            else:
                self.epochs_without_improvement += 1

            # Save latest checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, metrics, is_best=False)

            # ─── Early Stopping ──────────────────────────────────
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best AUC: {self.best_auc:.4f}")
        logger.info("=" * 60)

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.writer.close()

        return {
            "best_auc": self.best_auc,
            "history": self.training_history,
            "total_time_minutes": total_time / 60,
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_auc": self.best_auc,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        logger.info(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_auc = checkpoint.get("best_auc", 0.0)
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
