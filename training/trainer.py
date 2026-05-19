import torch
#from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        ema,
        config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.config = config

        self.scaler = GradScaler("cuda")
        self.device = config.device
        self.best_mean_dice = 0.0

    # ===============================
    # FAST VECTORISED DICE (NO BG)
    # ===============================
    def compute_classwise_dice(self, logits, targets):
        """
        Vectorized Dice computation.
        Excludes background (class 0).
        """

        num_classes = self.config.num_classes

        preds = torch.argmax(logits, dim=1)

        # One-hot encoding
        preds_onehot = torch.nn.functional.one_hot(
            preds, num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        targets_onehot = torch.nn.functional.one_hot(
            targets, num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        # Remove background
        preds_onehot = preds_onehot[:, 1:]
        targets_onehot = targets_onehot[:, 1:]

        dims = (0, 2, 3, 4)

        intersection = (preds_onehot * targets_onehot).sum(dim=dims)
        union = preds_onehot.sum(dim=dims) + targets_onehot.sum(dim=dims)

        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)

        return dice


    # ===============================
    # TRAIN
    # ===============================
    def train_epoch(self, epoch):

        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader)

        for img, mask in pbar:

            img = img.to(self.device, non_blocking=True).float()
            mask = mask.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                logits, aux_logits, node_feats, node_labels = self.model(img)

                loss, _ = self.criterion(
                    logits,
                    mask,
                    aux_logits,
                    node_feats,
                    node_labels,
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update()

            total_loss += loss.item()

            pbar.set_description(
                f"Epoch {epoch} | Loss {loss.item():.4f}"
            )

        self.scheduler.step()

        return total_loss / len(self.train_loader)

    # ===============================
    # VALIDATION (FAST)
    # ===============================
    def validate(self):

        self.model.eval()

        val_loss = 0
        dice_total = torch.zeros(self.config.num_classes - 1, device=self.device)

        with torch.no_grad():
            for images, masks in self.val_loader:

                images = images.to(self.device, non_blocking=True).float()
                masks = masks.to(self.device, non_blocking=True).long()

                with autocast("cuda"):
                    outputs, _, _, _ = self.model(images)
                    loss, _ = self.criterion(outputs, masks)

                val_loss += loss.item()

                dice_scores = self.compute_classwise_dice(outputs, masks)
                dice_total += dice_scores

        val_loss /= len(self.val_loader)
        class_dice_avg = (dice_total / len(self.val_loader)).tolist()
        mean_dice = sum(class_dice_avg) / len(class_dice_avg)

        return val_loss, class_dice_avg, mean_dice

    # ===============================
    # TRAIN LOOP
    # ===============================
    def fit(self):

        for epoch in range(self.config.epochs):

            train_loss = self.train_epoch(epoch)
            val_loss, class_dice, mean_dice = self.validate()

            print(f"\nEpoch [{epoch}/{self.config.epochs}]")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")

            # Print class Dice in single line
            dice_str = " | ".join(
                [f"C{i+1}: {d:.4f}" for i, d in enumerate(class_dice)]
            )
            print(f"Class Dice (no BG): {dice_str}")

            print(f"Mean Dice: {mean_dice:.4f}")
            #print(f"EMA Mean Dice: {ema_mean:.4f}")

            # Save ONLY best mean Dice model
            if mean_dice > self.best_mean_dice:
                self.best_mean_dice = mean_dice
                save_path = self.config.save_path
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print("Saved Best Model!")
