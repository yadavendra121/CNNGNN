import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


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

        self.scaler = GradScaler()
        self.device = config.device
        self.best_class_dice = [0.0] * self.config.num_classes
        self.best_mean_dice = 0.0

    # DC of each classes
    def compute_classwise_dice(self, preds, targets):
        num_classes = self.config.num_classes
        dice_scores = []

        preds = torch.argmax(preds, dim=1)

        for cls in range(num_classes):
            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()

            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            dice_scores.append(dice.item())

        return dice_scores

    # ===============================
    # TRAIN ONE EPOCH
    # ===============================
    def train_epoch(self, epoch):

        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader)

        for img, mask in pbar:

            img = img.to(self.device).float()
            mask = mask.to(self.device).long()

            self.optimizer.zero_grad()

            with autocast():
                logits, aux_logits, node_feats, node_labels = self.model(img)

                loss, loss_dict = self.criterion(
                    logits,
                    mask,
                    aux_logits,
                    node_feats,
                    node_labels,
                )

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 12
            )

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
    # VALIDATION
    # ===============================
    def validate(self):
        self.model.eval()

        val_loss = 0
        class_dice_total = torch.zeros(self.config.num_classes)
        ema_dice_total = torch.zeros(self.config.num_classes)

        with torch.no_grad():
            for images, masks in self.val_loader:

                images = images.to(self.device).float()
                masks = masks.to(self.device).long()

                # ======================
                # Normal Model
                # ======================
                outputs, _, _, _ = self.model(images)

                loss, _ = self.criterion(outputs, masks)
                
                val_loss += loss.item()

                dice_scores = self.compute_classwise_dice(outputs, masks)
                class_dice_total += torch.tensor(dice_scores, device=class_dice_total.device)

        # ======================
        # EMA Evaluation
        # ======================
        self.ema.apply_shadow()

        with torch.no_grad():
            for images, masks in self.val_loader:

                images = images.to(self.device).float()
                masks = masks.to(self.device).long()

                outputs, _, _, _ = self.model(images)

                dice_scores = self.compute_classwise_dice(outputs, masks)
                #ema_dice_total += torch.tensor(dice_scores)
                ema_dice_total += torch.tensor(dice_scores, device=ema_dice_total.device)

        self.ema.restore()

        val_loss /= len(self.val_loader)
        class_dice_avg = (class_dice_total / len(self.val_loader)).tolist()
        ema_dice_avg = (ema_dice_total / len(self.val_loader)).tolist()

        return val_loss, class_dice_avg, ema_dice_avg


    # ===============================
    # TRAIN LOOP
    # ===============================
    def fit(self):

        best_dice = 0.0  # change to dice-based saving

        for epoch in range(self.config.epochs):

            train_loss = self.train_epoch(epoch)
            val_loss, class_dice, ema_dice = self.validate()

            mean_dice = sum(class_dice) / len(class_dice)

            print(f"\nEpoch [{epoch+1}/{self.config.epochs}]")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")

            print("Class-wise Dice:")
            for i, d in enumerate(class_dice):
                print(f"  Class {i}: {d:.4f}")

            print("EMA Class-wise Dice:")
            for i, d in enumerate(ema_dice):
                print(f"  Class {i}: {d:.4f}")

            print(f"Mean Dice: {mean_dice:.4f}")

            # Save best model based on Dice instead of loss
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(self.model.state_dict(), "best_model.pth")
                print("Saved Best Model!")
            # ============================
            # Save Per-Class Best Models
            # ============================
            for i, dice_score in enumerate(class_dice):

                if dice_score > self.best_class_dice[i]:
                    self.best_class_dice[i] = dice_score

                    save_path = f"best_model_class_{i}.pth"
                    torch.save(self.model.state_dict(), save_path)

                    print(f"Saved Best Model for Class {i} | Dice: {dice_score:.4f}")