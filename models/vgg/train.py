import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from models.vgg.model import make_vgg16_model
from models.vgg.evaluate import Evaluator

class Trainer:
    def __init__(
        self,
        epochs=50,
        lr=1e-5,
        batch_size=16,
        train_mean=None,
        train_std=None,
        project="FIXED-vgg16-xray-regression",
        run_name=None,
        device=None,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
        seed=42,
    ):
        # -------------------------
        # Reproducibility
        # -------------------------
        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # -------------------------
        # Core components
        # -------------------------
        self.model = make_vgg16_model()
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.L1Loss()

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_mean = train_mean
        self.train_std = train_std

        self.project = project
        self.run_name = run_name

        self.model.to(self.device)
        os.makedirs("outputs", exist_ok=True)

        print(f"Using device: {self.device}")
        print(
            f"Hyperparameters | Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}"
        )

    # -------------------------
    # Dataloaders
    # -------------------------
    def create_dataloaders(self):
        g = torch.Generator().manual_seed(self.seed)

        def seed_worker(worker_id):
            np.random.seed(self.seed + worker_id)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        )

    # -------------------------
    # Training loop
    # -------------------------
    def train(self):
        self.create_dataloaders()

        # --- Evaluator ---
        evaluator = Evaluator(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
            train_mean=self.train_mean,
            train_std=self.train_std,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            output_path="outputs/best_model.pt",
            use_wandb=True,
            wandb_run=wandb,
        )

        wandb.init(
            project=self.project,
            name=self.run_name,
            config={
                "epochs": self.epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "model": "VGG16",
                "seed": self.seed,
            },
        )

        print("Beginning training...")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss_total = 0.0
            all_preds, all_labels = [], []

            for images, labels in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()

                train_loss_total += loss.item()
                all_preds.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())

            avg_train_loss = train_loss_total / len(self.train_loader)
            train_r = evaluator._pearson_corr(all_preds, all_labels)

            # crash-safe checkpoint
            torch.save(self.model.state_dict(), "outputs/last_model.pt")

            # --- Validation via Evaluator ---
            avg_val_loss, val_r = evaluator.validate(epoch)

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_mae": avg_train_loss,
                    "train_r": train_r,
                    "val_mae": avg_val_loss,
                    "val_r": val_r,
                }
            )

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train MAE: {avg_train_loss:.4f} | Train r: {train_r:.4f} | "
                f"Val MAE: {avg_val_loss:.4f} | Val r: {val_r:.4f}"
            )

        print("Training complete.")

        # --- Final test evaluation ---
        evaluator.evaluate()
