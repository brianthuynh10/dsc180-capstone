import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

class Evaluator:
    def __init__(
        self,
        model,
        criterion,
        device,
        train_mean,
        train_std,
        val_loader=None,
        test_loader=None,
        output_path="outputs/best_model.pt",
        use_wandb=False,
        wandb_run=None,
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.train_mean = train_mean
        self.train_std = train_std
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_path = output_path
        self.use_wandb = use_wandb
        self.wandb = wandb_run

        self.best_val_loss = float("inf")

    # -------------------------
    # Core utilities
    # -------------------------
    @staticmethod
    def _pearson_corr(preds, labels):
        preds = torch.cat(preds).numpy().flatten()
        labels = torch.cat(labels).numpy().flatten()
        return pearsonr(labels, preds)[0]

    def _destandardize(self, preds, labels):
        preds = preds * self.train_std + self.train_mean
        labels = labels * self.train_std + self.train_mean

        preds_real = 10 ** (preds - 1)
        labels_real = 10 ** (labels - 1)

        return preds_real, labels_real

    def _make_scatter(self, labels_real, preds_real, title):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(labels_real, preds_real, alpha=0.4, s=20, label="Predictions")

        # regression line in log space
        log_true = np.log10(labels_real + 1e-8).reshape(-1, 1)
        log_pred = np.log10(preds_real + 1e-8)
        reg = LinearRegression().fit(log_true, log_pred)

        x_range = np.linspace(log_true.min(), log_true.max(), 100).reshape(-1, 1)
        ax.plot(
            10 ** x_range,
            10 ** reg.predict(x_range),
            color="red",
            linewidth=2,
            label="Regression line",
        )

        lims = [
            min(labels_real.min(), preds_real.min()),
            max(labels_real.max(), preds_real.max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Actual NT-proBNP")
        ax.set_ylabel("Predicted NT-proBNP")
        ax.set_title(title)

        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.xaxis.set_major_formatter(
            ticker.LogFormatter(base=10, labelOnlyBase=True)
        )
        ax.yaxis.set_major_formatter(
            ticker.LogFormatter(base=10, labelOnlyBase=True)
        )

        ax.legend(frameon=False)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()

        return fig

    # -------------------------
    # Validation
    # -------------------------
    def validate(self, epoch=None):
        assert self.val_loader is not None, "Validation loader not provided"

        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels.view(-1, 1))
                val_loss += loss.item()

                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = val_loss / len(self.val_loader)
        val_r = self._pearson_corr(all_preds, all_labels)

        preds_std = torch.cat(all_preds).numpy().flatten()
        labels_std = torch.cat(all_labels).numpy().flatten()
        preds_real, labels_real = self._destandardize(preds_std, labels_std)

        fig = self._make_scatter(
            labels_real,
            preds_real,
            title=f"Validation: Predicted vs Actual NT-proBNP (r={val_r:.3f})",
        )

        if self.use_wandb:
            self.wandb.log(
                {
                    "val_loss": avg_val_loss,
                    "val_r": val_r,
                    "val_scatter": self.wandb.Image(fig),
                    "epoch": epoch,
                }
            )

        self._save_best_model(avg_val_loss)
        plt.close(fig)

        return avg_val_loss, val_r

    # -------------------------
    # Test evaluation
    # -------------------------
    def evaluate(self):
        assert self.test_loader is not None, "Test loader not provided"

        print("🔍 Evaluating best model on test set...")
        self.model.load_state_dict(
            torch.load(self.output_path, map_location=self.device)
        )
        self.model.eval()

        test_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels.view(-1, 1))
                test_loss += loss.item()

                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        avg_test_loss = test_loss / len(self.test_loader)
        test_r = self._pearson_corr(all_preds, all_labels)

        preds_std = torch.cat(all_preds).numpy().flatten()
        labels_std = torch.cat(all_labels).numpy().flatten()
        preds_real, labels_real = self._destandardize(preds_std, labels_std)

        fig = self._make_scatter(
            labels_real,
            preds_real,
            title=f"Test: Predicted vs Actual NT-proBNP (r={test_r:.3f})",
        )

        if self.use_wandb:
            self.wandb.log(
                {
                    "test_mae": avg_test_loss,
                    "test_r": test_r,
                    "test_scatter": self.wandb.Image(fig),
                }
            )

        plt.close(fig)
        print(f"Test MAE: {avg_test_loss:.4f} | Test r: {test_r:.4f}")

        return avg_test_loss, test_r

    # -------------------------
    # Model checkpointing
    # -------------------------
    def _save_best_model(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.output_path)
            print("💾 Saved new best model!")
