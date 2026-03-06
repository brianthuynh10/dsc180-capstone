import argparse
from models.CNNs.train import Trainer
from models.CNNs.data import main as clean


def main(model_name):
    # -- Pull Data --
    print("Beginning data cleaning")
    train, val, test, y_mean, y_std = clean()  # XRayDataset objects

    # -- Train Model (using papers setup) --
    print("Model created and training will start now")
    trainer = Trainer(
        model_name=model_name,
        epochs=50,
        lr=1e-5,
        batch_size=16,
        train_dataset=train,
        val_dataset=val,
        test_dataset=test,
        train_mean=y_mean,
        train_std=y_std,
    )
    # -- This will handle the training loop, validation, and testing --
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("model", choices=["resnet50", "vgg16"], help="Model to train")
    args = parser.parse_args()
    main(args.model)