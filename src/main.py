# main.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from clean_data import clean_main
from model_resnet import make_resnet50_model as ResNet152Regression
import argparse
import wandb



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_at_resolution(img_size):
    print(f"\n============================")
    print(f" Training at resolution {img_size}")
    print(f"============================\n")

    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_ds, val_ds, test_ds, y_mean, y_std = clean_main(img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    model = ResNet152Regression().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = eval_epoch(model, val_loader, criterion, DEVICE)
                # example inside training loop, after computing train_loss and val_loss:
        if args.wandb:
            wandb.log({"img_size": img_size, "epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[{img_size}] Epoch {epoch+1}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"resnet_best_{img_size}.pt")

    # Final test loss
    test_loss = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"[{img_size}] Final Test Loss: {test_loss:.4f}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    p.add_argument('--wandb_project', type=str, default='resnet-resolution-study')
    p.add_argument('--run_name', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # initialize wandb if requested
    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name or f"res_{args.img_size}", config={
            "img_size": args.img_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        })

    RESOLUTIONS = [args.img_size] if args.img_size else [128,256,512]

    for res in RESOLUTIONS:
        train_at_resolution(res, args)   # pass args through (see below)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
