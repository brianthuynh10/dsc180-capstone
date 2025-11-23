# main.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from clean_data import clean_main
from model_resnet import make_resnet50_model as ResNet152Regression
import argparse
import wandb

def train_epoch(model, loader, criterion, optimizer, device, use_wandb=False, img_size=None, epoch=None):
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
    avg_loss = total_loss / len(loader)

    if use_wandb and img_size is not None and epoch is not None:
        wandb.log({"img_size": img_size, "epoch": epoch+1, "train_loss": avg_loss})

    return avg_loss

def eval_epoch(model, loader, criterion, device, use_wandb=False, img_size=None, epoch=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)

    if use_wandb and img_size is not None and epoch is not None:
        wandb.log({"img_size": img_size, "epoch": epoch+1, "val_loss": avg_loss})

    return avg_loss

def train_at_resolution(img_size, args, use_wandb=False):
    print(f"\n============================")
    print(f" Training at resolution {img_size}")
    print(f"============================\n")

    BATCH_SIZE = args.batch_size
    LR = 1e-4
    EPOCHS = args.epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, y_mean, y_std = clean_main(img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = ResNet152Regression().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE,
                                 use_wandb=use_wandb, img_size=img_size, epoch=epoch)
        val_loss = eval_epoch(model, val_loader, criterion, DEVICE,
                              use_wandb=use_wandb, img_size=img_size, epoch=epoch)

        print(f"[{img_size}] Epoch {epoch+1}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"resnet_best_{img_size}.pt")

    test_loss = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"[{img_size}] Final Test Loss: {test_loss:.4f}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--img_size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    p.add_argument('--wandb_project', type=str, default='resnet-resolution-study')
    p.add_argument('--run_name', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()

    if args.wandb:
        wandb.init(
        entity="zohasan",  # your W&B username
        project="resnet-resolution-study",
        name=f"res_{args.img_size}",
        config={
            "img_size": args.img_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
    }
)

    RESOLUTIONS = [args.img_size] if args.img_size else [128, 256, 512]

    for res in RESOLUTIONS:
        train_at_resolution(res, args, use_wandb=args.wandb)

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
