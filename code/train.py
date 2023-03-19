import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import sentencepiece as spm

from utils import (
    seed_everything,
    trainable_model_params,
    total_model_params,
    save_train_metrics,
    process_train_eval,
    Loader,
)

from transformer import TransformerEncoder


parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument(
    "--tokenizer_path",
    type=str,
    default="spm_wiki.model",
    help="Trained sentencepiece model file",
)
parser.add_argument(
    "--id_dataset",
    type=str,
    default="20news",
    choices=["20news", "trec", "sst"],
    help="The in-domain dataset to train on",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "val", "test"],
    help="Which dataset split to use for training",
)
parser.add_argument(
    "--val_split",
    type=str,
    default="val",
    choices=["train", "val", "test"],
    help="Which dataset split to use for validation",
)
parser.add_argument("--saving_path", type=str, default="./")
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--max_len",
    type=int,
    default=512,
    help="Maximum sequence length before truncating at this value",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="Number of workers to be used in torch dataloaders",
)
parser.add_argument("--num_epochs", type=int, default=25)
parser.add_argument(
    "--lr1", type=float, default=1e-4, help="Learning rate before a change in epoch n"
)
parser.add_argument(
    "--lr2", type=float, default=5e-5, help="Learning rate after a change in epoch n"
)
parser.add_argument("--change_lr_in_epoch", type=int, default=10)

parser.add_argument(
    "--emb_dim", type=int, default=128, help="Transformer embedding dimension"
)
parser.add_argument(
    "--num_layers", type=int, default=1, help="Number of transformer blocks"
)
parser.add_argument(
    "--num_heads", type=int, default=2, help="Number of attention heads"
)
parser.add_argument(
    "--forward_dim",
    type=int,
    default=256,
    help="Number of dimensions in feed-forward model",
)
parser.add_argument("--dropout", type=float, default=0.05)
parser.add_argument(
    "--kind",
    type=str,
    default="sinkhorn",
    choices=["sinkhorn", "sto", "sto_dual"],
    help="Kind of stochastic attention",
)
parser.add_argument(
    "--tau1",
    type=int,
    default=None,
    help="Concentration factor on centroids. Only needed when kind=sto_dual is chosen",
)
parser.add_argument(
    "--tau2",
    type=int,
    default=None,
    help="Concentration factor on values. Only needed when kind=sto is chosen",
)
parser.add_argument(
    "--k_centroid",
    type=int,
    default=None,
    help="Number of centroids. Only needed when kind=sto_dual is chosen.",
)
parser.add_argument(
    "--spectral",
    default=False,
    action="store_true",
    help="Whether spectral normalization should be applied during training",
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=20,
    choices=[20, 50, 2],
    help="Number of classes in dataset",
)

args = parser.parse_args()


seed_everything(args.random_seed)
device = args.device
tokenizer = spm.SentencePieceProcessor(args.tokenizer_path)

loader = Loader(
    batch_size=args.batch_size, max_len=args.max_len, num_workers=args.num_workers
)
train_loader = loader.load(args.id_dataset, split=args.train_split)
val_loader = loader.load(args.id_dataset, split=args.val_split)


model = TransformerEncoder(
    vocab_size=len(tokenizer),
    emb_dim=args.emb_dim,
    n_layers=args.num_layers,
    n_heads=args.num_heads,
    forward_dim=args.forward_dim,
    dropout=args.dropout,
    max_len=args.max_len,
    pad_idx=tokenizer.pad_id(),
    kind=args.kind,
    spectral=args.spectral,
    tau1=args.tau1,
    tau2=args.tau2,
    k_centroid=args.k_centroid,
    n_classes=args.num_classes,
    device=args.device,
).to(args.device)

print(f"Total model params: {total_model_params(model):,d}")
print(f"Trainable model params: {trainable_model_params(model):,d}")

optimizer = optim.Adam(model.parameters(), lr=args.lr1)
criterion = nn.CrossEntropyLoss()


path = os.path.join(args.saving_path, args.kind)
now = datetime.now()
path_now = os.path.join(path, now.strftime("%Y-%m-%d %H:%M:%S"))
os.makedirs(path_now)
with open(path_now + "/params.json", "w") as f:
    json.dump(vars(args), f, indent=4)


highest_val_acc = 0
for epoch in range(1, args.num_epochs + 1):
    if epoch == args.change_lr_in_epoch:
        for g in optimizer.param_groups:
            g["lr"] = args.lr2
            print(f"updated learning rate in epoch {epoch}")
    model.train()
    train_loss, train_acc = process_train_eval(
        model, train_loader, criterion, optimizer
    )

    model.eval()
    with torch.no_grad():
        val_loss, val_acc = process_train_eval(model, val_loader, criterion)

    save_train_metrics(
        epoch,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
        path=path_now + "/results.csv",
    )

    if val_acc > highest_val_acc:
        highest_val_acc = val_acc
        _path = path_now + f"/acc{val_acc:.4f}_epoch{epoch}.pt"
        torch.save(model.state_dict(), _path)

    print(
        f"Training:   [Epoch {epoch:2d}, Loss: {train_loss:8.6f}, Acc: {train_acc:.6f}]"
    )
    print(f"Evaluation: [Epoch {epoch:2d}, Loss: {val_loss:8.6f}, Acc: {val_acc:.6f}]")
