import os
import csv
import sys
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader

import sentencepiece as spm

from metrics import ECELoss, TACELoss, entropy_per_sample


device = "cuda:0" if torch.cuda.is_available() else "cpu"


#############################################################################
# *** General Utils *** #
#############################################################################

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = True # type: ignore

def trainable_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def total_model_params(model):
    return sum(p.numel() for p in model.parameters())

def save_train_metrics(*args, path):
    if not os.path.isfile(path):
        with open(path, "w", newline="\n") as f:
            f.write(
                ",".join(
                    [
                        "epoch",
                        "train_loss",
                        "train_acc",
                        "val_loss",
                        "val_acc",
                    ]
                )
            )
            f.write("\n")
    if args:
        with open(path, "a", newline="\n") as f:
            f.write(",".join([str(arg) for arg in args]))
            f.write("\n")

def binary_ood_id_repr(ood_logits, id_logits, return_1d=None):
    if return_1d is not None:
        ood_logits = entropy_per_sample(ood_logits)
        id_logits = entropy_per_sample(id_logits)
        scores = torch.cat([ood_logits, id_logits])
    else:
        scores = torch.cat([ood_logits, id_logits])
    lbls = torch.zeros(scores.shape[0])
    lbls[:ood_logits.shape[0]] = 1
    return scores, lbls if return_1d is not None else lbls.to(device)


#############################################################################
# *** Train & Eval Loop Utils *** #
#############################################################################

def process_train_eval(model, loader, criterion, optim=None):
    epoch_loss, epoch_acc, total = 0, 0, 0

    for batch in tqdm(
        loader,
        desc="Train: " if optim is not None else "Eval ",
        file=sys.stdout,
        unit="batches",
    ):
        seq, lbl = batch.seq, batch.lbl
        seq, lbl = seq.to(device), lbl.to(device)

        logits = model(seq)
        loss = criterion(logits, lbl)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        epoch_loss += loss.item()
        epoch_acc += ((logits.argmax(dim=-1)) == lbl).sum().item()
        total += seq.shape[0]

    return epoch_loss / total, epoch_acc / total


@torch.no_grad()
def process_partially(model, loader, only_logits=None, return_embeddings=None):
    assert only_logits is not None or return_embeddings is not None
    logits, embeds, lbls = [], [], []

    for batch in tqdm(loader, desc="Partial: ", file=sys.stdout, unit="batches"):
        seq, lbl = batch.seq, batch.lbl
        seq, lbl = seq.to(device), lbl.to(device)

        if return_embeddings is not None:
            embed = model(seq, return_embeddings=return_embeddings)
            embeds.append(embed)
        else:
            _logits = model(seq)
            logits.append(_logits)
        lbls.append(lbl)

    return torch.cat(embeds) if return_embeddings else torch.cat(logits), torch.cat(
        lbls
    )


@torch.no_grad()
def get_best_temperature(model, loader, num_tries, num_bins, log=None):
    logits, lbls = process_partially(model, loader, only_logits=True)

    ce_criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss(num_bins=num_bins)
    tace_criterion = TACELoss(num_bins=num_bins)

    before_temp_ce = ce_criterion(logits, lbls).item()
    before_temp_ece = ece_criterion(logits, lbls)
    before_temp_tace = tace_criterion(logits, lbls)

    if log is not None:
        print(
            f"Before temperature scaling CEL: {before_temp_ce:.4f}, ECE: {before_temp_ece:.4f}, TACE: {before_temp_tace:.4f}"
        )

    best_ce, best_ece, best_tace = 1e7, 1e7, 1e7
    best_ce_temp, best_ece_temp, best_tace_temp = 1.0, 1.0, 1.0
    T = 0.1

    for _ in range(num_tries):
        after_temp_ce = ce_criterion(logits / T, lbls).item()
        after_temp_ece = ece_criterion(logits / T, lbls)
        after_temp_tace = tace_criterion(logits / T, lbls)

        if after_temp_ce < best_ce:
            best_ce = after_temp_ce
            best_ce_temp = T
        if after_temp_ece < best_ece:
            best_ece = after_temp_ece
            best_ece_temp = T
        if after_temp_tace < best_tace:
            best_tace = after_temp_tace
            best_tace_temp = T
        T += 0.1

    if log:
        print(f"After temperature scaling CEL: {best_ce:.4f}, ECE: {best_ece:.4f}, TACE: {best_tace:.4f}")

    return best_ce_temp, best_ece_temp, best_tace_temp


#############################################################################
# *** Data Loading *** #
#############################################################################

# pre-trained SentencePiece Tokenizer taken from Sinkformers Paper
# vocab size of 16000, trained on Wikipedia
tokenizer = spm.SentencePieceProcessor("spm_wiki.model")


def read_data(path):
    with open(path, "r") as csvfile:
        train_data = list(csv.reader(csvfile))[1:]  # skip col name
        sents, lbls = [], []
        for s, l in train_data:
            sents.append(s)
            lbls.append(int(l))
    return sents, lbls


class CustomDataset(Dataset):
    def __init__(self, seq, lbl):
        self.seq = seq
        self.lbl = lbl

    def __getitem__(self, idx):
        return self.seq[idx], self.lbl[idx]

    def __len__(self):
        return len(self.lbl)


class BucketSampler(Sampler):
    def __init__(self, dataset, tokenizer, batch_size, max_len):
        # pair each sequence with their *tokenized* length
        indices = [
            (idx, len(tokenizer.tokenize(review[0], num_threads=-1)[:max_len]))
            for idx, review in enumerate(dataset)
        ]
        random.shuffle(indices)

        idx_pools = []
        # generate pseudo-random batches of (arbitrary) size batch_size * 100
        # each batch of size batch_size * 100 is sorted in itself by seq length
        for i in range(0, len(indices), batch_size * 100):
            idx_pools.extend(
                sorted(indices[i : i + batch_size * 100], key=lambda x: x[1])
            )

        # filter only indices
        self.idx_pools = [x[0] for x in idx_pools]

    def __iter__(self):
        return iter(self.idx_pools)

    def __len__(self):
        return len(self.idx_pools)


class Batch:
    def __init__(self, batch):
        # batch: List[Tuple[str, int]]
        # ordered_batch: List[Tuple[str], Tuple[int]]
        ordered_batch = list(zip(*batch))
        seq = tokenizer.tokenize(list(ordered_batch[0]), num_threads=-1)
        # cut off at 510, add start and end token
        seq = [
            torch.tensor(
                [tokenizer.bos_id()] + s[:510] + [tokenizer.eos_id()], dtype=torch.int64
            )
            for s in seq
        ]
        self.seq = pad_sequence(seq, batch_first=True, padding_value=0)
        self.lbl = torch.tensor(list(ordered_batch[1]), dtype=torch.long)

    def pin_memory(self):
        self.seq = self.seq.pin_memory()
        self.lbl = self.lbl.pin_memory()
        return self


def batch_wrapper(batch):
    return Batch(batch)


class Loader:
    def __init__(self, batch_size, max_len, num_workers):
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_workers = num_workers

    def load(self, dataset, split):
        csv.field_size_limit(200000)
        known = ["20news", "trec", "sst", "snli", "imdb", "m30k", "wmt16", "yelp"]
        assert dataset in known, (
            f"found unknkown dataset: {dataset}, " f"known datasets: {known}"
        )

        datasets_path = '../../../datasets/'
        _path = dataset + '/'
        seq, lbl = read_data(datasets_path + _path + split + ".csv")
        custom_dataset = CustomDataset(seq, lbl)
        bucket_sampler = BucketSampler(
            dataset=custom_dataset,
            tokenizer=tokenizer,
            batch_size=self.batch_size,
            max_len=self.max_len,
        )
        sampler = BatchSampler(
            sampler=bucket_sampler, batch_size=self.batch_size, drop_last=False
        )
        loader = DataLoader(
            dataset=custom_dataset,
            batch_sampler=sampler,
            collate_fn=batch_wrapper,
            pin_memory=True,
        )
        return loader
