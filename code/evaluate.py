import torch
import argparse
import sentencepiece as spm

from utils import (
    seed_everything,
    total_model_params,
    trainable_model_params,
    binary_ood_id_repr,
    process_partially,
    get_best_temperature,
    Loader,
)

from metrics import ECELoss, TACELoss, get_auroc_auprc, fpr_at_k_recall

from gmm import fit_gmm, evaluate_gmm
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
    help="The in-domain dataset the model was trained on",
)
parser.add_argument(
    "--model_path", type=str, default="./", help="Path where the trained model is saved"
)
parser.add_argument("--dataset_path", type=str, default="../../../datasets")
parser.add_argument(
    "--eval_mode",
    type=str,
    default="temperature",
    choices=["gmm", "temperature"],
    help="Whether to only run temperature scaling or the GMM model for evaluation",
)
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
val_loader = loader.load(args.id_dataset, "val")
id_test_loader = loader.load(args.id_dataset, "test")
ood_snli_loader = loader.load("snli", "test")
ood_imdb_loader = loader.load("imdb", "test")
ood_m30k_loader = loader.load("m30k", "test")
ood_wmt16_loader = loader.load("wmt16", "test")
ood_yelp_loader = loader.load("yelp", "test")

ood_loaders = [
    ood_snli_loader,
    ood_imdb_loader,
    ood_m30k_loader,
    ood_wmt16_loader,
    ood_yelp_loader,
]
ood_names = ["snli", "imdb", "m30k", "wmt16", "yelp"]

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
model.load_state_dict(torch.load(args.model_path))
model.eval()

print(f"Total model params: {total_model_params(model):,d}")
print(f"Trainable model params: {trainable_model_params(model):,d}")

ece_criterion = ECELoss(num_bins=15)
tace_criterion = TACELoss(num_bins=15)

id_test_logits, id_test_lbls = process_partially(
    model, id_test_loader, only_logits=True
)  # shape: (test_set_size, num_classes)

# get best temperature
best_ce_temp, best_ece_temp, best_tace_temp = get_best_temperature(
    model, val_loader, num_tries=100, num_bins=15, log=True
)
print("Best CE temperature:", best_ce_temp)
print("Best ECE temperature:", best_ece_temp)
print("Best TACE temperature:", best_tace_temp)

t_ce_test_logits = id_test_logits / best_ce_temp
t_ece_test_logits = id_test_logits / best_ece_temp
t_tace_test_logits = id_test_logits / best_tace_temp

# ECE for id dataset
ece = ece_criterion(id_test_logits, id_test_lbls)
ce_ece = ece_criterion(t_ce_test_logits, id_test_lbls)
ece_ece = ece_criterion(t_ece_test_logits, id_test_lbls)
tace_ece = ece_criterion(t_tace_test_logits, id_test_lbls)

print(f"ECE no temperature scaling: {ece:.4f}")
print(f"ECE with CE temperature scaling: {ce_ece:.4f}")
print(f"ECE with ECE temperature scaling: {ece_ece:.4f}")
print(f"ECE with TACE temperature scaling: {tace_ece:.4f}")

# TACE for id dataset
tace = tace_criterion(id_test_logits, id_test_lbls)
ce_tace = tace_criterion(t_ce_test_logits, id_test_lbls)
ece_tace = tace_criterion(t_ece_test_logits, id_test_lbls)
tace_tace = tace_criterion(t_tace_test_logits, id_test_lbls)

print(f"TACE no temperature scaling: {tace:.4f}")
print(f"TACE with CE temperature scaling: {ce_tace:.4f}")
print(f"TACE with ECE temperature scaling: {ece_tace:.4f}")
print(f"TACE with TACE temperature scaling: {tace_tace:.4f}")

if args.eval_mode == "temperature":
    for ood_loader, name in zip(ood_loaders, ood_names):
        print("CURRENT OOD SET:", name)
        ood_logits, _ = process_partially(model, ood_loader, only_logits=True)
        ce_ood_logits = ood_logits / best_ce_temp
        ece_ood_logits = ood_logits / best_ece_temp
        tace_ood_logits = ood_logits / best_tace_temp

        # no temperature
        auroc, auprc = get_auroc_auprc(
            id_test_logits, ood_logits, measure="logsumexp", confidence=True
        )
        _logits, _lbls = binary_ood_id_repr(ood_logits, id_test_logits, return_1d=True)
        fpr90 = fpr_at_k_recall(
            _lbls.cpu().numpy(), _logits.cpu().numpy(), recall_level=0.9
        )
        print(f"no temperature, logsumexp, confidence=True, AUROC: {auroc*100:.2f}%")
        print(f"no temperature, logsumexp, confidence=True, AUPRC: {auprc*100:.2f}%")

        auroc, auprc = get_auroc_auprc(id_test_logits, ood_logits, measure="entropy")
        print(f"no temperature, entropy, confidence=False, AUROC: {auroc*100:.2f}%")
        print(f"no temperature, entropy, confidence=False, AUPRC: {auprc*100:.2f}%")
        print(f"FPR90: {fpr90*100:.2f}%")

        # CE temperature
        auroc, auprc = get_auroc_auprc(
            t_ce_test_logits, ce_ood_logits, measure="logsumexp", confidence=True
        )
        _logits, _lbls = binary_ood_id_repr(
            ce_ood_logits, t_ce_test_logits, return_1d=True
        )
        fpr90 = fpr_at_k_recall(
            _lbls.cpu().numpy(), _logits.cpu().numpy(), recall_level=0.9
        )
        print(f"CE temperature, logsumexp, confidence=True, AUROC: {auroc*100:.2f}%")
        print(f"CE temperature, logsumexp, confidence=True, AUPRC: {auprc*100:.2f}%")

        auroc, auprc = get_auroc_auprc(t_ce_test_logits, ce_ood_logits, measure="entropy")
        print(f"CE temperature, entropy, confidence=False, AUROC: {auroc*100:.2f}%")
        print(f"CE temperature, entropy, confidence=False, AUPRC: {auprc*100:.2f}%")
        print(f"FPR90: {fpr90*100:.2f}%")

        # ECE temperature
        auroc, auprc = get_auroc_auprc(
            t_ece_test_logits, ece_ood_logits, measure="logsumexp", confidence=True
        )
        _logits, _lbls = binary_ood_id_repr(
            ece_ood_logits, t_ece_test_logits, return_1d=True
        )
        fpr90 = fpr_at_k_recall(
            _lbls.cpu().numpy(), _logits.cpu().numpy(), recall_level=0.9
        )
        print(f"ECE temperature, logsumexp, confidence=True, AUROC: {auroc*100:.2f}%")
        print(f"ECE temperature, logsumexp, confidence=True, AUPRC: {auprc*100:.2f}%")

        auroc, auprc = get_auroc_auprc(t_ece_test_logits, ece_ood_logits, measure="entropy")
        print(f"ECE temperature, entropy, confidence=False, AUROC: {auroc*100:.2f}%")
        print(f"ECE temperature, entropy, confidence=False, AUPRC: {auprc*100:.2f}%")
        print(f"FPR90: {fpr90*100:.2f}%")

        # TACE temperature
        auroc, auprc = get_auroc_auprc(
            t_tace_test_logits, tace_ood_logits, measure="logsumexp", confidence=True
        )
        _logits, _lbls = binary_ood_id_repr(
            tace_ood_logits, t_tace_test_logits, return_1d=True
        )
        fpr90 = fpr_at_k_recall(
            _lbls.cpu().numpy(), _logits.cpu().numpy(), recall_level=0.9
        )
        print(f"TACE temperature, logsumexp, confidence=True, AUROC: {auroc*100:.2f}%")
        print(f"TACE temperature, logsumexp, confidence=True, AUPRC: {auprc*100:.2f}%")

        auroc, auprc = get_auroc_auprc(
            t_tace_test_logits, tace_ood_logits, measure="entropy"
        )
        print(f"TACE temperature, entropy, confidence=False, AUROC: {auroc*100:.2f}%")
        print(f"TACE temperature, entropy, confidence=False, AUPRC: {auprc*100:.2f}%")
        print(f"FPR90: {fpr90*100:.2f}%")
        print("------------------------------------------------")
else:
    train_loader = loader.load(args.id_dataset, "train")

    train_embeds, train_lbls = process_partially(
        model, train_loader, return_embeddings=True
    )
    gmm = fit_gmm(train_embeds, train_lbls, args.num_classes)
    id_gmm_logits, _ = evaluate_gmm(model, gmm, id_test_loader)

    for ood_loader, name in zip(ood_loaders, ood_names):
        print("CURRENT OOD SET:", name)
        ood_gmm_logits, _ = evaluate_gmm(model, gmm, ood_loader)

        # id_test_logits, no temperature
        auroc, auprc = get_auroc_auprc(
            id_gmm_logits, ood_gmm_logits, measure="logsumexp", confidence=True
        )
        _logits, _lbls = binary_ood_id_repr(
            ood_gmm_logits, id_gmm_logits, return_1d=True
        )
        fpr90 = fpr_at_k_recall(
            _lbls.cpu().numpy(), _logits.cpu().numpy(), recall_level=0.9
        )
        print(f"no temperature, logsumexp, confidence=True, AUROC: {auroc*100:.2f}%")
        print(f"no temperature, logsumexp, confidence=True, AUPRC: {auprc*100:.2f}%")

        auroc, auprc = get_auroc_auprc(id_gmm_logits, ood_gmm_logits, measure="entropy")
        print(f"no temperature, entropy, confidence=False, AUROC: {auroc*100:.2f}%")
        print(f"no temperature, entropy, confidence=False, AUPRC: {auprc*100:.2f}%")
        print(
            f"FPR90 (meaningless due to scaling problems after GMM): {fpr90*100:.2f}%"
        )
        print("------------------------------------------------")
