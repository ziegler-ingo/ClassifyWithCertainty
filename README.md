# Out-of-Distribution Detection in Text Classification
This repository implements a combination of transformers built around stochastic attention
with the Deep Deterministic Uncertainty (DDU) framework.

## Architecture
### Transformers
* Sinkformer
	* a transformer built around doubly-stochastic attention using Sinkhorn's algorithm
	* the softmax in attention is replaced with Sinkhorn's algorithm, which iteratively
	normalizes both the rows and columns of the attention matrix to sum up to 1
	* [Paper](https://arxiv.org/abs/2110.11773), [Code](https://github.com/michaelsdr/sinkformers)
* Hierarchical Stochastic Attention
	* Directly induce randomness through the sampling process
	* STO
		* Approximates softmax attention by sampling values from the Gumbel-Softmax distribution
		based on $QK^T$
	* STO-DUAL
		* First approximates $QK^T$
		* Allows the $K$ to attend to a set of learnable centroids
		* This process can be seen as clustering over hidden representations of $K$
		* These clusters then sample probabilities form the Gumbel-Softmax distribution
		* $QK^T$ is then approximated by the dot-product between original $Q$ and sampled
		probabilities
		* After that, the process of STO is followed, just with the centroid-approximated
		representation of $QK^T$
	* [Paper](https://arxiv.org/abs/2112.13776), [Code](https://github.com/amzn/sto-transformer)

### Deep Deterministic Uncertainty
A framework that is originally meant to make deterministic models (like ResNet) uncertainty-aware.
It consists of two main steps:
* Regularize the feature space with spectral normaliziation during training
* Fit a Gaussian Mixture Model (GMM) with one component per class *after* training to discriminate between
epistemic and aleatoric uncertainty
	* Alternatively, temperature scaling may replace the GMM step
* [Paper](https://arxiv.org/abs/2102.11582), [Code](https://github.com/omegafragger/DDU/blob/main/utils/gmm_utils.py)

The differentiation between aleatoric and epistemic uncertainty allows for out-of-distribution
detection because we seek to detect high epistemic uncertainty samples. 

Most importantly, this approach:
* only requires a single model to be trained with a single forward-pass at inference times
(as opposed to ensembles)
* does not require the model being conditioned to out-of-distribution data during training
(as opposed to [this](https://doi.org/10.1145/3447548.3467382) approach in text classification)
* does no require adversarial (pseudo-)samples during training
(as opposed to [this](https://doi.org/10.1145/3447548.3467382) approach in text classification
* outperforms spectral-normalized Gaussian Processes (SNGP) and even ensembles on selected image benchmarks
* Didn't lead to improvements on this text classification benchmark

## Experiments
The experiments closely follow the benchmark approach from [this](https://doi.org/10.1145/3447548.3467382)
paper. 

The transformers are separately trained on 3 different in-domain datasets (i.e. no pre-training needed!), and then evaluated on 5 different
out-of-domain datasets.

During training, we optimize for loss and accuracy on the original classes of the respective dataset. For evaluation,
we binarize  the labels and only differentiate between in-distribution (label 0) and out-of-distribution (label 1).

Performance is evaluated on AUROC, AUPRC, the false-positivity rate at 90% recall (FPR90).

Hyperparameters for training were tuned with a naive grid-search approach. The final hyperparameters for each 
model and evaluation combination can be found under `./code/run_configs`.

Each transformer experiment is performed with 3 different ablations to check for improvements and deteriorations.
1. Only transformers, combined with cross entropy loss (CE), expected calibration error (ECE),
thresholded adaptive calibration error (TACE) temperature scaling
2. Transformers trained with spectral normalization, combined with CE, ECE, TACE temperature scaling
3. Transformers trained with spectral normalization, followed by GMM

Approach 1 performed best, closely followed by 2, which outperforms 3 by a lot.

## Running the code
### Training
Check out the notes [here](https://github.com/snowood1/BERT-ENN) about getting the datasets before running
my [prep\_datasets.py](https://github.com/ziegler-ingo/ClassifyWithCertainty/blob/master/code/prep_datasets.py) script
because some datasets need to be downloaded separately as they are no longer included in torchtext's datasets.

The train script has the following options:

`--random_seed 
--tokenizer_path 
--id_dataset 
--train_split 
--val_split 
--saving_path 
--device 
--batch_size 
--max_len 
--num_workers 
--num_epochs 
--lr1 
--lr2
--change_lr_in_epoch 
--emb_dim 
--num_layers 
--num_heads 
--forward_dim 
--dropout 
--kind 
--num_classes
`

All model configurations, i.e. `sinkformer`, `sto`, `sto_dual`, and with or without `spectral` normalization,
can be found in `./code/run_configs/train` with the grid-search optimized values pre-configured.

To run the train script on the 20News dataset, simply execute `python3 train.py $(cat ./run_configs/train/sinkformer_20news.cfg)`.
Please mind that the pre-configured saving path is `./`, so adjust this to your needs. The final model should be named `model.pt`.

### Evaluation
The evaluation script has the following options:

`--random_seed 
--tokenizer_path 
--id_dataset 
--model_path 
--dataset_path 
--eval_mode 
--device 
--batch_size 
--max_len 
--num_workers 
--emb_dim 
--num_layers 
--num_heads 
--forward_dim 
--dropout 
--kind 
--num_classes 
`

Again, all optimized values are pre-configured in `./code/run_configs/eval`.

Before running evaluation, please check that your saving paths for the model and datasets align to your locations.
Then, you can run evaluation with `python3 evaluate.py $(cat ./run_configs/eval/sinkformer_20news.cfg)`.

The options here are `sinkformer_20news.cfg`, i.e. without spectral normalization (option 1 from above), `sinkformer_20news_spectral.cfg`
for a model with spectral normalization and temperature scaling evaluation (option 2 from above), and finally `sinkformer_20news_spectral_gmm.cfg`
to evaluate a model with spectral normalization followed by GMM (option 3 from above).

Results are printed to standard-out by default, so you might want to lead all output to a file by using
`python3 evaluate.py $(cat ./run_configs/eval/sinkformer_20news.cfg) >> res_sinkformer_20news.txt`.
