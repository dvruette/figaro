# FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control

Listen to the samples on [Soundcloud](https://soundcloud.com/user-751999449/sets/figaro-generating-symbolic-music-with-fine-grained-artistic-control).

Paper: https://openreview.net/forum?id=NyR8OZFHw6i

Colab Demo: https://colab.research.google.com/drive/1UAKFkbPQTfkYMq1GxXfGZOJXOXU_svo6

---

- [FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control](#figaro-generating-symbolic-music-with-fine-grained-artistic-control)
  - [Getting started](#getting-started)
    - [Setup](#setup)
    - [Preparing the Data](#preparing-the-data)
    - [Download Pre-Trained Models](#download-pre-trained-models)
  - [Training](#training)
  - [Generation](#generation)
  - [Evaluation](#evaluation)
  - [Parameters](#parameters)
    - [Training (`train.py`)](#training-trainpy)
    - [Generation (`generate.py`)](#generation-generatepy)
    - [Evaluation (`evaluate.py`)](#evaluation-evaluatepy)

## Getting started
Prerequisites:
- Python 3.9
- Conda

### Setup
1. Clone this repository to your disk
3. Install required packages (see requirements.txt).
With `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Preparing the Data

To train models and to generate new samples, we use the [Lakh MIDI](https://colinraffel.com/projects/lmd/) dataset (altough any collection of MIDI files can be used).
1. Download (size: 1.6GB) and extract the archive file:
```bash
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
tar -xzf lmd_full.tar.gz
```
2. You may wish to remove the archive file now: `rm lmd_full.tar.gz`

### Download Pre-Trained Models
If you don't wish to train your own models, you can download our pre-trained models.
1. Download (size: 2.3GB) and extract the archive file:
```bash
wget -O checkpoints.zip https://polybox.ethz.ch/index.php/s/a0HUHzKuPPefWkW/download
unzip checkpoints.zip
```
2. You may wish to remove the archive file now: `rm checkpoints.zip`



## Training
Training arguments such as model type, batch size, model params are passed to the training scripts via environment variables.

Available model types are:
- `vq-vae`: VQ-VAE model used for the learned desription
- `figaro`: FIGARO with both the expert and learned description
- `figaro-expert`: FIGARO with only the expert description
- `figaro-learned`: FIGARO with only the learned description
- `figaro-no-inst`: FIGARO (expert) without instruments
- `figaro-no-chord`: FIGARO (expert) without chords
- `figaro-no-meta`: FIGARO (expert) without style (meta) information
- `baseline`: Unconditional decoder-only baseline following [Huang et al. (2018)](https://arxiv.org/abs/1809.04281)

Example invocation of the training script is given by the following command:
```bash
MODEL=figaro-expert python src/train.py
```

For models using the learned description (`figaro` and `figaro-learned`), a pre-trained VQ-VAE checkpoint needs to be provided as well:
```bash
MODEL=figaro VAE_CHECKPOINT=./checkpoints/vq-vae.ckpt python src/train.py
```

## Generation
To generate samples, make sure you have a trained checkpoint prepared (either download one or train it yourself).
For this script, make sure that the dataset is prepared according to [Preparing the Data](#preparing-the-data).
This is needed to extract descriptions, based on which new samples can be generated.

An example invocation of the generation script is given by the following command:
```bash
python src/generate.py --model figaro-expert --checkpoint ./checkpoints/figaro-expert.ckpt
```

For models using the learned description (`figaro` and `figaro-learned`), a pre-trained VQ-VAE checkpoint needs to be provided as well:
```bash
python src/generate.py --model figaro --checkpoint ./checkpoints/figaro.ckpt --vae_checkpoint ./checkpoints/vq-vae.ckpt
```

## Evaluation

We provide the evaluation scripts used to calculate the desription metrics on some set of generated samples.
Refer to the previous section for how to generate samples yourself.

Example usage:
```bash
python src/evaluate.py --samples_dir ./samples/figaro-expert
```

It has been pointed out that the order of the dataset files (from which the splits are calculated) is non-deterministic and depends on the OS.
To address this and to ensure reproducibility, I have added the exact files used for training/validation/testing in the respective file in the `splits` folder.

## Parameters
The following environment variables are available for controlling hyperparameters beyond their default value.
### Training (`train.py`)
Model
| Variable | Description | Default value |
|-|-|-|
| `MODEL` | Model architecture to be trained | |
| `D_MODEL` | Hidden size of the model | 512 |
| `CONTEXT_SIZE` | Number of tokens in the context to be passed to the auto-encoder | 256 |
| `D_LATENT` | [VQ-VAE] Dimensionality of the latent space | 1024 |
| `N_CODES` | [VQ-VAE] Codebook size | 2048 |
| `N_GROUPS` | [VQ-VAE] Number of groups to split the latent vector into before discretization | 16 |

Optimization
| Variable | Description | Default value |
|-|-|-|
| `EPOCHS` | Max. number of training epochs | 16 |
| `MAX_TRAINING_STEPS` | Max. number of training iterations | 100,000 |
| `BATCH_SIZE` | Number of samples in each batch | 128 |
| `TARGET_BATCH_SIZE` | Number of samples in each backward step, gradients will be accumulated over `TARGET_BATCH_SIZE//BATCH_SIZE` batches | 256 |
| `WARMUP_STEPS` | Number of learning rate warmup steps | 4000 |
| `LEARNING_RATE` | Initial learning rate, will be decayed after constant warmup of `WARMUP_STEPS` steps | 1e-4 |

Others
| Variable | Description | Default value |
|-|-|-|
| `CHECKPOINT` | Path to checkpoint from which to resume training | |
| `VAE_CHECKPOINT` | Path to VQ-VAE checkpoint to be used for the learned description | |
| `ROOT_DIR` | The folder containing MIDI files to train on | `./lmd_full` |
| `OUTPUT_DIR` | Folder for saving checkpoints | `./results` |
| `LOGGING_DIR` | Folder for saving logs | `./logs` |
| `N_WORKERS` | Number of workers to be used for the dataloader | available CPUs |


### Generation (`generate.py`)
The generation script uses command line arguments instead of environment variables.
| Argument | Description | Default value |
|-|-|-|
| `--model` | Specify which model will be loaded | |
| `--checkpoint` | Path to the checkpoint for the specified model | |
| `--vae_checkpoint` | Path to the VQ-VAE checkpoint to be used for the learned description (if applicable) | |
| `--lmd_dir` | Folder containing MIDI files to extract descriptions from | `./lmd_full` |
| `--output_dir` | Folder to save generated MIDI samples to | `./samples` |
| `--max_iter` | Max. number of tokens that should be generated | 16,000 |
| `--max_bars` | Max. number of bars that should be generated | 32 |
| `--make_medleys` | Set to `True` if descriptions should be combined into medleys. | `False` |
| `--n_medley_pieces` | Number of pieces to be combined into one | 2 |
| `--n_medley_bars` | Number of bars to take from each piece | 16 |
| `--verbose` | Logging level, set to 0 for silent execution | 2 |
  

### Evaluation (`evaluate.py`)
The evaluation script uses command line arguments instead of environment variables.
| Argument | Description | Default value |
|-|-|-|
| `--samples_dir` | Folder containing generated samples which should be evaluated | `./samples` |
| `--output_file` | CSV file to which a detailed log of all metrics will be saved to | `./metrics.csv` |
| `--max_samples` | Limit the number of samples to be used for computing evaluation metrics | 1024 |
