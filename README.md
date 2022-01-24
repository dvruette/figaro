# FIGARO: Generating Symbolic Music with Fine-Grained Artistic Control
by Dimitri von Rütte, Luca Biggio and Yannic Kilcher

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
With Conda:
```bash
conda create --name figaro python=3.9
conda activate figaro
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
MODEL=figaro-expert CHECKPOINT=./checkpoints/figaro-expert.ckpt python src/generate.py
```

For models using the learned description (`figaro` and `figaro-learned`), a pre-trained VQ-VAE checkpoint needs to be provided as well:
```bash
MODEL=figaro CHECKPOINT=./checkpoints/figaro.ckpt VAE_CHECKPOINT=./checkpoints/vq-vae.ckpt python src/generate.py
```

## Evaluation

We provide the evaluation scripts used to calculate the desription metrics on some set of generated samples.
Refer to the previous section for how to generate samples yourself.

Example usage:
```bash
SAMPLE_DIR=./samples/figaro-expert python src/evaluate.py
```

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
| Variable | Description | Default value |
|-|-|-|
| `MODEL` | Specify which model will be loaded | |
| `CHECKPOINT` | Path to the checkpoint for the specified model | |
| `VAE_CHECKPOINT` | Path to the VQ-VAE checkpoint to be used for the learned description (if applicable) | |
| `ROOT_DIR` | Folder containing MIDI files to extract descriptions from | `./lmd_full` |
| `OUTPUT_DIR` | Folder to save generated MIDI samples to | `./samples` |
| `MAX_ITER` | Max. number of tokens that should be generated | 16,000 |
| `MAX_BARS` | Max. number of bars that should be generated | 32 |
| `MAKE_MEDLEYS` | Set to `True` if descriptions should be combined into medleys. | `False` |
| `N_MEDLEY_PIECES` | Number of pieces to be combined into one | 2 |
| `N_MEDLEY_BARS` | Number of bars to take from each piece | 16 |
| `VERBOSE` | Logging level, set to 0 for silent execution | 2 |
  

### Evaluation (`evaluate.py`)
| `SAMPLE_DIR` | Folder containing generated samples which should be evaluated | `./samples` |
| `OUT_FILE` | CSV file to which a detailed log of all metrics will be saved to | `./metrics.csv` |
| `MAX_SAMPLES` | Limit the number of samples to be used for computing evaluation metrics | 1024 |