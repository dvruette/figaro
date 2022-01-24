
import os
import glob
import time
import torch
import random
from torch.utils.data import DataLoader

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
from input_representation import remi2midi

MODEL = os.getenv('MODEL', '')

ROOT_DIR = os.getenv('ROOT_DIR', './lmd_full')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './samples')
MAX_N_FILES = int(float(os.getenv('MAX_N_FILES', -1)))
MAX_ITER = int(os.getenv('MAX_ITER', 16_000))
MAX_BARS = int(os.getenv('MAX_BARS', 32))

MAKE_MEDLEYS = os.getenv('MAKE_MEDLEYS', 'False') == 'True'
N_MEDLEY_PIECES = int(os.getenv('N_MEDLEY_PIECES', 2))
N_MEDLEY_BARS = int(os.getenv('N_MEDLEY_BARS', 16))
  
CHECKPOINT = os.getenv('CHECKPOINT', None)
VAE_CHECKPOINT = os.getenv('VAE_CHECKPOINT', None)
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1))
VERBOSE = int(os.getenv('VERBOSE', 2))

def reconstruct_sample(model, batch, 
  initial_context=1, 
  output_dir=None, 
  max_iter=-1, 
  max_bars=-1,
  verbose=0,
):
  batch_size, seq_len = batch['input_ids'].shape[:2]

  batch_ = { key: batch[key][:, :initial_context] for key in ['input_ids', 'bar_ids', 'position_ids'] }
  if model.description_flavor in ['description', 'both']:
    batch_['description'] = batch['description']
    batch_['desc_bar_ids'] = batch['desc_bar_ids']
  if model.description_flavor in ['latent', 'both']:
    batch_['latents'] = batch['latents']

  max_len = seq_len + 1024
  if max_iter > 0:
    max_len = min(max_len, initial_context + max_iter)
  if verbose:
    print(f"Generating sequence ({initial_context} initial / {max_len} max length / {max_bars} max bars / {batch_size} batch size)")
  sample = model.sample(batch_, max_length=max_len, max_bars=max_bars, verbose=verbose//2)

  xs = batch['input_ids'].detach().cpu()
  xs_hat = sample['sequences'].detach().cpu()
  events = [model.vocab.decode(x) for x in xs]
  events_hat = [model.vocab.decode(x) for x in xs_hat]

  pms, pms_hat = [], []
  n_fatal = 0
  for rec, rec_hat in zip(events, events_hat):
    try:
      pm = remi2midi(rec)
      pms.append(pm)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)
    try:
      pm_hat = remi2midi(rec_hat)
      pms_hat.append(pm_hat)
    except Exception as err:
      print("ERROR: Could not convert events to midi:", err)
      n_fatal += 1

  if output_dir:
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    for pm, pm_hat, file in zip(pms, pms_hat, batch['files']):
      if verbose:
        print(f"Saving to {output_dir}/{file}")
      pm.write(os.path.join(output_dir, 'gt', file))
      pm_hat.write(os.path.join(output_dir, file))

  return events


def main():
  if MAKE_MEDLEYS:
    max_bars = N_MEDLEY_PIECES * N_MEDLEY_BARS
  else:
    max_bars = MAX_BARS

  if OUTPUT_DIR:
    params = []
    if MAKE_MEDLEYS:
      params.append(f"n_pieces={N_MEDLEY_PIECES}")
      params.append(f"n_bars={N_MEDLEY_BARS}")
    if MAX_ITER > 0:
      params.append(f"max_iter={MAX_ITER}")
    if MAX_BARS > 0:
      params.append(f"max_bars={MAX_BARS}")
    output_dir = os.path.join(OUTPUT_DIR, MODEL, ','.join(params))
  else:
    raise ValueError("OUTPUT_DIR must be specified.")

  print(f"Saving generated files to: {output_dir}")

  if VAE_CHECKPOINT:
    vae_module = VqVaeModule.load_from_checkpoint(VAE_CHECKPOINT)
    vae_module.cpu()
  else:
    vae_module = None

  model = Seq2SeqModule.load_from_checkpoint(CHECKPOINT)
  model.freeze()
  model.eval()


  midi_files = glob.glob(os.path.join(ROOT_DIR, '**/*.mid'), recursive=True)
  
  dm = model.get_datamodule(midi_files, vae_module=vae_module)
  dm.setup('test')
  midi_files = dm.test_ds.files
  random.shuffle(midi_files)

  if MAX_N_FILES > 0:
    midi_files = midi_files[:MAX_N_FILES]


  description_options = None
  if MODEL in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
    description_options = model.description_options

  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    description_options=description_options,
    max_bars=model.context_size,
    vae_module=vae_module
  )


  start_time = time.time()
  coll = SeqCollator(context_size=-1)
  dl = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=coll)

  if MAKE_MEDLEYS:
    dl = medley_iterator(dl, 
      n_pieces=N_MEDLEY_BARS, 
      n_bars=N_MEDLEY_BARS, 
      description_flavor=model.description_flavor
    )
  
  with torch.no_grad():
    for batch in dl:
      reconstruct_sample(model, batch, 
        output_dir=output_dir, 
        max_iter=MAX_ITER, 
        max_bars=max_bars,
        verbose=VERBOSE,
      )

if __name__ == '__main__':
  main()
