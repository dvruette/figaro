import argparse
import os
import glob
import torch
import random
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertAttention

from models.vae import VqVaeModule
from models.seq2seq import Seq2SeqModule
from datasets import MidiDataset, SeqCollator
from utils import medley_iterator
from input_representation import remi2midi


def parse_args():
  parser = argparse.ArgumentParser()
  # parser.add_argument('--model', type=str, required=True, help="Model name (one of 'figaro', 'figaro-expert', 'figaro-learned', 'figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta')")
  # parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
  parser.add_argument('--model', type=str, default="figaro-expert")
  parser.add_argument('--checkpoint', type=str, default="../figaro-expert.ckpt")
  parser.add_argument('--vae_checkpoint', type=str, default=None, help="Path to the VQ-VAE model checkpoint (optional)")
  parser.add_argument('--lmd_dir', type=str, default='./lmd_full', help="Path to the root directory of the LakhMIDI dataset")
  parser.add_argument('--output_dir', type=str, default='./samples', help="Path to the output directory")
  parser.add_argument('--max_n_files', type=int, default=-1)
  parser.add_argument('--max_iter', type=int, default=16_000)
  parser.add_argument('--max_bars', type=int, default=32)
  parser.add_argument('--make_medleys', type=bool, default=False)
  parser.add_argument('--n_medley_pieces', type=int, default=2)
  parser.add_argument('--n_medley_bars', type=int, default=16)
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--verbose', type=int, default=2)
  args = parser.parse_args()
  return args


def load_old_or_new_checkpoint(model_class, checkpoint):
  # assuming transformers>=4.36.0
  pl_ckpt = torch.load(checkpoint, map_location="cpu")
  kwargs = pl_ckpt['hyper_parameters']
  if 'flavor' in kwargs:
    del kwargs['flavor']
  if 'vae_run' in kwargs:
    del kwargs['vae_run']
  model = model_class(**kwargs)
  state_dict = pl_ckpt['state_dict']
  # position_ids are no longer saved in the state_dict starting with transformers==4.31.0
  state_dict = {k: v for k, v in state_dict.items() if not k.endswith('embeddings.position_ids')}
  try:
    # succeeds for checkpoints trained with transformers>4.13.0
    model.load_state_dict(state_dict)
  except RuntimeError:
    # work around a breaking change introduced in transformers==4.13.0, which fixed the position_embedding_type of cross-attention modules "absolute"
    config = model.transformer.decoder.bert.config
    for layer in model.transformer.decoder.bert.encoder.layer:
      layer.crossattention = BertAttention(config, position_embedding_type=config.position_embedding_type)
    model.load_state_dict(state_dict)
  model.freeze()
  model.eval()
  return model


def load_model(checkpoint, vae_checkpoint=None, device='auto'):
  if device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  vae_module = None
  if vae_checkpoint:
    vae_module = load_old_or_new_checkpoint(VqVaeModule, vae_checkpoint)
    vae_module.cpu()

  model = load_old_or_new_checkpoint(Seq2SeqModule, checkpoint)
  model.to(device)

  return model, vae_module


@torch.no_grad()
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
    os.makedirs(os.path.join(output_dir, 'ground_truth'), exist_ok=True)
    for pm, pm_hat, file in zip(pms, pms_hat, batch['files']):
      if verbose:
        print(f"Saving to {output_dir}/{file}")
      pm.write(os.path.join(output_dir, 'ground_truth', file))
      pm_hat.write(os.path.join(output_dir, file))

  return events


def main():
  args = parse_args()
  if args.make_medleys:
    max_bars = args.n_medley_pieces * args.n_medley_bars
  else:
    max_bars = args.max_bars

  if args.output_dir:
    params = []
    if args.make_medleys:
      params.append(f"n_pieces={args.n_medley_pieces}")
      params.append(f"n_bars={args.n_medley_bars}")
    if args.max_iter > 0:
      params.append(f"max_iter={args.max_iter}")
    if args.max_bars > 0:
      params.append(f"max_bars={args.max_bars}")
    output_dir = os.path.join(args.output_dir, args.model, ','.join(params))
  else:
    raise ValueError("args.output_dir must be specified.")

  print(f"Saving generated files to: {output_dir}")

  model, vae_module = load_model(args.checkpoint, args.vae_checkpoint)


  midi_files = glob.glob(os.path.join(args.lmd_dir, '**/*.mid'), recursive=True)
  
  dm = model.get_datamodule(midi_files, vae_module=vae_module)
  dm.setup('test')
  midi_files = dm.test_ds.files
  random.shuffle(midi_files)

  if args.max_n_files > 0:
    midi_files = midi_files[:args.max_n_files]


  description_options = None
  if args.model in ['figaro-no-inst', 'figaro-no-chord', 'figaro-no-meta']:
    description_options = model.description_options

  dataset = MidiDataset(
    midi_files,
    max_len=-1,
    description_flavor=model.description_flavor,
    description_options=description_options,
    max_bars=model.context_size,
    vae_module=vae_module
  )

  coll = SeqCollator(context_size=-1)
  dl = DataLoader(dataset, batch_size=args.batch_size, collate_fn=coll)

  if args.make_medleys:
    dl = medley_iterator(dl, 
      n_pieces=args.n_medley_pieces, 
      n_bars=args.n_medley_bars, 
      description_flavor=model.description_flavor
    )
  
  with torch.no_grad():
    for batch in dl:
      reconstruct_sample(model, batch, 
        output_dir=output_dir, 
        max_iter=args.max_iter, 
        max_bars=max_bars,
        verbose=args.verbose,
      )

if __name__ == '__main__':
  main()
