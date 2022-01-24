import torch
from torch.nn.utils.rnn import pad_sequence

def combine_batches(batches, bars_per_sequence=8, description_flavor='none', device=None):
  if device is None:
    device = batches[0]['input_ids'].device

  batch_size = batches[0]['input_ids'].size(0)

  zero = torch.zeros(1, device=device, dtype=torch.int)

  contexts = []
  batch_ = {}

  for i in range(batch_size):
    curr_bar = 0
    ctx = {
      'input_ids': [],
      'bar_ids': [],
      'position_ids': [],
      'slices': [],
      'description': [],
      'desc_bar_ids': [],
      'desc_slices': [],
      'latents': [],
      'latent_slices': [],
      'files': [],
    }

    for batch in batches:
      if i >= batch['input_ids'].size(0):
        continue

      curr = curr_bar

      bar_ids = batch['bar_ids'][i]
      starts = (bar_ids >= curr).nonzero()
      ends = (bar_ids >= max(1, curr) + bars_per_sequence).nonzero()
      if starts.size(0) == 0:
        continue
      start = starts[0, 0]

      if ends.size(0) == 0:
        end = bar_ids.size(0)
        curr_bar = bar_ids[-1] + 1
      else:
        end = ends[0, 0]
        curr_bar = bar_ids[end]

      if description_flavor in ['description', 'both']:
        desc_bar_ids = batch['desc_bar_ids'][i]
        desc_start = (desc_bar_ids >= curr).nonzero()[0, 0]
        desc_ends = (desc_bar_ids >= max(1, curr) + bars_per_sequence).nonzero()

        if desc_ends.size(0) == 0:
          desc_end = desc_bar_ids.size(0)
        else:
          desc_end = desc_ends[0, 0]

      if description_flavor in ['latent', 'both']:
        latent_start = curr
        latent_end = max(1, curr) + bars_per_sequence


      ctx['input_ids'].append(batch['input_ids'][i, start:end])
      ctx['bar_ids'].append(batch['bar_ids'][i, start:end])
      ctx['position_ids'].append(batch['position_ids'][i, start:end])
      ctx['slices'].append((start, end))
      if description_flavor in ['description', 'both']:
        ctx['description'].append(batch['description'][i, desc_start:desc_end])
        ctx['desc_bar_ids'].append(batch['desc_bar_ids'][i, desc_start:desc_end])
        ctx['desc_slices'].append((desc_start, desc_end))
      if description_flavor in ['latent', 'both']:
        ctx['latents'].append(batch['latents'][i, latent_start:latent_end])
        ctx['latent_slices'].append((latent_start, latent_end))
      ctx['files'].append(batch['files'][i])

    if len(ctx['files']) <= 1:
      continue
  
    keys = ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids', 'latents']
    for key in keys:
      if key in ctx and len(ctx[key]) > 0:
        ctx[key] = torch.cat(ctx[key])
    ctx['labels'] = torch.cat([ctx['input_ids'][1:], zero])
    ctx['files'] = '__'.join(ctx['files']).replace('.mid', '') + '.mid'

    contexts.append(ctx)

  batch_['files'] = [ctx['files'] for ctx in contexts]

  for key in ['input_ids', 'bar_ids', 'position_ids', 'description', 'desc_bar_ids', 'latents', 'labels']:
    xs = [ctx[key] for ctx in contexts if isinstance(ctx[key], torch.Tensor)]
    if len(xs) > 0:
      xs = pad_sequence(xs, batch_first=True, padding_value=0)
      if not key in ['latents']:
        xs = xs.long()
      batch_[key] = xs

  return batch_


def medley_iterator(dl, n_pieces=2, n_bars=8, description_flavor='none'):
  dl_iter = iter(dl)
  try:
    while True:
      batches = [next(dl_iter) for _ in range(n_pieces)]
      batch = combine_batches(batches, 
        bars_per_sequence=n_bars, 
        description_flavor=description_flavor
      )
      yield batch
  except StopIteration:
    return
