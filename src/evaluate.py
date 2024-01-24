import argparse
import os, glob
from statistics import NormalDist
import pandas as pd
import numpy as np

import input_representation as ir

METRICS = [
  'inst_prec', 'inst_rec', 'inst_f1', 
  'chord_prec', 'chord_rec', 'chord_f1', 
  'time_sig_acc', 
  'note_dens_oa', 'pitch_oa', 'velocity_oa', 'duration_oa',
  'chroma_crossent', 'chroma_kldiv', 'chroma_sim',
  'groove_crossent', 'groove_kldiv', 'groove_sim',
]

DF_KEYS = ['id', 'original', 'sample'] + METRICS

keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
qualities = ['maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7', 'None']
CHORDS = [f"{k}:{q}" for k in keys for q in qualities] + ['N:N']


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--samples_dir', type=str, default="./samples")
  parser.add_argument('--output_file', type=str, default="./metrics.csv")
  parser.add_argument('--max_samples', type=int, default=1024)
  args = parser.parse_args()
  return args

def get_group_id(file):
  # change this depending on name of generated samples
  name = os.path.basename(file)
  return name.split('.')[0]

def get_file_groups(path, max_samples=1024):
  # change this depending on file structure of generated samples
  files = glob.glob(os.path.join(path, '*.mid'), recursive=True)
  assert len(files), f"provided directory was empty: {path}"

  samples = sorted(files)
  origs = sorted([os.path.join(path, 'ground_truth', os.path.basename(file)) for file in files])
  pairs = list(zip(origs, samples))

  pairs = list(filter(lambda pair: os.path.exists(pair[0]), pairs))
  if max_samples > 0:
    pairs = pairs[:max_samples]

  groups = dict()
  for orig, sample in pairs:
    sample_id = get_group_id(sample)
    orig_id = get_group_id(orig)
    assert sample_id == orig_id, f"Sample id doesn't match original id: {sample} and {orig}"
    if sample_id not in groups:
      groups[sample_id] = list()
    groups[sample_id].append((orig, sample))

  return list(groups.values())

def read_file(file):
  with open(file, 'r') as f:
    events = f.read().split('\n')
    events = [e for e in events if e]
    return events

def get_chord_groups(desc):
  bars = [1 if 'Bar_' in item else 0 for item in desc]
  bar_ids = np.cumsum(bars) - 1
  groups = [[] for _ in range(bar_ids[-1] + 1)]
  for i, item in enumerate(desc):
    if 'Chord_' in item:
      chord = item.split('_')[-1]
      groups[bar_ids[i]].append(chord)
  return groups

def instruments(events):
  insts = [128 if item.instrument == 'drum' else int(item.instrument) for item in events[1:-1] if item.name == 'Note']
  insts = np.bincount(insts, minlength=129)
  return (insts > 0).astype(int)

def chords(events):
  chords = [CHORDS.index(item) for item in events]
  chords = np.bincount(chords, minlength=129)
  return (chords > 0).astype(int)

def chroma(events):
  pitch_classes = [item.pitch % 12 for item in events[1:-1] if item.name == 'Note' and item.instrument != 'drum']
  if len(pitch_classes):
    count = np.bincount(pitch_classes, minlength=12)
    count = count / np.sqrt(np.sum(count ** 2))
  else:
    count = np.array([1/12] * 12)
  return count

def groove(events, start=0, pos_per_bar=48, ticks_per_bar=1920):
  flags = np.linspace(start, start + ticks_per_bar, pos_per_bar, endpoint=False)
  onsets = [item.start for item in events[1:-1] if item.name == 'Note']
  positions = [np.argmin(np.abs(flags - beat)) for beat in onsets]
  if len(positions):
    count = np.bincount(positions, minlength=pos_per_bar)
    count = np.convolve(count, [1, 4, 1], 'same')
    count = count / np.sqrt(np.sum(count ** 2))
  else:
    count = np.array([1/pos_per_bar] * pos_per_bar)
  return count

def multi_class_accuracy(y_true, y_pred):
  tp = ((y_true == 1) & (y_pred == 1)).sum()
  p = tp / y_pred.sum()
  r = tp / y_true.sum()
  if p + r > 0:
    f1 = 2*p*r / (p + r)
  else:
    f1 = 0
  return p, r, f1

def cross_entropy(p_true, p_pred, eps=1e-8):
  return -np.sum(p_true * np.log(p_pred + eps)) / len(p_true)

def kl_divergence(p_true, p_pred, eps=1e-8):
  return np.sum(p_true * (np.log(p_true + eps) - np.log(p_pred + eps))) / len(p_true)

def cosine_sim(p_true, p_pred):
  return np.sum(p_true * p_pred)

def sliding_window_metrics(items, start, end, window=1920, step=480, ticks_per_beat=480):
  glob_start, glob_end = start, end
  notes = [item for item in items if item.name == 'Note']
  starts = np.arange(glob_start, glob_end - window, step=step)

  groups = []
  start_idx, end_idx = 0, 0
  for start in starts:
    while notes[start_idx].start < start:
      start_idx += 1
    while end_idx < len(notes) and notes[end_idx].start < start + window:
      end_idx += 1

    groups.append([start] + notes[start_idx:end_idx] + [start + window])
  return groups

def meta_stats(group, ticks_per_beat=480):
  start, end = group[0], group[-1]
  ns = [item for item in group[1:-1] if item.name == 'Note']
  ns_ = [note for note in ns if note.instrument != 'drum']
  pitches = [note.pitch for note in ns_]
  vels = [note.velocity for note in ns_]
  durs = [(note.end - note.start) / ticks_per_beat for note in ns_]

  return {
    'note_density': len(ns) / ((end - start) / ticks_per_beat),
    'pitch_mean': np.mean(pitches) if len(pitches) else np.nan,
    'velocity_mean': np.mean(vels) if len(vels) else np.nan,
    'duration_mean': np.mean(durs) if len(durs) else np.nan,
    'pitch_std': np.std(pitches) if len(pitches) else np.nan,
    'velocity_std': np.std(vels) if len(vels) else np.nan,
    'duration_std': np.std(durs) if len(durs) else np.nan,
  }

def overlapping_area(mu1, sigma1, mu2, sigma2, eps=0.01):
  sigma1, sigma2 = max(eps, sigma1), max(eps, sigma2)
  return NormalDist(mu=mu1, sigma=sigma1).overlap(NormalDist(mu=mu2, sigma=sigma2))



def main():
  args = parse_args()
  file_groups = get_file_groups(args.samples_dir, max_samples=args.max_samples)

  metrics = pd.DataFrame()
  for sample_id, group in enumerate(file_groups):

    micro_metrics = pd.DataFrame()
    for orig_file, sample_file in group:
      print(f"[info] Group {sample_id+1}/{len(file_groups)} | original: {orig_file} | sample: {sample_file}")
      orig = ir.InputRepresentation(orig_file)
      sample = ir.InputRepresentation(sample_file)

      orig_desc, sample_desc = orig.get_description(), sample.get_description()
      if len(orig_desc) == 0 or len(sample_desc) == 0:
        print("[warning] empty sample! skipping")
        continue

      chord_groups1 = get_chord_groups(orig_desc)
      chord_groups2 = get_chord_groups(sample_desc)

      note_density_gt = []

      for g1, g2, cg1, cg2 in zip(orig.groups, sample.groups, chord_groups1, chord_groups2):
        row = pd.DataFrame([{ 'id': sample_id, 'original': orig_file, 'sample': sample_file }])

        meta1, meta2 = meta_stats(g1, ticks_per_beat=orig.pm.resolution), meta_stats(g2, ticks_per_beat=sample.pm.resolution)
        row['pitch_oa'] = overlapping_area(meta1['pitch_mean'], meta1['pitch_std'], meta2['pitch_mean'], meta2['pitch_std'])
        row['velocity_oa'] = overlapping_area(meta1['velocity_mean'], meta1['velocity_std'], meta2['velocity_mean'], meta2['velocity_std'])
        row['duration_oa'] = overlapping_area(meta1['duration_mean'], meta1['duration_std'], meta2['duration_mean'], meta2['duration_std'])
        row['note_density_abs_err'] = np.abs(meta1['note_density'] - meta2['note_density'])
        row['mean_pitch_abs_err'] = np.abs(meta1['pitch_mean'] - meta2['pitch_mean'])
        row['mean_velocity_abs_err'] = np.abs(meta1['velocity_mean'] - meta2['velocity_mean'])
        row['mean_duration_abs_err'] = np.abs(meta1['duration_mean'] - meta2['duration_mean'])
        note_density_gt.append(meta1['note_density'])

        ts1, ts2 = orig._get_time_signature(g1[0]), sample._get_time_signature(g2[0])
        ts1, ts2 = f"{ts1.numerator}/{ts1.denominator}", f"{ts2.numerator}/{ts2.denominator}"
        row['time_sig_acc'] = 1 if ts1 == ts2 else 0

        inst1, inst2 = instruments(g1), instruments(g2)
        prec, rec, f1 = multi_class_accuracy(inst1, inst2)
        row['inst_prec'] = prec
        row['inst_rec'] = rec
        row['inst_f1'] = f1

        chords1, chords2 = chords(cg1), chords(cg2)
        prec, rec, f1 = multi_class_accuracy(chords1, chords2)
        row['chord_prec'] = prec
        row['chord_rec'] = rec
        row['chord_f1'] = f1

        c1, c2 = chroma(g1), chroma(g2)
        row['chroma_crossent'] = cross_entropy(c1, c2)
        row['chroma_kldiv'] = kl_divergence(c1, c2)
        row['chroma_sim'] = cosine_sim(c1, c2)

        ppb = max(orig._get_positions_per_bar(g1[0]), sample._get_positions_per_bar(g2[0]))
        tpb = max(orig._get_ticks_per_bar(g1[0]), sample._get_ticks_per_bar(g2[0]))
        r1 = groove(g1, start=g1[0], pos_per_bar=ppb, ticks_per_bar=tpb)
        r2 = groove(g2, start=g2[0], pos_per_bar=ppb, ticks_per_bar=tpb)
        row['groove_crossent'] = cross_entropy(r1, r2)
        row['groove_kldiv'] = kl_divergence(r1, r2)
        row['groove_sim'] = cosine_sim(r1, r2)

        micro_metrics = pd.concat([micro_metrics, row], ignore_index=True)
    if len(micro_metrics) == 0:
      continue

    nd_mean = np.mean(note_density_gt)
    micro_metrics['note_density_nsq_err'] = micro_metrics['note_density_abs_err']**2 / nd_mean**2

    metrics = pd.concat([metrics, micro_metrics], ignore_index=True)

    micro_avg = micro_metrics.mean(numeric_only=True)
    print("[info] Group {}: inst_f1={:.2f} | chord_f1={:.2f} | pitch_oa={:.2f} | vel_oa={:.2f} | dur_oa={:.2f} | chroma_sim={:.2f} | groove_sim={:.2f}".format(
      sample_id+1, micro_avg['inst_f1'], micro_avg['chord_f1'], micro_avg['pitch_oa'], micro_avg['velocity_oa'], micro_avg['duration_oa'], micro_avg['chroma_sim'], micro_avg['groove_sim']
    ))
  
  os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
  metrics.to_csv(args.output_file)

  summary_keys = ['inst_f1', 'chord_f1', 'time_sig_acc', 'pitch_oa', 'velocity_oa', 'duration_oa', 'chroma_sim', 'groove_sim']
  summary = metrics[summary_keys + ['id']].groupby('id').mean().mean()

  nsq_err = metrics.groupby('id')['note_density_nsq_err'].mean()
  summary['note_density_nrmse'] = np.sqrt(nsq_err).mean()

  print('***** SUMMARY *****')
  print(summary)

if __name__ == '__main__':
  main()
