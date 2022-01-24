import numpy as np

# parameters for input representation
DEFAULT_POS_PER_QUARTER = 12
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_DURATION_BINS = np.sort(np.concatenate([
  np.arange(1, 13), # smallest possible units up to 1 quarter
  np.arange(12, 24, 3)[1:], # 16th notes up to 1 bar
  np.arange(13, 24, 4)[1:], # triplets up to 1 bar
  np.arange(24, 48, 6), # 8th notes up to 2 bars
  np.arange(48, 4*48, 12), # quarter notes up to 8 bars
  np.arange(4*48, 16*48+1, 24) # half notes up to 16 bars
]))
DEFAULT_TEMPO_BINS = np.linspace(0, 240, 32+1, dtype=np.int)
DEFAULT_NOTE_DENSITY_BINS = np.linspace(0, 12, 32+1)
DEFAULT_MEAN_VELOCITY_BINS = np.linspace(0, 128, 32+1)
DEFAULT_MEAN_PITCH_BINS = np.linspace(0, 128, 32+1)
DEFAULT_MEAN_DURATION_BINS = np.logspace(0, 7, 32+1, base=2) # log space between 1 and 128 positions (~2.5 bars)

# parameters for output
DEFAULT_RESOLUTION = 480

# maximum length of a single bar is 3*4 = 12 beats
MAX_BAR_LENGTH = 3
# maximum number of bars in a piece is 512 (this covers almost all sequences)
MAX_N_BARS = 512

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
MASK_TOKEN = '<mask>'

TIME_SIGNATURE_KEY = 'Time Signature'
BAR_KEY = 'Bar'
POSITION_KEY = 'Position'
INSTRUMENT_KEY = 'Instrument'
PITCH_KEY = 'Pitch'
VELOCITY_KEY = 'Velocity'
DURATION_KEY = 'Duration'
TEMPO_KEY = 'Tempo'
CHORD_KEY = 'Chord'

NOTE_DENSITY_KEY = 'Note Density'
MEAN_PITCH_KEY = 'Mean Pitch'
MEAN_VELOCITY_KEY = 'Mean Velocity'
MEAN_DURATION_KEY = 'Mean Duration'