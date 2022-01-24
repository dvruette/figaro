import numpy as np

class MIDIChord(object):
  def __init__(self, pm):
    self.pm = pm
    # define pitch classes
    self.PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # define chord maps (required)
    self.CHORD_MAPS = {'maj': [0, 4],
                        'min': [0, 3],
                        'dim': [0, 3, 6],
                        'aug': [0, 4, 8],
                        'dom7': [0, 4, 10],
                        'maj7': [0, 4, 11],
                        'min7': [0, 3, 10]}
    # define chord insiders (+10)
    self.CHORD_INSIDERS = {'maj': [7],
                            'min': [7],
                            'dim': [9],
                            'aug': [],
                            'dom7': [7],
                            'maj7': [7],
                            'min7': [7]}
    # define chord outsiders (-1)
    self.CHORD_OUTSIDERS_1 = {'maj': [2, 5, 9],
                              'min': [2, 5, 8],
                              'dim': [2, 5, 10],
                              'aug': [2, 5, 9],
                              'dom7': [2, 5, 9],
                              'maj7': [2, 5, 9],
                              'maj7': [2, 5, 9],
                              'min7': [2, 5, 8]}
    # define chord outsiders (-2)
    self.CHORD_OUTSIDERS_2 = {'maj': [1, 3, 6, 8, 10, 11],
                              'min': [1, 4, 6, 9, 11],
                              'dim': [1, 4, 7, 8, 11],
                              'aug': [1, 3, 6, 7, 10],
                              'dom7': [1, 3, 6, 8, 11],
                              'maj7': [1, 3, 6, 8, 10],
                              'min7': [1, 4, 6, 9, 11]}

  def sequencing(self, chroma):
    candidates = {}
    for index in range(len(chroma)):
      if chroma[index]:
        root_note = index
        _chroma = np.roll(chroma, -root_note)
        sequence = np.where(_chroma == 1)[0]
        candidates[root_note] = list(sequence)
    return candidates

  def scoring(self, candidates):
    scores = {}
    qualities = {}
    for root_note, sequence in candidates.items():
      if 3 not in sequence and 4 not in sequence:
        scores[root_note] = -100
        qualities[root_note] = 'None'
      elif 3 in sequence and 4 in sequence:
        scores[root_note] = -100
        qualities[root_note] = 'None'
      else:
        # decide quality
        if 3 in sequence:
          if 6 in sequence:
            quality = 'dim'
          else:
            if 10 in sequence:
              quality = 'min7'
            else:
              quality = 'min'
        elif 4 in sequence:
          if 8 in sequence:
            quality = 'aug'
          else:
            if 10 in sequence:
              quality = 'dom7'
            elif 11 in sequence:
              quality = 'maj7'
            else:
              quality = 'maj'
        # decide score
        maps = self.CHORD_MAPS.get(quality)
        _notes = [n for n in sequence if n not in maps]
        score = 0
        for n in _notes:
          if n in self.CHORD_OUTSIDERS_1.get(quality):
            score -= 1
          elif n in self.CHORD_OUTSIDERS_2.get(quality):
            score -= 2
          elif n in self.CHORD_INSIDERS.get(quality):
            score += 10
        scores[root_note] = score
        qualities[root_note] = quality
    return scores, qualities

  def find_chord(self, chroma, threshold=10):
      chroma = np.sum(chroma, axis=1)
      chroma = np.array([1 if c > threshold else 0 for c in chroma])
      if np.sum(chroma) == 0:
          return 'N', 'N', 'N', 10
      else:
          candidates = self.sequencing(chroma=chroma)
          scores, qualities = self.scoring(candidates=candidates)
          # bass note
          sorted_notes = []
          for i, v in enumerate(chroma):
              if v > 0:
                  sorted_notes.append(int(i%12))
          bass_note = sorted_notes[0]
          # root note
          __root_note = []
          _max = max(scores.values())
          for _root_note, score in scores.items():
              if score == _max:
                  __root_note.append(_root_note)
          if len(__root_note) == 1:
              root_note = __root_note[0]
          else:
              #TODO: what should i do
              for n in sorted_notes:
                  if n in __root_note:
                      root_note = n
                      break
          # quality
          quality = qualities.get(root_note)
          sequence = candidates.get(root_note)
          # score
          score = scores.get(root_note)
          return self.PITCH_CLASSES[root_note], quality, self.PITCH_CLASSES[bass_note], score

  def greedy(self, candidates, max_tick, min_length):
    chords = []
    # start from 0
    start_tick = 0
    while start_tick < max_tick:
      _candidates = candidates.get(start_tick)
      _candidates = sorted(_candidates.items(), key=lambda x: (x[1][-1], x[0]))
      # choose
      end_tick, (root_note, quality, bass_note, _) = _candidates[-1]
      if root_note == bass_note:
        chord = '{}:{}'.format(root_note, quality)
      else:
        chord = '{}:{}/{}'.format(root_note, quality, bass_note)
      chords.append([start_tick, end_tick, chord])
      start_tick = end_tick
    # remove :None
    temp = chords
    while ':None' in temp[0][-1]:
      try:
        temp[1][0] = temp[0][0]
        del temp[0]
      except:
        print('NO CHORD')
        return []
    temp2 = []
    for chord in temp:
      if ':None' not in chord[-1]:
        temp2.append(chord)
      else:
        temp2[-1][1] = chord[1]
    return temp2
  
  def dynamic(self, candidates, max_tick, min_length):
    # store index of best chord at each position
    chords = [None for i in range(max_tick + 1)]
    # store score of best chords at each position
    scores = np.zeros(max_tick + 1)
    scores[1:].fill(np.NINF)

    start_tick = 0
    while start_tick < max_tick:
      if start_tick in candidates:
        for i, (end_tick, candidate) in enumerate(candidates.get(start_tick).items()):
          root_note, quality, bass_note, score = candidate
          # if this candidate is best yet, update scores and chords
          if scores[end_tick] < scores[start_tick] + score:
            scores[end_tick] = scores[start_tick] + score
            if root_note == bass_note:
              chord = '{}:{}'.format(root_note, quality)
            else:
              chord = '{}:{}/{}'.format(root_note, quality, bass_note)
            chords[end_tick] = (start_tick, end_tick, chord)
      start_tick += 1
    # Read the best path
    start_tick = len(chords) - 1
    results = []
    while start_tick > 0:
      chord = chords[start_tick]
      start_tick = chord[0]
      results.append(chord)
    
    return list(reversed(results))

  def dedupe(self, chords):
    if len(chords) == 0:
      return []
    deduped = []
    start, end, chord = chords[0]
    for (curr, next) in zip(chords[:-1], chords[1:]):
      if chord == next[2]:
        end = next[1]
      else:
        deduped.append([start, end, chord])
        start, end, chord = next
    deduped.append([start, end, chord])
    return deduped

  def get_candidates(self, chroma, max_tick, intervals=[1, 2, 3, 4]):
    candidates = {}
    for interval in intervals:
      for start_beat in range(max_tick):
        # set target pianoroll
        end_beat = start_beat + interval
        if end_beat > max_tick:
          end_beat = max_tick
        _chroma = chroma[:, start_beat:end_beat]
        # find chord
        root_note, quality, bass_note, score = self.find_chord(chroma=_chroma)
        # save
        if start_beat not in candidates:
          candidates[start_beat] = {}
          candidates[start_beat][end_beat] = (root_note, quality, bass_note, score)
        else:
          if end_beat not in candidates[start_beat]:
            candidates[start_beat][end_beat] = (root_note, quality, bass_note, score)
    return candidates

  def extract(self):
    # read
    beats = self.pm.get_beats()
    chroma = self.pm.get_chroma(times=beats)
    # get lots of candidates
    candidates = self.get_candidates(chroma, max_tick=len(beats))
    
    # greedy
    chords = self.dynamic(candidates=candidates, 
                          max_tick=len(beats), 
                          min_length=1)
    chords = self.dedupe(chords)
    for chord in chords:
      chord[0] = beats[chord[0]]
      if chord[1] >= len(beats):
        chord[1] = self.pm.get_end_time()
      else:
        chord[1] = beats[chord[1]]
    return chords