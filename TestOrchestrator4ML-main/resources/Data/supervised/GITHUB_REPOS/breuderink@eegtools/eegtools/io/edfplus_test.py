# -*- coding: utf-8 -*-
import os.path
import numpy as np
from eegtools.io import edfplus


def test_with_generate_edf():
  '''Test EDF reader using artificial EDF dataset. Note that this is not an
  EDF+ dataset and as such does not contain annotations. Annotations decoding
  is separately tested, *but not from a real file*!.
  '''
  reader = edfplus.BaseEDFReader(
    open(os.path.join(os.path.dirname(__file__), 
    'data', 'sine3Hz_block0.2Hz.edf'), 'rb'))
  reader.read_header()

  h = reader.header
  # check important header fields
  assert h['label'] == ['3Hz +5/-5 V', '0.2Hz Blk 1/0uV']
  assert h['units'] == ['V', 'uV']
  assert h['contiguous'] == True
  assert h['n_header_bytes'] == 768
  
  assert reader.bytes_per_record == 2256

  fs = np.asarray(h['n_samples_per_record']) / h['record_length']

  # get records
  recs = list(reader.records())
  time = zip(*recs)[0]
  signals = zip(*recs)[1]
  annotations = list(zip(*recs)[2])

  # check EDF+ fields that are *not present in this file*
  np.testing.assert_equal(time, np.zeros(11) * np.nan)
  assert annotations == [[]] * 11

  # check 3 Hz sine wave
  sine, block = [np.hstack(s) for s in zip(*signals)]
  target = 5 * np.sin(3 * np.pi * 2 * np.arange(0, sine.size) / fs[0])
  assert np.max((sine - target) ** 2) < 1e-4
  assert len(sine) == 110 * 100

  # check .2 Hz block wave
  target = np.sin(.2 * np.pi * 2 * np.arange(1, block.size + 1) / fs[1]) >= 0
  assert np.max((block - target) ** 2) < 1e-4
  assert len(block) == 110 * 12.8
    
  # check again with partial read - need to reopen file to reset cursor 
  partial_reader = edfplus.BaseEDFReader(
    open(os.path.join(os.path.dirname(__file__), 
    'data', 'sine3Hz_block0.2Hz.edf'), 'rb'))
  partial_reader.read_header()
  partial_recs = list(partial_reader.select_records(offset=2, amount=1))
  partial_signals = zip(*partial_recs)[1]
  partial_sine, partial_block = [np.hstack(s) for s in zip(*partial_signals)]
  assert len(partial_sine) == 10 * 100
  assert len(partial_block) == 10 * 12.8
  

def test_edfplus_tal():
  mult_annotations = '+180\x14Lights off\x14Close door\x14\x00'
  with_duration = '+1800.2\x1525.5\x14Apnea\x14\x00'
  test_unicode = '+180\x14€\x14\x00\x00'
  int_time = '+1800\x1525\x14Apnea\x14\x00'

  # test annotation with duration
  assert edfplus.tal(with_duration) == [(1800.2, 25.5, [u'Apnea'])]
  assert edfplus.tal(int_time) == [(1800, 25, [u'Apnea'])]

  # test multiple annotations
  assert edfplus.tal('\x00' * 4 + with_duration * 3) == \
    [(1800.2, 25.5, [u'Apnea'])] * 3

  # test multiple annotations for one time point
  assert edfplus.tal(mult_annotations) == \
    [(180., 0., [u'Lights off', u'Close door'])]

  # test unicode support
  assert edfplus.tal(test_unicode) == [(180., 0., [u'€'])]


def test_load_edf():
  '''Test high-level edf-loading interface.'''
  fname = os.path.join(os.path.dirname(__file__), 
    'data', 'kemp-positive_spikes.edf')
  X, sample_rate, sens_lab, time, annotations = edfplus.load_edf(fname)
  np.testing.assert_almost_equal(np.flatnonzero(X), np.arange(12) * 100)

  assert X.shape == (1, 1200)
  assert X.dtype == np.float32
  assert sample_rate == 100.
