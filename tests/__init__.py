from nose.tools import *
import numpy as np
import superpowered
from superpowered.decoder import Decoder


def test_decoder():
    decoder = Decoder()
    decoder.open('./example/spotify-track-1Ew3CCat8okts5yG4I5LpA.mp3')
    samples = decoder.read()
    assert samples.shape == (decoder.durationSamples, 2)
    assert samples.dtype == np.float32


def test_decoder_1():
    decoder = Decoder()
    with assert_raises_regexp(IOError, "Can't open this file"):
        decoder.open('./example/missing.mp3')


# def test_filter():
#     decoder = Decoder()
#     decoder.open('./resources/spotify-track-1Ew3CCat8okts5yG4I5LpA.mp3')
#     samples = decoder.read()

#     assert samples.shape == (decoder.durationSamples, 2)
#     assert samples.dtype == np.float32
