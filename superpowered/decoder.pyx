import numpy as np
cimport numpy as np
from enum import Enum
from .simple import short_int_to_float


np.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef extern from "util.cc":

    int _read(SuperpoweredDecoder *decoder, float **sampleBuffer,
           long &startSample, long &endSample, bint precise,
           long &numSamples, long &maxNumSamples, 
           float rateShift, int pitchShift, int decibel)

    int _analyze(SuperpoweredDecoder *decoder, 
              long &startSample, long &endSample, bint precise, int decibel,
              unsigned char **averageWaveform,
              unsigned char **peakWaveform, unsigned char **lowWaveform,
              unsigned char **midWaveform, unsigned char **highWaveform,
              unsigned char **notes, int *waveformSize, char **overviewWaveform,
              int *overviewSize, float *averageDecibel,
              float *loudpartsAverageDecibel, float *peakDecibel, float *bpm,
              float *beatgridStartMs, int *keyIndex)


cdef data_to_numpy_array_float32(float *data, np.npy_intp size):
    cdef np.ndarray[float, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT32, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

cdef data_to_numpy_array_uint8(unsigned char *data, np.npy_intp size):
    cdef np.ndarray[unsigned char, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr

cdef data_to_numpy_array_int8(char *data, np.npy_intp size):
    cdef np.ndarray[char, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_INT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


class DecoderType(Enum):
    MP3         = 0
    AAC         = 1
    AIFF        = 2
    WAV         = 3
    MediaServer = 4


class DecoderError(Exception):
    pass


class DecoderEOF(Exception):
    pass


cdef class Decoder:
    """
     @brief Audio file decoder. Provides uncompresses PCM samples from various compressed formats.

     Thread safety: single threaded, not thread safe. After a succesful open(), samplePosition and duration may change.
     Supported file types:
     - Stereo or mono pcm WAV and AIFF (16-bit int, 24-bit int, 32-bit int or 32-bit IEEE float).
     - MP3 (all kind).
     - AAC-LC in M4A container (iTunes).
     - AAC-LC in ADTS container (.aac).
     - Apple Lossless (on iOS only).

     @param durationSeconds The duration of the current file in seconds. Read only.
     @param durationSamples The duration of the current file in samples. Read only.
     @param samplePosition The current position in samples. May change after each decode() or seekTo(). Read only.
     @param samplerate The sample rate of the current file. Read only.
     @param samplesPerFrame How many samples are in one frame of the source file. For example: 1152 in mp3 files.
     @param kind The format of the current file.
    """
    # cdef SuperpoweredDecoder _decoder

    @property
    def durationSeconds(self):
        return self._decoder.durationSeconds

    @property
    def durationSamples(self):
        return self._decoder.durationSamples

    @property
    def samplePosition(self):
        return self._decoder.samplePosition

    @property
    def samplerate(self):
        return self._decoder.samplerate

    @property
    def samplesPerFrame(self):
        return self._decoder.samplesPerFrame

    @property
    def kind(self):
        return DecoderType(self._decoder.kind)

    def open(self, str path, bint metaOnly = False, int offset = 0, int length = 0, int stemsIndex = 0):
        """
         @brief Opens a file for decoding.
         
         @param path Full file system path or progressive download path (http or https).
         @param metaOnly If true, it opens the file for fast metadata reading only, not for decoding audio. Available for fully available local files only (no network access).
         @param offset Byte offset in the file.
         @param length Byte length from offset. Set offset and length to 0 to read the entire file.
         @param stemsIndex Stems track index for Native Instruments Stems format.
         @param customHTTPHeaders NULL terminated list of custom headers for http communication.

         @return NULL if successful, or an error string.
        """
        cdef const char *openError
        openError = self._decoder.open(bytes(path, 'utf-8'), metaOnly, offset, length, stemsIndex)
        if openError is not NULL:
            raise IOError(openError.decode('utf-8'))

    def decode(self, np.ndarray[short int, ndim=2, mode='c'] pcmOutput, unsigned int samples):
        """
         @brief Decodes the requested number of samples.

         @return End of file (0), ok (1) or error (2).

         @param pcmOutput The buffer to put uncompressed audio. Must be at least this big: (*samples * 4) + 16384 bytes.
         @param samples On input, the requested number of samples. Should be >= samplesPerFrame. On return, the samples decoded.
        """
        cdef unsigned int code

        assert pcmOutput.shape[1] == 2, 'pcmOutput should be a stereo interleaved buffer'
        assert samples >= self.samplesPerFrame, 'samples requested should be >= samplesPerFrame'
        assert pcmOutput.nbytes >= (samples * 4) + 16384, 'Buffer must be at least this big: (*samples * 4) + 16384 bytes'
        
        code = self._decoder.decode(&pcmOutput[0,0], &samples)

        if code == SUPERPOWEREDDECODER_EOF:
            raise DecoderEOF

        if code == SUPERPOWEREDDECODER_ERROR:
            raise DecoderError

        return samples  # SUPERPOWEREDDECODER_OK

    def seek(self, long sample, bint precise):
        """
         @brief Jumps to a specific position.

         @return The new position.

         @param sample The position (a sample index).
         @param precise Some codecs may not jump precisely due internal framing. Set precise to true if you want exact positioning (for a little performance penalty of 1 memmove).
        """
        return self._decoder.seek(sample, precise)

    def getAudioStartSample(self, unsigned int limitSamples = 0, int decibel = 0):
        """
         @return End of file (0), ok (1), error (2) or buffering(3). This function changes position!
         
         @param limitSamples How far to search for. 0 means "the entire audio file".
         @param decibel Optional loudness threshold in decibel. 0 means "any non-zero audio sample". The value -49 is useful for vinyl rips.
         @param startSample Returns with the position where audio starts.
        """
        cdef unsigned int startSample
        retcode = self._decoder.getAudioStartSample(&startSample, limitSamples, decibel)
        if retcode == 2:
            raise IOError('could not get audio start sample')
        return startSample

    def __iter__(self):
        return DecoderIterator(self)

    def read(self):
        """
         Read all data as float
        """
        pcmOutput = np.empty((self.durationSamples + 4096, 2), dtype=np.int16)

        try:
            self.decode(pcmOutput, self.durationSamples)
        except DecoderEOF:  # all OK
            pass

        pcmOutput = pcmOutput[:self.durationSamples]

        samples = np.empty_like(pcmOutput, dtype=np.float32)
        short_int_to_float(pcmOutput, samples)

        return samples


class DecoderIterator:

    def __init__(self, decoder):
        self.decoder = decoder
        # initialize buffer, with extra scratch space at the end
        self.intBuffer = np.empty((decoder.samplesPerFrame + 4096, 2), dtype=np.int16)  # assumes stereo
        # decoder.audioStartSample()  # seek start

    def __next__(self):
        try:
            samplesDecoded = self.decoder.decode(self.intBuffer, self.decoder.samplesPerFrame)
            return self.intBuffer[:samplesDecoded,:]
        except DecoderEOF:
            raise StopIteration
