import numpy as np
cimport numpy as np


np.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef extern from "SuperpoweredAnalyzer.h":
    float frequencyOfNote(int note)

    cdef cppclass SuperpoweredOfflineAnalyzer:
        SuperpoweredOfflineAnalyzer(unsigned int samplerate, float bpm, int lengthSeconds, 
            float minimumBpm, float maximumBpm) except +

        void process(float *input, unsigned int numberOfSamples)
        void getresults(unsigned char **averageWaveform, unsigned char **peakWaveform,
            unsigned char **lowWaveform, unsigned char **midWaveform, unsigned char **highWaveform,
            unsigned char **notes, int *waveformSize, char **overviewWaveform, int *overviewSize,
            float *averageDecibel, float *loudpartsAverageDecibel, float *peakDecibel, float *bpm,
            float *beatgridStartMs, int *keyIndex)

    cdef cppclass SuperpoweredWaveform:
        SuperpoweredWaveform(unsigned int samplerate, int lengthSeconds) except +

        void process(float *input, unsigned int numberOfSamples)
        unsigned char *getresult(int *size)


cdef data_to_numpy_array_uint8(unsigned char *data, np.npy_intp size):
    cdef np.ndarray[unsigned char, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_UINT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


cdef data_to_numpy_array_int8(char *data, np.npy_intp size):
    cdef np.ndarray[char, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_INT8, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


musicalChordNames = [
    "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#",                # major
    "Am", "A#m", "Bm", "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m"     # minor
]

camelotChordNames = [
    "11B", "6B", "1B", "8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B",      # major
    "8A", "3A", "10A", "5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A"       # minor
]

openkeyChordNames = [
    "4d", "11d", "6d", "1d", "8d", "3d", "10d", "5d", "12d", "7d", "2d", "9d",      # major
    "1m", "8m", "3m", "10m", "5m", "12m", "7m", "2m", "9m", "4m", "11m", "6m"       # minor
]

chordToSyllable = [
    5, 5, 6, 0, 0, 1, 1, 2, 3, 3, 4, 4,
    5, 5, 6, 0, 0, 1, 1, 2, 3, 3, 4, 4,
]

chordToNoteStartingFromC = [
    9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8,
]

camelotSort = [
    21, 11, 1, 15, 5, 19, 9, 23, 13, 3, 17, 7,
    14, 4, 18, 8, 22, 12, 2, 16, 6, 20, 10, 0
]

SUPERPOWERED_WAVEFORM_POINTS_PER_SEC = 150


cdef class OfflineAnalyzer:
    """
     @brief Performs bpm and key detection, loudness/peak analysis. Provides compact waveform data (150 points/sec and 1 point/sec resolution), beatgrid information.
    """
    cdef SuperpoweredOfflineAnalyzer *_analyzer

    def __cinit__(self, unsigned int samplerate, float bpm = 0, int lengthSeconds = 0, float minimumBpm = 60.0, float maximumBpm = 200.0):
        self._analyzer = new SuperpoweredOfflineAnalyzer(samplerate, bpm, lengthSeconds, minimumBpm, maximumBpm)

    def __dealloc__(self):
        del self._analyzer

    def process(self, np.ndarray[float, ndim=2, mode='c'] input):
        """
         @brief Processes a chunk of audio.

         @param input 32-bit interleaved floating-point input.
         @param numberOfSamples How many samples to process.
         @param lengthSeconds If the source's length may change, set this to it's current value, otherwise leave it at -1.
        """
        cdef unsigned int numberOfSamples = input.shape[0]
        assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
        self._analyzer[0].process(&input[0,0], numberOfSamples)

    def getresults(self):
        """
         @brief Get results. Call this method ONCE, after all samples are processed.
         
         @param averageWaveform 150 points/sec waveform data displaying the average volume. Each sample is an unsigned char from 0 to 255. You take ownership on this (must free memory).
         @param peakWaveform 150 points/sec waveform data displaying the peak volume. Each sample is an unsigned char from 0 to 255. You take ownership on this (must free memory).
         @param lowWaveform 150 points/sec waveform data displaying the low frequencies. Each sample is an unsigned char from 0 to 255. You take ownership on this (must free memory).
         @param midWaveform 150 points/sec waveform data displaying the mid frequencies. Each sample is an unsigned char from 0 to 255. You take ownership on this (must free memory).
         @param highWaveform 150 points/sec waveform data displaying the high frequencies. Each sample is an unsigned char from 0 to 255. You take ownership on this (must free memory).
         @param notes 150 points/sec data displaying the bass and mid keys. Upper 4 bits are the bass notes 0 to 11, lower 4 bits are the mid notes 0 to 11 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). The note value is 12 means "unknown note due low volume". You take ownership on this (must free memory).
         @param waveformSize The number of points in averageWaveform, peakWaveform or lowMidHighWaveform.
         @param overviewWaveform 1 point/sec waveform data displaying the average volume in decibels. Useful for displaying the overall structure of a track. Each sample is a signed char, from -128 to 0 decibel. You take ownership on this (must free memory).
         @param overviewSize The number points in overviewWaveform.
         @param averageDecibel The average loudness of all samples processed in decibel.
         @param loudpartsAverageDecibel The average loudness of the "loud" parts in the music in decibel. (Breakdowns and other quiet parts are excluded.)
         @param peakDecibel The loudest sample in decibel.
         @param bpm Beats per minute.
         @param beatgridStartMs The position where the beatgrid should start. Important! On input set it to 0, or the ms position of the first audio sample.
         @param keyIndex The dominant key (chord) of the music. 0..11 are major keys from A to G#, 12..23 are minor keys from A to G#. Check the static constants in this header for musical, Camelot and Open Key notations.
        """
        cdef unsigned char *averageWaveform = NULL
        cdef unsigned char *peakWaveform = NULL
        cdef unsigned char *lowWaveform = NULL
        cdef unsigned char *midWaveform = NULL
        cdef unsigned char *highWaveform = NULL
        cdef unsigned char *notes = NULL
        cdef int waveformSize
        cdef char *overviewWaveform = NULL
        cdef int overviewSize
        cdef float averageDecibel = 0
        cdef float loudpartsAverageDecibel = 0
        cdef float peakDecibel = 0
        cdef float bpm = 0
        cdef float beatgridStartMs = 0
        cdef int keyIndex

        self._analyzer[0].getresults(&averageWaveform, &peakWaveform, &lowWaveform, &midWaveform, &highWaveform, 
            &notes, &waveformSize, &overviewWaveform, &overviewSize, &averageDecibel, &loudpartsAverageDecibel, 
            &peakDecibel, &bpm, &beatgridStartMs, &keyIndex)

        return {
            'averageWaveform' : data_to_numpy_array_uint8(averageWaveform, waveformSize),
            'peakWaveform' : data_to_numpy_array_uint8(peakWaveform, waveformSize),
            'lowWaveform' : data_to_numpy_array_uint8(lowWaveform, waveformSize),
            'midWaveform' : data_to_numpy_array_uint8(midWaveform, waveformSize),
            'highWaveform' : data_to_numpy_array_uint8(highWaveform, waveformSize),
            'notes' : data_to_numpy_array_uint8(notes, waveformSize),
            'waveformSize' : waveformSize,
            'overviewWaveform' : data_to_numpy_array_int8(overviewWaveform, overviewSize),
            'overviewSize' : overviewSize,
            'averageDecibel' : averageDecibel,
            'loudpartsAverageDecibel' : loudpartsAverageDecibel,
            'peakDecibel' : peakDecibel,
            'bpm' : bpm,
            'beatgridStartMs' : beatgridStartMs,
            'keyIndex' : keyIndex,
        }
