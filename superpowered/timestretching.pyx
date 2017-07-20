import cython
import numpy as np
cimport numpy as np
from .audiobuffers cimport AudiobufferlistElement, AudiopointerList
from .audiobuffers cimport SuperpoweredAudiobufferlistElement, SuperpoweredAudiopointerList


np.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef extern from "SuperpoweredTimeStretching.h":

    cdef cppclass SuperpoweredTimeStretching:
        SuperpoweredTimeStretching(unsigned int samplerate, float minimumRate) except +

        float rate
        int pitchShift
        int pitchShiftCents
        int numberOfInputSamplesNeeded

        bint setRateAndPitchShift(float newRate, int newShift)
        bint setRateAndPitchShiftCents(float newRate, int newShiftCents)
        void setStereoPairs(unsigned int numStereoPairs)
        void setSampleRate(unsigned int samplerate)
        void reset()
        void removeSamplesFromInputBuffersEnd(unsigned int samples)
        void process(SuperpoweredAudiobufferlistElement *input, SuperpoweredAudiopointerList *outputList)


cdef extern from "stretch.cc":

    void _stretch(float *sampleBuffer, long numSamples, unsigned int samplerate,
                  float rateShift, int pitchShift, float **outBuffer, 
                  long &numSamplesOut, long &maxNumSamples)


cdef data_to_numpy_array_float32(float *data, np.npy_intp size):
    cdef np.ndarray[float, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT32, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


cdef class TimeStretching:
    cdef SuperpoweredTimeStretching *_time_stretching

    def __cinit__(self, unsigned int samplerate, float minimumRate = 0.0):
        self._time_stretching = new SuperpoweredTimeStretching(samplerate, minimumRate)

    def __dealloc__(self):
        del self._time_stretching

    def setRateAndPitchShift(self, float newRate, int newShift):
        self._time_stretching.setRateAndPitchShift(newRate, newShift)

    def setSampleRate(self, unsigned int samplerate):
        self._time_stretching.setSampleRate(samplerate)

    def process(self, AudiobufferlistElement input, AudiopointerList outputList):
        self._time_stretching.process(&(input._audiobuffer_list_element), outputList._audiopointer_list)


def stretch(np.ndarray[float, ndim=2, mode='c'] samples, unsigned int samplerate, 
            float rateShift, int pitchShift):
    cdef float *outBuffer = NULL
    cdef long numSamplesOut = 0, maxNumSamples = 0
    cdef long numSamples = samples.shape[0]
    assert samples.shape[1] == 2, 'input is not a stereo interleaved buffer'
    _stretch(&samples[0,0], numSamples, samplerate,
        rateShift, pitchShift, &outBuffer, numSamplesOut, maxNumSamples)
    return data_to_numpy_array_float32(outBuffer, numSamplesOut * 2).reshape(numSamplesOut, 2)
