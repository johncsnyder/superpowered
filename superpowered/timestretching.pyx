import cython
import numpy as np
cimport numpy as np
from .audiobuffers cimport AudiobufferlistElement, AudiopointerList
from .audiobuffers cimport SuperpoweredAudiobufferlistElement, SuperpoweredAudiopointerList


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
