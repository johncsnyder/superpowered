import cython
import numpy as np
cimport numpy as np


cdef extern from "SuperpoweredAudioBuffers.h":
    ctypedef struct bufferPoolInternals:
        pass

    ctypedef struct pointerListInternals:
        pass

    ctypedef struct SuperpoweredAudiobufferlistElement:
        void *buffers[4]
        np.int64_t samplePosition
        int startSample, endSample
        float samplesUsed

    cdef void _ping "SuperpoweredAudiobufferPool::ping"() 
    cdef void *_getBuffer "SuperpoweredAudiobufferPool::getBuffer"(unsigned int sizeBytes) 
    cdef void *_allocBuffer "SuperpoweredAudiobufferPool::allocBuffer"(unsigned int sizeBytes) 
    cdef void _releaseBuffer "SuperpoweredAudiobufferPool::releaseBuffer"(void *buffer)
    cdef void _retainBuffer "SuperpoweredAudiobufferPool::retainBuffer"(void *buffer)

    cdef cppclass SuperpoweredAudiopointerList:
        SuperpoweredAudiopointerList(unsigned int bytesPerSample, unsigned int typicalNumElements) except +

        int sampleLength

        void append(SuperpoweredAudiobufferlistElement *buffer)
        void insert(SuperpoweredAudiobufferlistElement *buffer)
        void clear()
        void copyAllBuffersTo(SuperpoweredAudiopointerList *anotherList)
        void truncate(int numSamples, bint fromTheBeginning)
        np.int64_t startSamplePosition()
        np.int64_t nextSamplePosition()
        bint makeSlice(int fromSample, int lengthSamples)
        np.int64_t samplePositionOfSliceBeginning()
        void *nextSliceItem(int *lengthSamples, float *samplesUsed = 0, int stereoPairIndex = 0)
        void rewindSlice()
        void forwardToLastSliceBuffer()
        void *prevSliceItem(int *lengthSamples, float *samplesUsed, int stereoPairIndex)


cdef class AudiobufferlistElement:
    cdef SuperpoweredAudiobufferlistElement _audiobuffer_list_element
    cdef int numSamples[4]


cdef class AudiopointerList:
    cdef SuperpoweredAudiopointerList *_audiopointer_list

