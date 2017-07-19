import cython
import numpy as np
cimport numpy as np


np.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef data_to_numpy_array_float32(void *data, np.npy_intp size):
    cdef np.ndarray[float, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT32, data)
    return arr


def ping():
    """
     @brief Let the system know that we are start using this class.
    """
    _ping()


def getBuffer(unsigned int numSamples):
    # assume float32 stereo buffer 
    cdef unsigned int sizeBytes = 2 * numSamples * sizeof(float)
    cdef void *data = _getBuffer(sizeBytes)
    return data_to_numpy_array_float32(data, 2 * numSamples).reshape(numSamples, 2)


cdef class AudiobufferlistElement:
    """
     @brief An audio buffer list item.

     @param buffers The buffers, coming from SuperpoweredAudiobufferPool.
     @param samplePosition The buffer beginning's sample position in an audio file or stream.
     @param startSample The first sample in the buffer.
     @param endSample The last sample in the buffer.
     @param samplesUsed How many "original" samples were used to create this chunk of audio. Good for time-stretching for example, to track the movement of the playhead.
    """
    # cdef SuperpoweredAudiobufferlistElement _audiobuffer_list_element
    # cdef int numSamples[4]

    def __cinit__(self):
        self._audiobuffer_list_element.buffers[0] = NULL
        self._audiobuffer_list_element.buffers[1] = NULL
        self._audiobuffer_list_element.buffers[2] = NULL
        self._audiobuffer_list_element.buffers[3] = NULL

    def __getitem__(self, index):
        cdef void *buffer = self._audiobuffer_list_element.buffers[index]
        if buffer == NULL:
            return None
        cdef unsigned int numSamples = self.numSamples[index]
        return data_to_numpy_array_float32(buffer, 2 * numSamples).reshape(numSamples, 2)

    def __setitem__(self, index, np.ndarray[float, ndim=2, mode='c'] buffer):
        assert buffer.shape[1] == 2, 'input is not a stereo interleaved buffer'
        self.numSamples[index] = buffer.shape[0]
        self._audiobuffer_list_element.buffers[index] = &buffer[0,0]

    @property
    def samplePosition(self):
        return self._audiobuffer_list_element.samplePosition

    @samplePosition.setter
    def samplePosition(self, np.int64_t samplePosition):
        self._audiobuffer_list_element.samplePosition = samplePosition

    @property
    def startSample(self):
        return self._audiobuffer_list_element.startSample

    @startSample.setter
    def startSample(self, int startSample):
        self._audiobuffer_list_element.startSample = startSample

    @property
    def endSample(self):
        return self._audiobuffer_list_element.endSample

    @endSample.setter
    def endSample(self, int endSample):
        self._audiobuffer_list_element.endSample = endSample

    @property
    def samplesUsed(self):
        return self._audiobuffer_list_element.samplesUsed

    @samplesUsed.setter
    def samplesUsed(self, float samplesUsed):
        self._audiobuffer_list_element.samplesUsed = samplesUsed


cdef class AudiopointerList:
    # cdef SuperpoweredAudiopointerList *_audiopointer_list

    def __cinit__(self, unsigned int bytesPerSample, unsigned int typicalNumElements):
        self._audiopointer_list = new SuperpoweredAudiopointerList(bytesPerSample, typicalNumElements)

    def __dealloc__(self):
        del self._audiopointer_list

    @property
    def sampleLength(self):
        return self._audiopointer_list.sampleLength

    def append(self, AudiobufferlistElement buffer):
        # cdef SuperpoweredAudiobufferlistElement *ptr = &(buffer._audiobuffer_list_element)
        self._audiopointer_list.append(&(buffer._audiobuffer_list_element))

    def clear(self):
        self._audiopointer_list.clear()

    def makeSlice(self, int fromSample, int lengthSamples):
        return self._audiopointer_list.makeSlice(fromSample, lengthSamples)

    def nextSliceItem(self):
        cdef int lengthSamples = 0
        cdef void *data = self._audiopointer_list.nextSliceItem(&lengthSamples)
        if data == NULL:
            return None
        return data_to_numpy_array_float32(data, 2 * lengthSamples).reshape(lengthSamples, 2)
