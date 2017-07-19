from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np


np.import_array()


cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cdef extern from "util.cc":

    void _stretch(float *sampleBuffer, long numSamples, unsigned int samplerate,
                      float rateShift, int pitchShift, float **outBuffer, 
                      long &numSamplesOut, long &maxNumSamples)


cdef data_to_numpy_array_float32(float *data, np.npy_intp size):
    cdef np.ndarray[float, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_FLOAT32, data)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
    return arr


def stretch(np.ndarray[float, ndim=2, mode='c'] samples,
        unsigned int samplerate, float rateShift, int pitchShift):
    cdef float *outBuffer = NULL
    cdef long numSamplesOut = 0, maxNumSamples = 0
    cdef long numSamples = samples.shape[0]
    assert samples.shape[1] == 2, 'input is not a stereo interleaved buffer'
    _stretch(&samples[0,0], numSamples, samplerate,
        rateShift, pitchShift, &outBuffer, numSamplesOut, maxNumSamples)
    return data_to_numpy_array_float32(outBuffer, numSamplesOut * 2).reshape(numSamplesOut, 2)
