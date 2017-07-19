import numpy as np
cimport numpy as np


cdef extern from "SuperpoweredBandpassFilterbank.h":

    cdef cppclass SuperpoweredBandpassFilterbank:
        SuperpoweredBandpassFilterbank(int numBands, float *frequencies, float *widths, 
            unsigned int samplerate) except +

        void setSamplerate(unsigned int samplerate)
        void process(float *input, float *bands, float *peak, float *sum, 
            unsigned int numberOfSamples, int group = 0)
        void processNoAdd(float *input, float *bands, float *peak, float *sum, 
            unsigned int numberOfSamples, int group = 0)


cdef class BandpassFilterbank:

    cdef SuperpoweredBandpassFilterbank *_filterbank
    cdef int numBands

    def __cinit__(self, np.ndarray[float, ndim=1] frequencies,
                        np.ndarray[float, ndim=1] widths,
                        unsigned int samplerate):
        """
         Create a filterbank instance.
         
         @param numBands The number of bands. Must be a multiply of 8.
         @param frequencies The center frequencies of the bands.
         @param widths The width of the bands. 1.0f is one octave, 1.0f / 12.0f is one halfnote.
         @param samplerate The initial sample rate.
        """
        self.numBands = frequencies.shape[0]
        assert frequencies.shape[0] == widths.shape[0], 'number of frequencies and widths must match'
        self._filterbank = new SuperpoweredBandpassFilterbank(self.numBands, &frequencies[0],
            &widths[0], samplerate)

    def __dealloc__(self):
        del self._filterbank

    def setSamplerate(self, unsigned int samplerate):
        self._filterbank[0].setSamplerate(samplerate)

    def _process(self, np.ndarray[float, ndim=2, mode='c'] input):
        cdef unsigned int numberOfSamples = input.shape[0]
        assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
        cdef float peak = 0.0
        cdef float sum = 0.0
        cdef np.ndarray[float, ndim=1] bands = np.zeros(self.numBands, dtype=np.float32)
        self._filterbank[0].processNoAdd(&input[0,0], &bands[0], &peak, &sum,
            numberOfSamples)
        return bands / numberOfSamples

    def process(self, np.ndarray[float, ndim=2, mode='c'] input, 
                unsigned int chunks=1, unsigned int chunksize=0):
        cdef unsigned int i
        cdef unsigned int numberOfSamples = input.shape[0]
        assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
        cdef float peak = 0.0
        cdef float sum = 0.0
        
        if chunksize == 0:
            chunksize = numberOfSamples // chunks
        else:
            chunks = numberOfSamples // chunksize

        cdef np.ndarray[float, ndim=2, mode='c'] bands = \
            np.zeros((chunks, self.numBands), dtype=np.float32)

        for i in range(chunks):
            self._filterbank[0].processNoAdd(&input[i*chunksize,0], &bands[i,0], 
                &peak, &sum, chunksize)

        return bands / chunksize
