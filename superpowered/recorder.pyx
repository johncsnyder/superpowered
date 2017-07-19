import cython
import numpy as np
cimport numpy as np
from libc.stdio cimport FILE, fwrite


cdef extern from "SuperpoweredRecorder.h":
    FILE *createWAV(const char *path, unsigned int samplerate, unsigned short int numChannels)
    void closeWAV(FILE *fd)


cdef class WAV:
    """
     @brief Creates a 16-bit WAV file.

     @param path The full filesystem path of the file.
     @param samplerate Sample rate.
     @param numChannels Number of channels.
    """
    cdef FILE *fd

    def __init__(self, str path, unsigned int samplerate = 44100, unsigned short int numChannels = 2):
        self.fd = createWAV(bytes(path, 'utf-8'), samplerate, numChannels)
        if self.fd is NULL:
            raise IOError('Could not create WAV file')

    def write(self, np.ndarray[short int, ndim=2, mode='c'] samples):
        cdef unsigned int count = samples.shape[0] * samples.shape[1]
        fwrite(&samples[0,0], samples.itemsize, count, self.fd)

    def __del__(self):
        closeWAV(self.fd)

    def close(self):
        closeWAV(self.fd)
