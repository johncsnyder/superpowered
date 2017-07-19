import numpy as np
cimport numpy as np


cdef extern from "SuperpoweredMixer.h":

    cdef cppclass SuperpoweredStereoMixer:
        SuperpoweredStereoMixer() except +

        void process(float *inputs[4], float *outputs[2], float inputLevels[8], 
            float outputLevels[2], float inputMeters[8], float outputMeters[2], 
            unsigned int numberOfSamples)
        # void processPFL(float *channels[4], float *outputs[2], bool channelSwitches[4], 
            # float channelOutputLevels[4], unsigned int numberOfSamples)


cdef class StereoMixer:
    """
     @brief Mixer and splitter.
     
     Mixes max. 4 interleaved stereo inputs together. Output can be interleaved or non-interleaved (split). Separate input channel levels (good for gain and pan), separate output channel levels (master gain and pan). Returns maximum values for metering.
     
     One instance allocates just a few bytes of memory.
    """
    cdef SuperpoweredStereoMixer *_mixer

    def __cinit__(self):
        self._mixer = new SuperpoweredStereoMixer()

    def __dealloc__(self):
        del self._mixer

    def mix2(self, np.ndarray[float, ndim=2, mode='c'] in1,
                   np.ndarray[float, ndim=2, mode='c'] in2,
                   np.ndarray[float, ndim=2, mode='c'] out):
        """
         @brief Mixes max. 4 interleaved stereo inputs into a stereo output.
         
         @param inputs Four pointers to stereo interleaved input buffers. Any pointer can be NULL.
         @param outputs If outputs[1] is NULL, output is interleaved stereo in outputs[0]. If outputs[1] is not NULL, output is non-interleaved (left side in outputs[0], right side in outputs[1]).
         @param inputLevels Input volume level for each channel. Value changes between consecutive processes are automatically smoothed.
         @param outputLevels Output levels [left, right]. Value changes between consecutive processes are automatically smoothed.
         @param inputMeters Returns with the maximum values for metering. Can be NULL.
         @param outputMeters Returns with the maximum values for metering. Can be NULL.
         @param numberOfSamples The number of samples to process. Minimum 2, maximum 2048, must be exactly divisible with 2.
        """
        cdef unsigned int numberOfSamples = in1.shape[0]
        assert in1.shape[1] == in2.shape[1] == 2, 'input is not a stereo interleaved buffer'
        assert in1.shape[0] == in2.shape[0], 'inputs should have the same number of samples'
        cdef float *inputLevels = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        cdef float *outputLevels = [1.0, 1.0]
        cdef float **inputs = [&in1[0,0], &in2[0,0], NULL, NULL]
        cdef float **outputs = [&out[0,0], NULL]
        cdef float *inputMeters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cdef float *outputMeters = [0.0, 0.0]
        self._mixer[0].process(inputs, outputs, inputLevels, outputLevels,
            inputMeters, outputMeters, numberOfSamples)
