import cython
import numpy as np
cimport numpy as np


cdef extern from "SuperpoweredSimple.h":
    void SuperpoweredVolume (float *input, float *output, float volumeStart, float volumeEnd, unsigned int numberOfSamples)
    void SuperpoweredChangeVolume (float *input, float *output, float volumeStart, float volumeChange, unsigned int numberOfSamples)
    void SuperpoweredVolumeAdd (float *input, float *output, float volumeStart, float volumeEnd, unsigned int numberOfSamples)
    void SuperpoweredChangeVolumeAdd (float *input, float *output, float volumeStart, float volumeChange, unsigned int numberOfSamples)
    float SuperpoweredPeak (float *input, unsigned int numberOfValues)
    void SuperpoweredCharToFloat (signed char *input, float *output, unsigned int numberOfSamples, unsigned int numChannels)
    void SuperpoweredFloatToChar (float *input, signed char *output, unsigned int numberOfSamples, unsigned int numChannels)
    void Superpowered24bitToFloat (void *input, float *output, unsigned int numberOfSamples, unsigned int numChannels)
    void SuperpoweredFloatTo24bit (float *input, void *output, unsigned int numberOfSamples, unsigned int numChannels)
    void SuperpoweredIntToFloat (int *input, float *output, unsigned int numberOfSamples, unsigned int numChannels)
    void SuperpoweredFloatToInt (float *input, int *output, unsigned int numberOfSamples, unsigned int numChannels)
    void SuperpoweredFloatToShortInt (float *input, short int *output, unsigned int numberOfSamples, unsigned int numChannels)
    void SuperpoweredFloatToShortInt (float *inputLeft, float *inputRight, short int *output, unsigned int numberOfSamples)
    void SuperpoweredShortIntToFloat (short int *input, float *output, unsigned int numberOfSamples, float *peaks)
    void SuperpoweredShortIntToFloat (short int *input, float *output, unsigned int numberOfSamples, unsigned int numChannels)
    void SuperpoweredInterleave (float *left, float *right, float *output, unsigned int numberOfSamples)
    void SuperpoweredInterleaveAndGetPeaks (float *left, float *right, float *output, unsigned int numberOfSamples, float *peaks)
    void SuperpoweredDeInterleave (float *input, float *left, float *right, unsigned int numberOfSamples)
    void SuperpoweredDeInterleaveAdd (float *input, float *left, float *right, unsigned int numberOfSamples)
    bint SuperpoweredHasNonFinite (float *buffer, unsigned int numberOfValues)
    void SuperpoweredStereoToMono (float *input, float *output, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples)
    void SuperpoweredStereoToMono2 (float *input, float *output0, float *output1, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples)
    void SuperpoweredCrossMono (float *left, float *right, float *output, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples)
    void SuperpoweredCrossMono2 (float *left, float *right, float *output0, float *output1, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples)
    void SuperpoweredCrossStereo (float *inputA, float *inputB, float *output, float gainStart[4], float gainEnd[4], unsigned int numberOfSamples)
    const unsigned char *SuperpoweredVersion ()


def volume(np.ndarray[float, ndim=2, mode='c'] input, np.ndarray[float, ndim=2, mode='c'] output, float volumeStart, float volumeEnd):
    """
     @fn SuperpoweredVolume(float *input, float *output, float volumeStart, float volumeEnd, unsigned int numberOfSamples);
     @brief Applies volume on a single stereo interleaved buffer.

     @param input Input buffer.
     @param output Output buffer. Can be equal to input (in-place processing).
     @param volumeStart Volume for the first sample.
     @param volumeEnd Volume for the last sample. Volume will be smoothly calculated between start end end.
     @param numberOfSamples The number of samples to process.
    """
    cdef unsigned int numberOfSamples = input.shape[0]
    assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
    return SuperpoweredVolume(&input[0,0], &output[0,0], volumeStart, volumeEnd, numberOfSamples)

def change_volume(np.ndarray[float, ndim=2, mode='c'] input, np.ndarray[float, ndim=2, mode='c'] output, float volumeStart, float volumeChange):
    """
     @fn SuperpoweredChangeVolume(float *input, float *output, float volumeStart, float volumeChange, unsigned int numberOfSamples);
     @brief Applies volume on a single stereo interleaved buffer.

     @param input Input buffer.
     @param output Output buffer. Can be equal to input (in-place processing).
     @param volumeStart Voume for the first sample.
     @param volumeChange Change volume by this amount for every sample.
     @param numberOfSamples The number of samples to process.
    """
    cdef unsigned int numberOfSamples
    numberOfSamples = input.shape[0]
    assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
    return SuperpoweredChangeVolume(&input[0,0], &output[0,0], volumeStart, volumeChange, numberOfSamples)

def volume_add(np.ndarray[float, ndim=2, mode='c'] input, np.ndarray[float, ndim=2, mode='c'] output, float volumeStart, float volumeEnd):
    """
     @fn SuperpoweredVolumeAdd(float *input, float *output, float volumeStart, float volumeEnd, unsigned int numberOfSamples);
     @brief Applies volume on a single stereo interleaved buffer and adds it to the audio in the output buffer.

     @param input Input buffer.
     @param output Output buffer.
     @param volumeStart Volume for the first sample.
     @param volumeEnd Volume for the last sample. Volume will be smoothly calculated between start end end.
     @param numberOfSamples The number of samples to process.
    """
    cdef unsigned int numberOfSamples
    numberOfSamples = input.shape[0]
    assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
    return SuperpoweredVolumeAdd(&input[0,0], &output[0,0], volumeStart, volumeEnd, numberOfSamples)

def change_volume_add(np.ndarray[float, ndim=2, mode='c'] input, np.ndarray[float, ndim=2, mode='c'] output, float volumeStart, float volumeChange):
    """
     @fn SuperpoweredChangeVolumeAdd(float *input, float *output, float volumeStart, float volumeChange, unsigned int numberOfSamples);
     @brief Applies volume on a single stereo interleaved buffer and adds it to the audio in the output buffer.

     @param input Input buffer.
     @param output Output buffer.
     @param volumeStart Volume for the first sample.
     @param volumeChange Change volume by this amount for every sample.
     @param numberOfSamples The number of samples to process.
    """
    cdef unsigned int numberOfSamples = input.shape[0]
    assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
    return SuperpoweredChangeVolumeAdd(&input[0,0], &output[0,0], volumeStart, volumeChange, numberOfSamples)

def peak(np.ndarray[float, ndim=2, mode='c'] input):
    """
     @fn SuperpoweredPeak(float *input, unsigned int numberOfValues);
     @return Returns with the peak value.

     @param input An array of floating point values.
     @param numberOfValues The number of values to process. (2 * numberOfSamples for stereo input) Must be a multiply of 8.
    """
    cdef unsigned int numberOfValues = len(input)
    # assert numberOfValues % 8 == 0, 'Must be a multiple of 8'
    return SuperpoweredPeak(&input[0,0], numberOfValues)

def char_to_float(np.ndarray[signed char, ndim=2, mode='c'] input, np.ndarray[float, ndim=2, mode='c'] output):
    """
     @fn SuperpoweredCharToFloat(signed char *input, float *output, unsigned int numberOfSamples, unsigned int numChannels);
     @brief Converts 8-bit audio to 32-bit floating point.
     
     @param input Input buffer.
     @param output Output buffer.
     @param numberOfSamples The number of samples to process.
     @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
    """
    cdef unsigned int numberOfSamples = input.shape[0]
    cdef unsigned int numChannels = input.shape[1]
    # assert input.shape == output.shape, 'input and output buffers must have the same shape'
    return SuperpoweredCharToFloat(&input[0,0], &output[0,0], numberOfSamples, numChannels)

def float_to_char(np.ndarray[float, ndim=2, mode='c'] input, np.ndarray[signed char, ndim=2, mode='c'] output):
    """
     @fn SuperpoweredFloatToChar(float *input, signed char *output, unsigned int numberOfSamples, unsigned int numChannels);
     @brief Converts 32-bit floating point audio 8-bit audio.

     @param input Input buffer.
     @param output Output buffer.
     @param numberOfSamples The number of samples to process.
     @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
    """
    cdef unsigned int numberOfSamples = input.shape[0]
    cdef unsigned int numChannels = input.shape[1]
    # assert input.shape == output.shape, 'input and output buffers must have the same shape'
    return SuperpoweredFloatToChar(&input[0,0], &output[0,0], numberOfSamples, numChannels)

# def Superpowered24bitToFloat(np.ndarray input, np.ndarray[float, ndim=1] output, unsigned int numberOfSamples, unsigned int numChannels = 2):
#     """
#      @fn Superpowered24bitToFloat(void *input, float *output, unsigned int numberOfSamples, unsigned int numChannels);
#      @brief Converts 24-bit audio to 32-bit floating point.

#      @param input Input buffer.
#      @param output Output buffer.
#      @param numberOfSamples The number of samples to process.
#      @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
#     """
#     return _Superpowered24bitToFloat(&input[0], &output[0], numberOfSamples, numChannels)

# def SuperpoweredFloatTo24bit(np.ndarray[float, ndim=1] input, np.ndarray output, unsigned int numberOfSamples, unsigned int numChannels = 2):
#     """
#      @fn SuperpoweredFloatTo24bit(float *input, void *output, unsigned int numberOfSamples, unsigned int numChannels);
#      @brief Converts 32-bit floating point audio to 24-bit.

#      @param input Input buffer.
#      @param output Output buffer.
#      @param numberOfSamples The number of samples to process.
#      @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
#     """
#     return _SuperpoweredFloatTo24bit(&input[0], &output[0], numberOfSamples, numChannels)

# def SuperpoweredIntToFloat(np.ndarray[int, ndim=1] input, np.ndarray[float, ndim=1] output, unsigned int numberOfSamples, unsigned int numChannels = 2):
#     """
#      @fn SuperpoweredIntToFloat(int *input, float *output, unsigned int numberOfSamples, unsigned int numChannels);
#      @brief Converts 32-bit integer audio to 32-bit floating point.

#      @param input Input buffer.
#      @param output Output buffer.
#      @param numberOfSamples The number of samples to process.
#      @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
#     """
#     return _SuperpoweredIntToFloat(&input[0], &output[0], numberOfSamples, numChannels)

# def SuperpoweredFloatToInt(np.ndarray[float, ndim=1] input, np.ndarray[int, ndim=1] output, unsigned int numberOfSamples, unsigned int numChannels = 2):
#     """
#      @fn SuperpoweredFloatToInt(float *input, int *output, unsigned int numberOfSamples, unsigned int numChannels);
#      @brief Converts 32-bit floating point audio to 32-bit integer.

#      @param input Input buffer.
#      @param output Output buffer.
#      @param numberOfSamples The number of samples to process.
#      @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
#     """
#     return _SuperpoweredFloatToInt(&input[0], &output[0], numberOfSamples, numChannels)

def float_to_short_int(np.ndarray[float, ndim=2, mode='c'] input, np.ndarray[short int, ndim=2, mode='c'] output):
    """
     @fn SuperpoweredFloatToShortInt(float *input, short int *output, unsigned int numberOfSamples);
     @brief Converts 32-bit float input to 16-bit signed integer output.

     @param input Input buffer.
     @param output Output buffer.
     @param numberOfSamples The number of samples to process.
     @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
    """
    cdef unsigned int numberOfSamples = input.shape[0]
    cdef unsigned int numChannels = input.shape[1]
    # assert input.shape == output.shape, 'input and output buffers must have the same shape'
    return SuperpoweredFloatToShortInt(&input[0,0], &output[0,0], numberOfSamples, numChannels)

# def SuperpoweredFloatToShortInt(np.ndarray[float, ndim=1] inputLeft, np.ndarray[float, ndim=1] inputRight, np.ndarray[short int, ndim=1] output, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredFloatToShortInt(float *inputLeft, float *inputRight, short int *output, unsigned int numberOfSamples);
#      @brief Converts two 32-bit float input channels to stereo interleaved 16-bit signed integer output.

#      @param inputLeft 32-bit input for the left side. Should be numberOfSamples + 8 big minimum.
#      @param inputRight 32-bit input for the right side. Should be numberOfSamples + 8 big minimum.
#      @param output Stereo interleaved 16-bit output. Should be numberOfSamples * 2 + 16 big minimum.
#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredFloatToShortInt(&inputLeft[0], &inputRight[0], &output[0], numberOfSamples)

# def SuperpoweredShortIntToFloat(np.ndarray[short int, ndim=1] input, np.ndarray[float, ndim=1] output, unsigned int numberOfSamples, np.ndarray[float, ndim=1] peaks):
#     """
#      @fn SuperpoweredShortIntToFloat(short int *input, float *output, unsigned int numberOfSamples, float *peaks);
#      @brief Converts a stereo interleaved 16-bit signed integer input to stereo interleaved 32-bit float output.

#      @param input Stereo interleaved 16-bit input. Should be numberOfSamples + 8 big minimum.
#      @param output Stereo interleaved 32-bit output. Should be numberOfSamples + 8 big minimum.
#      @param numberOfSamples The number of samples to process.
#      @param peaks Peak value result (2 floats: left peak, right peak).
#     """
#     return _SuperpoweredShortIntToFloat(&input[0], &output[0], numberOfSamples, &peaks[0])

def short_int_to_float(np.ndarray[short int, ndim=2, mode='c'] input, np.ndarray[float, ndim=2, mode='c'] output):
    """
     @fn SuperpoweredShortIntToFloat(short int *input, float *output, unsigned int numberOfSamples, unsigned int numChannels);
     @brief Converts 16-bit signed integer input to 32-bit float output.

     @param input Input buffer.
     @param output Output buffer.
     @param numberOfSamples The number of samples to process.
     @param numChannels The number of channels. One sample may be 1 value (1 channels) or N values (N channels).
    """
    cdef unsigned int numberOfSamples = input.shape[0]
    cdef unsigned int numChannels = input.shape[1]
    # assert input.shape == output.shape, 'input and output buffers must have the same shape'
    return SuperpoweredShortIntToFloat(&input[0,0], &output[0,0], numberOfSamples, numChannels)

# def SuperpoweredInterleave(np.ndarray[float, ndim=1] left, np.ndarray[float, ndim=1] right, np.ndarray[float, ndim=1] output, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredInterleave(float *left, float *right, float *output, unsigned int numberOfSamples);
#      @brief Makes an interleaved output from two input channels.
     
#      @param left Input for left channel.
#      @param right Input for right channel.
#      @param output Interleaved output.
#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredInterleave(&left[0], &right[0], &output[0], numberOfSamples)

# def SuperpoweredInterleaveAndGetPeaks(np.ndarray[float, ndim=1] left, np.ndarray[float, ndim=1] right, np.ndarray[float, ndim=1] output, unsigned int numberOfSamples, np.ndarray[float, ndim=1] peaks):
#     """
#      @fn SuperpoweredInterleaveAndGetPeaks(float *left, float *right, float *output, unsigned int numberOfSamples, float *peaks);
#      @brief Makes an interleaved output from two input channels, and measures the input volume.

#      @param left Input for left channel.
#      @param right Input for right channel.
#      @param output Interleaved output.
#      @param numberOfSamples The number of samples to process.
#      @param peaks Peak value result (2 floats: left peak, right peak).
#     """
#     return _SuperpoweredInterleaveAndGetPeaks(&left[0], &right[0], &output[0], numberOfSamples, &peaks[0])

# def SuperpoweredDeInterleave(np.ndarray[float, ndim=1] input, np.ndarray[float, ndim=1] left, np.ndarray[float, ndim=1] right, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredDeInterleave(float *input, float *left, float *right, unsigned int numberOfSamples);
#      @brief Deinterleaves an interleaved input.

#      @param input Interleaved input.
#      @param left Output for left channel.
#      @param right Output for right channel.

#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredDeInterleave(&input[0], &left[0], &right[0], numberOfSamples)

# def SuperpoweredDeInterleaveAdd(np.ndarray[float, ndim=1] input, np.ndarray[float, ndim=1] left, np.ndarray[float, ndim=1] right, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredDeInterleaveAdd(float *input, float *left, float *right, unsigned int numberOfSamples);
#      @brief Deinterleaves an interleaved input and adds the results to the output channels.

#      @param input Interleaved input.
#      @param left Output for left channel.
#      @param right Output for right channel.

#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredDeInterleaveAdd(&input[0], &left[0], &right[0], numberOfSamples)

# def SuperpoweredHasNonFinite(np.ndarray[float, ndim=1] buffer, unsigned int numberOfValues):
#     """
#      @fn SuperpoweredHasNonFinite(float *buffer, unsigned int numberOfValues);
#      @brief Checks if the samples has non-valid samples, such as infinity or NaN (not a number).
     
#      @param buffer The buffer to check.
#      @param numberOfValues Number of values in buffer. For stereo buffers, multiply by two!
#     """
#     return _SuperpoweredHasNonFinite(&buffer[0], numberOfValues)

# def SuperpoweredStereoToMono(np.ndarray[float, ndim=1] input, np.ndarray[float, ndim=1] output, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredStereoToMono(float *input, float *output, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples);
#      @brief Makes mono output from stereo input.

#      @param input Stereo interleaved input.
#      @param output Output.
#      @param leftGainStart Gain of the first sample on the left channel.
#      @param leftGainEnd Gain for the last sample on the left channel. Gain will be smoothly calculated between start end end.
#      @param rightGainStart Gain of the first sample on the right channel.
#      @param rightGainEnd Gain for the last sample on the right channel. Gain will be smoothly calculated between start end end.
#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredStereoToMono(&input[0], &output[0], leftGainStart, leftGainEnd, rightGainStart, rightGainEnd, numberOfSamples)

# def SuperpoweredStereoToMono2(np.ndarray[float, ndim=1] input, np.ndarray[float, ndim=1] output0, np.ndarray[float, ndim=1] output1, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredStereoToMono2(float *input, float *output0, float *output1, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples);
#      @brief Makes two mono outputs from stereo input.

#      @param input Stereo interleaved input.
#      @param output0 Output.
#      @param output1 Output.
#      @param leftGainStart Gain of the first sample on the left channel.
#      @param leftGainEnd Gain for the last sample on the left channel. Gain will be smoothly calculated between start end end.
#      @param rightGainStart Gain of the first sample on the right channel.
#      @param rightGainEnd Gain for the last sample on the right channel. Gain will be smoothly calculated between start end end.
#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredStereoToMono2(&input[0], &output0[0], &output1[0], leftGainStart, leftGainEnd, rightGainStart, rightGainEnd, numberOfSamples)

# def SuperpoweredCrossMono(np.ndarray[float, ndim=1] left, np.ndarray[float, ndim=1] right, np.ndarray[float, ndim=1] output, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredCrossMono(float *left, float *right, float *output, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples);
#      @brief Makes mono output from two separate input channels.
     
#      @param left Input for left channel.
#      @param right Input for right channel.
#      @param output Output.
#      @param leftGainStart Gain of the first sample on the left channel.
#      @param leftGainEnd Gain for the last sample on the left channel. Gain will be smoothly calculated between start end end.
#      @param rightGainStart Gain of the first sample on the right channel.
#      @param rightGainEnd Gain for the last sample on the right channel. Gain will be smoothly calculated between start end end.
#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredCrossMono(&left[0], &right[0], &output[0], leftGainStart, leftGainEnd, rightGainStart, rightGainEnd, numberOfSamples)

# def SuperpoweredCrossMono2(np.ndarray[float, ndim=1] left, np.ndarray[float, ndim=1] right, np.ndarray[float, ndim=1] output0, np.ndarray[float, ndim=1] output1, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredCrossMono2(float *left, float *right, float *output0, float *output1, float leftGainStart, float leftGainEnd, float rightGainStart, float rightGainEnd, unsigned int numberOfSamples);
#      @brief Makes two mono outputs from two separate input channels.

#      @param left Input for left channel.
#      @param right Input for right channel.
#      @param output0 Output.
#      @param output1 Output.
#      @param leftGainStart Gain of the first sample on the left channel.
#      @param leftGainEnd Gain for the last sample on the left channel. Gain will be smoothly calculated between start end end.
#      @param rightGainStart Gain of the first sample on the right channel.
#      @param rightGainEnd Gain for the last sample on the right channel. Gain will be smoothly calculated between start end end.
#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredCrossMono2(&left[0], &right[0], &output0[0], &output1[0], leftGainStart, leftGainEnd, rightGainStart, rightGainEnd, numberOfSamples)

# def SuperpoweredCrossStereo(np.ndarray[float, ndim=1] inputA, np.ndarray[float, ndim=1] inputB, np.ndarray[float, ndim=1] output, float[:] gainStart, float[:] gainEnd, unsigned int numberOfSamples):
#     """
#      @fn SuperpoweredCrossStereo(float *inputA, float *inputB, float *output, float gainStart[4], float gainEnd[4], unsigned int numberOfSamples);
#      @brief Crossfades two separate input channels.

#      @param inputA Interleaved stereo input (first).
#      @param inputB Interleaved stereo input (second).
#      @param output Interleaved stereo output.
#      @param gainStart Gain of the first sample (first left, first right, second left, second right).
#      @param gainEnd Gain for the last sample (first left, first right, second left, second right). Gain will be smoothly calculated between start end end.
#      @param numberOfSamples The number of samples to process.
#     """
#     return _SuperpoweredCrossStereo(&inputA[0], &inputB[0], &output[0], &gainStart[0], &gainEnd[0], numberOfSamples)



# void SuperpoweredAdd1(float *input, float *output, unsigned int numberOfValues);
# void SuperpoweredAdd2(float *inputA, float *inputB, float *output, unsigned int numberOfValues);
# void SuperpoweredAdd4(float *inputA, float *inputB, float *inputC, float *inputD, float *output, unsigned int numberOfValues);


def version():
    return SuperpoweredVersion().decode('utf-8')

