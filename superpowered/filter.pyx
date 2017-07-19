import cython
import numpy as np
cimport numpy as np
from enum import Enum


cdef extern from "SuperpoweredFilter.h":

    ctypedef enum SuperpoweredFilterType:
        pass

    cdef cppclass SuperpoweredFilter:
        # SuperpoweredFilter() except +
        SuperpoweredFilter(SuperpoweredFilterType filterType, unsigned int samplerate) except +

        # READ ONLY parameters
        float frequency
        float decibel
        float resonance
        float octave
        float slope
        SuperpoweredFilterType type

        void setResonantParameters(float frequency, float resonance)
        void setShelfParameters(float frequency, float slope, float dbGain)
        void setBandlimitedParameters(float frequency, float octaveWidth)
        void setParametricParameters(float frequency, float octaveWidth, float dbGain)
        void setResonantParametersAndType(float frequency, float resonance, SuperpoweredFilterType type)
        void setShelfParametersAndType(float frequency, float slope, float dbGain, SuperpoweredFilterType type)
        void setBandlimitedParametersAndType(float frequency, float octaveWidth, SuperpoweredFilterType type)
        void setCustomCoefficients(float *coefficients)
        void enable(bint flag)
        void setSamplerate(unsigned int samplerate)
        void reset()
        bint process(float *input, float *output, unsigned int numberOfSamples)


class FilterType(Enum):
    Resonant_Lowpass     = 0
    Resonant_Highpass    = 1
    Bandlimited_Bandpass = 2
    Bandlimited_Notch    = 3
    LowShelf             = 4
    HighShelf            = 5
    Parametric           = 6
    CustomCoefficients   = 7


cdef class Filter:
    """
     @brief IIR filters.

     It doesn't allocate any internal buffers and needs just a few bytes of memory.

     @param frequency Current frequency value. Read only.
     @param decibel Current decibel value for shelving and parametric filters. Read only.
     @param resonance Current resonance value for resonant filters. Read only.
     @param octave Current octave value for bandlimited and parametric filters. Read only.
     @param slope Current slope value for shelving filters. Read only.
     @param type Filter type. Read only.
    """
    cdef SuperpoweredFilter *_filter

    def __cinit__(self, filterType, unsigned int samplerate):
        self._filter = new SuperpoweredFilter(filterType.value, samplerate)

    def __dealloc__(self):
        del self._filter

    @property
    def frequency(self):
        return self._filter[0].frequency

    @property
    def decibel(self):
        return self._filter[0].decibel

    @property
    def resonance(self):
        return self._filter[0].resonance

    @property
    def octave(self):
        return self._filter[0].octave

    @property
    def slope(self):
        return self._filter[0].slope

    @property
    def type(self):
        return FilterType(self._filter[0].type)

    def setResonantParameters(self, float frequency, float resonance):
        """Change parameters for resonant filters."""
        self._filter[0].setResonantParameters(frequency, resonance)

    def setShelfParameters(self, float frequency, float slope, float dbGain):
        """Change parameters for shelving filters."""
        self._filter[0].setShelfParameters(frequency, slope, dbGain)

    def setBandlimitedParameters(self, float frequency, float octaveWidth):
        """Change parameters for bandlimited filters."""
        self._filter[0].setBandlimitedParameters(frequency, octaveWidth)

    def setParametricParameters(self, float frequency, float octaveWidth, float dbGain):
        """Change parameters for parametric filters."""
        self._filter[0].setParametricParameters(frequency, octaveWidth, dbGain)

    def setResonantParametersAndType(self, float frequency, float resonance, filterType):
        """
         @brief Set params and type at once for resonant filters.

         @param frequency The frequency in Hz.
         @param resonance Resonance value.
         @param type Must be lowpass or highpass.
        """
        self._filter[0].setResonantParametersAndType(frequency, resonance, filterType.value)

    def setShelfParametersAndType(self, float frequency, float slope, float dbGain, filterType):
        """
         @brief Set params and type at once for shelving filters.

         @param frequency The frequency in Hz.
         @param slope Slope.
         @param dbGain Gain in decibel.
         @param type Must be low shelf or high shelf.
        """
        self._filter[0].setShelfParametersAndType(frequency, slope, dbGain, filterType.value)

    def setBandlimitedParametersAndType(self, float frequency, float octaveWidth, filterType):
        """
         @brief Set params and type at once for bandlimited filters.

         @param frequency The frequency in Hz.
         @param octaveWidth Width in octave.
         @param type Must be bandpass or notch.
        """
        self._filter[0].setBandlimitedParametersAndType(frequency, octaveWidth, filterType.value)

    def enable(self, bint flag):
        """Turns the effect on/off."""
        self._filter[0].enable(flag)

    def setSamplerate(self, unsigned int samplerate):
        """
         @brief Sets the sample rate.

         @param samplerate 44100, 48000, etc.
        """
        self._filter[0].setSamplerate(samplerate)

    def reset(self):
        """
         @brief Reset all internals, sets the instance as good as new and turns it off.
        """
        self._filter[0].reset()

    def process(self, np.ndarray[float, ndim=2, mode='c'] input, np.ndarray[float, ndim=2, mode='c'] output):
        """
         @brief Processes interleaved audio.

         @return Put something into output or not.

         @param input 32-bit interleaved stereo input buffer. Can point to the same location with output (in-place processing).
         @param output 32-bit interleaved stereo output buffer. Can point to the same location with input (in-place processing).
         @param numberOfSamples Should be 32 minimum.
        """
        cdef unsigned int numberOfSamples = input.shape[0]
        assert input.shape[1] == 2, 'input is not a stereo interleaved buffer'
        assert input.shape == output.shape
        self._filter[0].process(&input[0,0], &output[0,0], numberOfSamples)
