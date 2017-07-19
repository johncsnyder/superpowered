import numpy as np
cimport numpy as np


cdef extern from 'SuperpoweredDecoder.h':
    cdef int SUPERPOWEREDDECODER_EOF
    cdef int SUPERPOWEREDDECODER_OK
    cdef int SUPERPOWEREDDECODER_ERROR
    cdef int SUPERPOWEREDDECODER_BUFFERING
    cdef int SUPERPOWEREDDECODER_NETWORK_ERROR

    ctypedef enum SuperpoweredDecoder_Kind:
        pass

    ctypedef struct stemsCompressor:
        pass

    ctypedef struct stemsLimiter:
        pass

    ctypedef void (* SuperpoweredDecoderID3Callback) (void *clientData, void *frameName, void *frameData, int frameDataSize)

    cdef cppclass SuperpoweredDecoder:
        SuperpoweredDecoder() except +

        # READ ONLY properties
        double durationSeconds
        long durationSamples, samplePosition
        unsigned int samplerate, samplesPerFrame
        SuperpoweredDecoder_Kind kind

        const char *open(const char *path, bint metaOnly, int offset, int length, int stemsIndex, char **customHTTPHeaders = 0)
        unsigned char decode(short int *pcmOutput, unsigned int *samples)
        unsigned char seek(long sample, bint precise)
        unsigned char getAudioStartSample(unsigned int *startSample, unsigned int limitSamples, int decibel);
        void reconnectToMediaserver()
        # void getMetaData(char **artist, char **title, void **image, int *imageSizeBytes, float *bpm, SuperpoweredDecoderID3Callback callback, void *clientData, int maxFrameDataSize)
        # bint getStemsInfo(char *names[4], char *colors[4], stemsCompressor *compressor, stemsLimiter *limiter)


cdef class Decoder:
    """
     @brief Audio file decoder. Provides uncompresses PCM samples from various compressed formats.

     Thread safety: single threaded, not thread safe. After a succesful open(), samplePosition and duration may change.
     Supported file types:
     - Stereo or mono pcm WAV and AIFF (16-bit int, 24-bit int, 32-bit int or 32-bit IEEE float).
     - MP3 (all kind).
     - AAC-LC in M4A container (iTunes).
     - AAC-LC in ADTS container (.aac).
     - Apple Lossless (on iOS only).

     @param durationSeconds The duration of the current file in seconds. Read only.
     @param durationSamples The duration of the current file in samples. Read only.
     @param samplePosition The current position in samples. May change after each decode() or seekTo(). Read only.
     @param samplerate The sample rate of the current file. Read only.
     @param samplesPerFrame How many samples are in one frame of the source file. For example: 1152 in mp3 files.
     @param kind The format of the current file.
    """
    cdef SuperpoweredDecoder _decoder
