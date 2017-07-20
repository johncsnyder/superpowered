#include <stdlib.h>
#include "SuperpoweredAnalyzer.h"
#include "SuperpoweredAudioBuffers.h"
#include "SuperpoweredDecoder.h"
#include "SuperpoweredRecorder.h"
#include "SuperpoweredSimple.h"
#include "SuperpoweredTimeStretching.h"


void _stretch(float *sampleBuffer, long numSamples, unsigned int samplerate,
                  float rateShift, int pitchShift, float **outBuffer,
                  long &numSamplesOut, long &maxNumSamples, unsigned int samplesPerFrame = 1024) {
  SuperpoweredTimeStretching *timeStretch = new SuperpoweredTimeStretching(samplerate);
  timeStretch->setRateAndPitchShift(rateShift, pitchShift);
  // This buffer list will receive the time-stretched samples.
  SuperpoweredAudiopointerList *outputBuffers = new SuperpoweredAudiopointerList(8, 16);

  // Create buffer to hold all samples
  maxNumSamples = numSamples / rateShift;
  *outBuffer = (float *)malloc(maxNumSamples * 2 * sizeof(float));

  // int retcode = SUPERPOWEREDDECODER_OK;

  long samplesProcessed = 0;
  numSamplesOut = 0;

  while (true) {
    if (samplesProcessed >= numSamples) break;

    if (samplesProcessed + samplesPerFrame > numSamples) {
        samplesPerFrame = numSamples - samplesProcessed;
    }

    // Create an input buffer for the time stretcher.
    SuperpoweredAudiobufferlistElement inputBuffer;
    inputBuffer.samplePosition = samplesProcessed;
    inputBuffer.startSample = 0;
    inputBuffer.samplesUsed = 0;
    inputBuffer.endSample = samplesPerFrame;  // <-- Important!
    inputBuffer.buffers[0] =
        SuperpoweredAudiobufferPool::getBuffer(samplesPerFrame * 8 + 64);
    inputBuffer.buffers[1] = inputBuffer.buffers[2] = inputBuffer.buffers[3] =
        NULL;

    // Copy samples to inputBuffer
    memcpy((float *)inputBuffer.buffers[0], sampleBuffer + samplesProcessed * 2,
                samplesPerFrame * 2 * sizeof(float));

    // Time stretching.
    timeStretch->process(&inputBuffer, outputBuffers);

    // Do we have some output?
    if (outputBuffers->makeSlice(0, outputBuffers->sampleLength)) {
      while (true) {  // Iterate on every output slice.
        // Get pointer to the output samples.
        int sampleCount = 0;
        float *timeStretchedAudio = (float *)outputBuffers->nextSliceItem(&sampleCount);
        if (!timeStretchedAudio) break;

        if (numSamplesOut < maxNumSamples) {

            if (numSamplesOut + sampleCount >= maxNumSamples) {
                sampleCount = maxNumSamples - numSamplesOut;
            }

            memcpy(*outBuffer + numSamplesOut * 2, timeStretchedAudio,
                sampleCount * 2 * sizeof(float));
            numSamplesOut += sampleCount;
        }

      };

      // Clear the output buffer list.
      outputBuffers->clear();
    };

    samplesProcessed += samplesPerFrame;  // keep track of samples processed
  };

  // Cleanup.
  delete timeStretch;
  delete outputBuffers;
}
