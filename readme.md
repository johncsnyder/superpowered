
### superpowered

_superpowered_ is a python/cython wrapper around the _Superpowered SDK_. 
Requires [NumPy](http://www.numpy.org/).

(-) Get the SDK at 
<https://github.com/superpoweredSDK/Low-Latency-Android-Audio-iOS-Audio-Engine>.

(-) Install via pip
```
export SUPERPOWERED_ROOT=/path/to/superpowered/sdk
pip install git+https://github.com/johncsnyder/superpowered.git
```

(-) Currently, _superpowered_ does not wrap all the features, and is 
mainly focused on utilizing the offline processing features. For example,
decoding an audio file and reading all the samples is as easy as

```python
import superpowered
from superpowered.decoder import Decoder
decoder = Decoder()
decoder.open('track.mp3')
samples = decoder.read()  # ndarray of shape (numSamples, numChannels)
```

To see more features, check out _examples/usage.ipynb_.

(-) Tested on OS X against _Superpowered SDK, Version 1.0.5 (Jun 15, 2017)_. 
If you run into trouble, please open an issue.

### License

Although the _Superpowered SDK_ is not bundled with this wrapper, it includes 
some of the original source code. Any use is 
subject to the terms of the _Superpowered SDK_ license. The latest version can be found at 
<http://superpowered.com/license>.
