# aether — audio buffers and the time grid

`aether.audio` implements mono PCM float32 little-endian. A packet contains 1–3840
samples; NaN/Infinity, values outside [-1,1], and incomplete float32 values are
rejected. An empty packet is not replaced with silence. `pop()` returns `None`
while no full frame is available.

`PCMBuffer` assembles frames of exactly 1920 samples (80 ms at 24000 Hz).
`AudioFrame` contains an index, sample_offset, and PCM bytes. Packet boundaries
do not affect frame bytes. The queue holds no more than max_queue_frames full
frames plus one incomplete tail. An incoming packet is either accepted in full
or rejected in full with a `BufferError`; previously accepted samples are
preserved. The number of overflows is counted. After each push, the consumer
retrieves frames via pop; sustained overload is expected to terminate the
future runtime session rather than drop audio from the middle of the queue.

Counters: received_samples, emitted_samples, padded_samples, discarded_samples,
accepted_packets, overflow_count, and buffered_samples. For accepted audio:
`received + padded = emitted + buffered + discarded`.

`finish(pad=False)` closes the input, counting and discarding only the
incomplete tail; full frames remain available for reading. For offline use,
`finish(pad=True)` is permitted: the tail is zero-padded, and the amount of
padding is recorded. When the queue is full, the ready frames are retrieved
first, after which finish is called again. `reset()` fully clears data,
counters, and the closed state. A runtime stop is expected to use reset when
all ready frames must also be removed.

`resample_pcm` is a reference offline transformation of an entire recording
using a windowed sinc filter with anti-aliasing on downsampling. Integer
sample rates from 8000–192000 Hz are supported. The result length is
`ceil(N * target/source)`; edges are extended with the nearest sample. The
output is clipped to [-1,1] after any filter overshoot. When sample rates
match, the bytes are preserved unchanged. This is not a streaming resampler:
it cannot be applied independently to network packets. The first version's
transport accepts only 24000 Hz. Device resampling, drift compensation, and
real speech quality have not yet been verified.

The tests verify random packet boundaries, exact PCM preservation, counters,
overflow, tail handling, reset, buffer independence, DC, length, and
frequency response on synthetic signals. The microphone, codec, model, and
training are not exercised.
