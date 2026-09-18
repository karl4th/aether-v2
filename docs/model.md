# aether — model specification

## 1. Document accuracy level

The document fixes the initial computational contract of aether. This is not a claim that trained weights exist. Compatibility of a specific checkpoint is confirmed by comparing configuration, parameter and output names and shapes on reference examples. Matching overall transformer sizes is not sufficient.

## 2. Notation and tensors

`B` — batch size, `T` — number of audio frames, `N` — number of PCM samples, `Q=8` — number of active codebooks for one voice, `V=2048` — audio vocabulary, `Vt=32000` — text vocabulary without an additional input BOS token.

| Data | Shape | Type and meaning |
|---|---|---|
| PCM | `[B, 1, N]` | float32, finite values, nominally −1…1 |
| Input frame | `[B, 1, 1920]` | 80 ms at 24 kHz |
| Codes for one voice | `[B, 8, T]` | int64, values 0…2047 |
| Full sequence | `[B, 17, T]` | Text, 8 output codes, 8 input codes |
| Temporal states | `[B, T, 4096]` | Compute dtype from configuration |
| Text logits | `[B, 1, T, 32000]` | Pre-softmax |
| Output audio logits | `[B, 8, T, 2048]` | Own voice only |
| Finished step output | `[B, 9, 1]` | Text and 8 aligned codes |

The external API must not conflate stream numbers, audio-book numbers, and physical PCM channels. Mono audio has one physical channel but eight codebooks.

## 3. Initial configuration

| Component | Value |
|---|---|
| Temporal | 32 layers, width 4096, 32 heads, causal attention |
| Temporal normalization | RMSNorm; normalization accumulation in float32 |
| Temporal positions | RoPE |
| Depth | 6 layers, width 1024, 16 heads |
| Depth positions | No temporal RoPE; distinguishes codebook step |
| Temporal audio inputs | 16 books: 8 own and 8 user |
| Predicted audio books | 8 own |
| Temporal context | 3000 time steps, i.e. 240 seconds |
| Text | Single stream, vocabulary 32000 |
| Delays | `[0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1]` |

The exact gated-FFN sizes, projection order, epsilon parameters, depth-weight variants for different books, and RoPE convention must reside in the checkpoint's machine configuration. They cannot be computed from an approximate parameter count. The loader is required to reject any bundle in which this data is insufficient to unambiguously construct the network.

Context 3000 applies to time steps, not to text tokens or to 17 × 3000 sequential positions. Until a different policy is verified, the duration of the first session is bounded so as not to exceed the supported context, with a reserve for initialization and shutdown.

## 4. Audio codec

Decision of 2026-09-16: the first working result uses the initial codec. The future replacement is the Manifestro codec; its parameters are undefined. The shared Python interface `AudioCodec` defines `create_state/encode/decode/reset`, with separate state per direction and session. This is currently a Protocol, not an implemented codec.

The configuration contains the contract version, implementation, sample rate, frame size, books, vocabulary, and token-semantics status; a future bundle is required to contain a matching contract. Matching numerical sizes does not prove semantic compatibility. A replacement may require adaptation/retraining of the voice model and re-tokenization of the data. Until then, the current contract does not change.

Required operations: `encode(pcm, state)`, `decode(codes, state)`, `reset(state)`. Encoding and decoding state are separate. Codec parameters may be shared; caches are local to direction and session only.

The encoder performs causal convolutions and reduces the feature rate; temporal blocks add local context, after which the features are converted to 12.5 Hz. The quantizer produces indices. The decoder reconstructs features, increases their temporal resolution, and synthesizes the waveform.

For residual quantization, conceptually:

```text
r0 = z
for q from 0 to Q−1:
    cq = nearest code to rq
    rq+1 = rq − embeddingq(cq)
z_hat = sum of selected vectors
```

The semantic branch may be separated from the acoustic one, so the specific split quantizer should not automatically be replaced with an ordinary RVQ. For compatible weights, its actual projections and addition order are preserved.

The number of active books for dialogue may be smaller than the number contained in the codec checkpoint. The loader sets exactly eight and verifies the result. The BOS token 2048 is not a valid code for the audio decoder.

## 5. Temporal and inner prediction

After the delay schedule is applied, the step input is the sum of embeddings across 17 channels. Different audio books correspond to their own tables. The large transformer outputs state `h`; the text head predicts token `w`.

The depth network then outputs `a0…a7`. The first prediction is conditioned on `h` and `w`; subsequent ones are also conditioned on the previous codes of the inner chain. Whether separate projections and weights per book index are used is set by the configuration.

Conceptual factorization for the shifted model step:

```text
p(w, a0…a7 | history)
  = p(w | history) × ∏q p(aq | h, w, a0…aq−1)
```

This is the formula for the shifted space. It does not claim that all `aq` represent the same physical moment of sound.

The depth cache is cleared after the inner chain of eight books. The temporal cache is preserved across audio frames. Carrying the depth cache over to the next time step is an implementation error.

## 6. Delay schedule

Let `C[k,t]` be channel `k` of physical frame `t`, and `d[k]` its offset. Define the shifted sequence:

```text
D[k,s] = C[k, s − d[k]]
```

Negative indices are filled with initial values. The large network uses the previous shifted step to predict the next one. User channels are substituted from observations; own channels from generation.

| Channel | Number | Offset |
|---|---|---|
| aether text | 0 | 0 |
| First aether code | 1 | 0 |
| Remaining aether codes | 2…8 | 1 |
| First user code | 9 | 0 |
| Remaining user codes | 10…16 | 1 |

Consequently, a single inner step can contain the first code of a new frame together with acoustic refinements of the previous one. The decoder only receives the complete set for one physical frame.

Example of reverse assembly: for output frame `t`, the first code is extracted from shifted position `t`, and the refinements from `t+1`. The first finished output requires the delay to be filled. Initial placeholder values are never voiced.

A single schedule function is used by training, offline inference, and live generation alike. Verification is performed on artificial unique frame indices, so that the offset is visible without listening to audio.

## 7. Text and special values

Initial design contract: audio BOS = 2048; text BOS = 32000; text padding = 3; end-of-text-padding marker = 0. The real tokenizer is verified against these values. Input embedding tables account for BOS; audio output does not generate BOS as an ordinary code.

A missing position and a not-yet-generated position are represented by distinct internal sentinel values. Neither participates in ordinary lookup or in loss without a mask.

An empty text step does not mean the absence of audio. The voice may continue a word, produce a non-verbal sound, or remain silent. Alignment of text subtokens with frames is preserved from the data rather than reconstructed from string length.

## 8. Sampling

Text and audio have independent temperature and top-k. The initial experimental pair: text `0.7 / 25`, audio `0.8 / 250`. For numerical tests, greedy decoding or a fixed RNG state is used.

Changing temperature is not considered a way to increase the model's knowledge. Any tuning is evaluated on content, intelligibility, and naturalness. On non-numeric logits, the session terminates with an error; silent substitution of random codes is not permitted.

## 9. Memory and computation

BF16 requires approximately two bytes per weight parameter, excluding caches, temporary tensors, and the codec. This is an arithmetic lower bound, not a requirement on available GPU memory.

For ordinary multi-head attention, a rough estimate of the KV cache:

```text
bytes ≈ B × layers × context × 2 × kv_heads × head_dim × bytes_per_element
```

For GQA, a quantized cache, or other storage, the formula changes. Peak allocated, reserved, and external process memory are measured after warm-up. Maximum batch size cannot be determined from the weight file size alone.

## 10. Reproduction boundary

Until the checkpoint is confirmed, the following remain open: full FFN sizes, weight layout, codec architecture, exact training alignment scheme, vocabulary completeness, context boundaries, and source training data. The documentation fixes checkpoints but does not replace these artifacts. Implementation of the large model begins only after these are fixed; the small test configuration can be created earlier.

**Future codec gate (user decision, 2026-09-16):** aether is to be fully
implemented and verified with the initial codec first. The transition to the Manifestro codec
is deferred to the second stage of its development, even given a successful aether baseline.
Dependencies, adapters, and migration for the future codec are not being added at this time.
The shared Protocol fixes the interface of the initial codec, not integration of the future one.
