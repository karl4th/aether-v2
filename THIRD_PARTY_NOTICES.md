# aether — third-party notices

External names in this file and integration metadata are retained for required
provenance and attribution. They do not rename the aether project.

## Runtime and base weights

The Python runtime dependency is Moshi by Kyutai, pinned to source commit
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`:
https://github.com/kyutai-labs/moshi/tree/e6a55d2722a65870ef52a6c9f6ecfc0e90f38362
Copyright (c) Kyutai, all rights reserved. Python source is MIT licensed.
Its installed distribution retains the upstream notices. The initial codec is Mimi.
The aether low-rank adapter re-expresses the final gated feedforward algebra so
its projection is trainable; original upstream weights and initial codec remain frozen.

Base voice model, codec weights and text tokenizer come from
https://huggingface.co/kyutai/moshiko-pytorch-bf16/tree/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289
Revision: `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`.
Weights are published under Creative Commons Attribution 4.0:
https://creativecommons.org/licenses/by/4.0/
No base weights are stored in this repository. An exported aether adapter is a
modification and must keep these attribution notices when shared with the base.

Citation: Alexandre Défossez, Laurent Mazaré, Manu Orsini, Amélie Royer,
Patrick Pérez, Hervé Jégou, Edouard Grave and Neil Zeghidour (2024),
“Moshi: a speech-text foundation model for real-time dialogue”.
https://arxiv.org/abs/2410.00037

## English read-speech data

Mini LibriSpeech, OpenSLR SLR31, subset of LibriSpeech by Vassil Panayotov,
Guoguo Chen, Daniel Povey and Sanjeev Khudanpur (2015).
https://www.openslr.org/31/
CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
The preparation pipeline selects short recordings; the model pipeline resamples
and crops them for the initial codec. It does not provide dialogue annotations.
Dataset manifests and exports retain source attribution and integrity identifiers.

## Upstream Python MIT permission notice

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
