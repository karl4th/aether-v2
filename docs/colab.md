# aether — full run in Colab

`notebooks/aether_colab.ipynb` fetches the code via Git fetch and a detached
checkout, and creates a Python 3.12.14 `.venv` via uv from a pinned lock file.
No archive upload is required. For a private repository, a read-only Colab
Secret `GITHUB_TOKEN` is used.

## Launch parameters

- Full inference + limited adaptation scenario: a remote managed paid Colab
  with an A100 40 GB or larger. A minimum of 38 GiB of free VRAM is checked.
- Inference only: a BF16 GPU with at least 22 GiB of free VRAM, e.g. L4/A100.
- The T4 from the user's report (15360 MiB) is not supported by the BF16
  profile. The rejection happens before weight loading; setting the flag does
  not bypass the resource check.
- `CONFIRM_REMOTE_PAID_COLAB=True` confirms that the correct runtime has been
  selected.
- `RUN_TRAINING=False` runs inference and baseline only; `True` additionally
  runs the pilot, resume, adaptation, re-evaluation, playback, and export.
- `UPLOAD_QUESTION=True` uploads a short custom English question in
  WAV/FLAC. When False, a held-out reading-evaluation recording is used
  instead.
- `GIT_REF=main` retrieves the current commit. For resume, the exact
  `source_revision` from `runs/<RUN_ID>/run.json` and the `RESUME_CHECKPOINT`
  path are specified.
- `RUN_ID` is unique; all results go to Google Drive at
  `aether/runs/<RUN_ID>`.

A separate preflight report does not need to be submitted: diagnostics are
built in immediately before model operations. GPU thresholds do not
guarantee the absence of OOM. On a memory error, frames is reduced to 32 for
a new experiment, or a GPU with more VRAM is selected; for an exact resume,
the configuration stays unchanged.

## What the notebook executes

1. Loading of the pinned voice model bundle, base codec, and tokenizer.
   Safetensors, strict loading via the pinned runtime; SHA256/file sizes,
   revision, seed, and context are recorded in inference.json.
2. Loading of the published English reading corpus, verification of
   official checksums, selection of 16 train / 4 evaluation utterances
   2-5 seconds long, and a check that speakers do not overlap. Working
   files reside on the VM; the manifest/attribution reside on Drive.
3. Streaming encoding, text/audio generation, and playback of the result
   directly in the notebook. Each generation has its own separate streaming
   state.
4. Teacher-forced audio CE and perplexity on the evaluation set before
   adaptation.
5. When RUN_TRAINING=True: a rank-8 adapter on the output of the last
   temporal FFN, with a frozen base and codec, batch=1, 64 frames, 20
   optimizer steps, lr=1e-4, clipping=1. The pilot stops after 5 steps; the
   next run loads the checkpoint and continues the same plan up to 20. The
   training loop is capped at 1800 seconds; loading and setup are not
   included in this limit.
6. Re-evaluation and generation with the adapter, and export of the
   verified checkpoint.

The data source does not contain word timestamps. The text channel is
masked; loss is computed only over valid audio codes. This is a limited
speech adaptation, not training for dialogue, knowledge, or the English
language. An improvement in CE does not prove an improvement in
conversational quality. Substantive evaluation remains manual.

## Save and resume

The checkpoint contains only the adapter, optimizer/scheduler/RNG state,
history, and metadata; base weights are not saved again. The file is first
created on the VM, then the copy is verified and marked COMPLETE on Drive.
Resume accepts only a confirmed checkpoint with the same code, backend/base
revision, training config, and dataset SHA256. Audio is reloaded from the
same verified archives; its location does not change its identity. State
loading uses `weights_only=True`.

To continue after the VM disconnects, the same GIT_REF, a new RUN_ID, the
RESUME_CHECKPOINT on Drive, and RUN_TRAINING=True are used and the notebook
is run again. Old results are not overwritten. Source terms-of-use files
are stored alongside the results; legal attribution is in
THIRD_PARTY_NOTICES.md.

## Boundary of what has been verified

Schemas, the CLI, mocks, masks, resume identities, integrity checks, Git
checkout, and cell compilation are verified locally. There have been no
real optimizer steps, full weight loading, or GPU forward passes on the
current machine. The code and notebook are ready for manual remote
testing; successful training and quality have not yet been claimed.

The user's report has already confirmed a T4 15360 MiB, driver 580.82.07,
53467192 KiB of RAM, and about 202 GB of free disk space. The absence of
google.colab in the uv process is normal. The gate uses Linux, several
Colab environment observations, the NVIDIA driver, permissions tied to the
boot ID/live kernel, and a separate training permission. This is a
safeguard against accidental local execution, not proof of a paid tier.
The 500-unit budget is controlled by the user in the UI; the package
limits steps/time but does not track Colab unit consumption. Once
finished, the GPU runtime is disconnected.

Full conversational evaluation remains open. The notebook itself provides
an offline audio demo with playback, not live full-duplex; a separate live
session (below) covers that path.

## Live session (optional, after a successful baseline)

A remote Colab VM has no public IP, so reaching its GPU-resident model over
a live WebSocket connection needs a tunnel. `aether tunnel` wraps a
Cloudflare quick tunnel, which needs no account, no DNS record, and no
inbound firewall change; it is unauthenticated and disposable, so it is
suitable for a short interactive test, not a persistent endpoint.

In the same Colab session that already has the environment set up (so the
same pinned weights are reused rather than downloaded twice):

```bash
# One-time per runtime: install the cloudflared binary.
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
!chmod +x /usr/local/bin/cloudflared

# Start the live backend in the background, bound to localhost only.
!uv run --locked --group model aether serve --permit runtime-permit.json --port 8080 &

# Expose it and print the public wss:// endpoint.
!uv run --locked aether tunnel --port 8080
```

`aether tunnel` prints `TUNNEL_READY: https://<random>.trycloudflare.com`
and the matching `wss://.../v1/session` endpoint. From a local machine with
a microphone and speakers (not from inside Colab):

```bash
uv run --locked --group audio aether talk --url wss://<random>.trycloudflare.com/v1/session
```

Test in headphones before speakers, since echo cancellation has not been
implemented. Closing either the `aether serve` process or the tunnel ends
the session; the hostname stops resolving as soon as the tunnel process
exits.
