# aether — local artifact verification

A07 is partially complete. Verification of a synthetic bundle without a backend has been implemented:

```sh
uv run --locked aether inspect --config configs/model/tiny.json --manifest /path/to/manifest.json
```

The CLI outputs a table verifying the schema, configuration, files/SHA-256, and parameters.
Exit code 0 confirms only the synthetic contract; `NOT_VERIFIED` explicitly indicates
that model readiness and tokenizer semantics have not been verified. Errors produce exit code 2.
There is no working weight bundle in the repository; small fixtures are created by the tests.

`BundleManifest` version 1 contains:

- `layout_version=1`, `format=synthetic_json_v1`, a non-empty `provenance`, `dtype`;
- complete `model` and `codec` fields matching the requested configuration;
- `config`, `tokenizer`, `weights`: distinct relative paths, size, and SHA-256;
- `parameters`: the exact name, shape, and dtype of each parameter.

The config file contains a complete ExperimentConfig. Synthetic weights are JSON:
`{"parameters":{"layer.weight":{"shape":[2,2],"dtype":"float32","values":[1.0,2.0,3.0,4.0]}}}`.
The number of values must match the product of the shape; all numbers are finite.
The tokenizer is checked for presence and hash, but its actual special tokens and
semantics are not yet verified. Provenance is recorded but not authenticated:
SHA-256 verifies integrity relative to the manifest, not trust in its author.

Paths outside the bundle, symlinks, duplicate JSON keys, unknown fields/versions,
and missing or extra parameters are rejected. The manifest is limited to 1 MiB, and
the entire verified bundle to 16 MiB. This path is not intended for real weights.
Pickle, downloads, and automatic layout/backend conversions are not used.

`load_synthetic_bundle` first verifies the bundle and the exact equality of the
`ParameterTarget` specifications, then calls `install_parameters` exactly once. The
adapter is responsible for atomic installation; a test mock confirms that no calls
occur on error. This is a loader for synthetic values, not an implementation of
model state_dict loading.

Completing A07 requires selected weights and a backend, a full architectural
contract, real tokenizer verification, and strict state_dict matching. A layout
converter is added only for a specific, documented version pair. Unknown
layouts are rejected for now. Full-weight verification will be performed on
remote Colab later.
