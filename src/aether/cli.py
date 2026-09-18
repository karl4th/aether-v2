"""Backend-free CLI parsing; real GPU operations require a live notebook permit."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aether import __version__
from aether.config import _unique_object, load_config
from aether.config.adaptation import TrainingConfig
from aether.config.operations import InferenceConfig
from aether.remote import RemoteRuntimeError, require_remote_runtime


def read_json(path: str) -> str:
    raw = Path(path).read_text(encoding="utf-8")
    if not isinstance(json.loads(raw, object_pairs_hook=_unique_object), dict):
        raise ValueError("configuration must be a JSON object")
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aether",
        description="aether — streaming dialogue model research.",
        epilog="Local: development and tests. Model and training: remote Google Colab.",
    )
    parser.add_argument("--version", action="version", version=f"aether {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Validate a synthetic bundle.")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--manifest", required=True)
    data = commands.add_parser("prepare-data", help="Prepare the English corpus in Colab.")
    data.add_argument("--output", required=True)
    data.add_argument("--permit", required=True)
    data.add_argument("--max-train", type=int, default=16)
    data.add_argument("--max-eval", type=int, default=4)
    data.add_argument("--max-seconds", type=float, default=5.0)
    for name in ("infer", "train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--permit")
        command.add_argument("--output")
        command.add_argument("--checkpoint")
        if name == "infer":
            command.add_argument("--input", required=True)
        else:
            command.add_argument("--dataset")
        if name == "train":
            command.add_argument("--validate-only", action="store_true")
            command.add_argument("--resume")
            command.add_argument("--stop-after-steps", type=int)
    server = commands.add_parser("serve", help="Run the guarded live WebSocket backend in Colab.")
    server.add_argument("--permit", required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    server.add_argument("--token-env", default="AETHER_LIVE_TOKEN")
    server.add_argument("--checkpoint")
    client = commands.add_parser("talk", help="Open the microphone/speaker and a live client.")
    client.add_argument("--url", required=True)
    client.add_argument("--duration", type=float)
    client.add_argument("--seed", type=int, default=42)
    client.add_argument("--token-env", default="AETHER_LIVE_TOKEN")
    tunnel = commands.add_parser("tunnel", help="Expose a local port via a Cloudflare quick tunnel")
    tunnel.add_argument("--port", type=int, default=8080)
    return parser


def _remote_operation(args: argparse.Namespace) -> dict[str, Any]:
    if not args.permit:
        code = (
            "TRAINING_ENVIRONMENT_REQUIRED"
            if args.command == "train"
            else "REMOTE_RUNTIME_REQUIRED"
        )
        raise RemoteRuntimeError(f"{code}: notebook permit is required; local runtime is forbidden")
    require_remote_runtime(training=args.command == "train", permit_path=args.permit)
    if args.command == "prepare-data":
        from aether.dataset import prepare_dataset

        def authorize_download() -> None:
            require_remote_runtime(permit_path=args.permit)

        result = prepare_dataset(
            Path(args.output),
            authorize_download=authorize_download,
            max_train=args.max_train,
            max_eval=args.max_eval,
            max_seconds=args.max_seconds,
        )
        return {
            "manifest": str(Path(args.output) / "dataset.json"),
            "train": len(result.train),
            "evaluation": len(result.evaluation),
        }
    if args.command == "infer":
        from aether.backend import infer_file

        config = InferenceConfig.model_validate_json(read_json(args.config))
        if not args.output:
            raise ValueError("--output directory is required")
        return infer_file(
            args.input,
            args.output,
            args.permit,
            tail_seconds=config.tail_seconds,
            context=config.context,
            seed=config.seed,
            adapter_checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        )
    from aether.dataset import load_manifest
    from aether.trainer import evaluate, train

    training = TrainingConfig.model_validate_json(read_json(args.config))
    if not args.dataset:
        raise ValueError("--dataset manifest is required")
    dataset = load_manifest(Path(args.dataset))
    if args.command == "train":
        if not args.output:
            raise ValueError("--output run directory is required")
        return train(
            dataset,
            Path(args.output),
            permit_path=Path(args.permit),
            config=training,
            resume=Path(args.resume) if args.resume else None,
            stop_after_steps=args.stop_after_steps,
        )
    return evaluate(
        dataset,
        permit_path=Path(args.permit),
        config=training,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "serve":
            from aether.server import serve

            serve(
                args.permit,
                host=args.host,
                port=args.port,
                token=os.environ.get(args.token_env),
                checkpoint=args.checkpoint,
            )
            return 0
        if args.command == "talk":
            from aether.client import talk

            asyncio.run(
                talk(
                    args.url,
                    duration=args.duration,
                    seed=args.seed,
                    auth_token=os.environ.get(args.token_env),
                )
            )
            return 0
        if args.command == "tunnel":
            from aether.tunnel import run_tunnel

            run_tunnel(args.port)
            return 0
        if args.command == "inspect":
            from aether.artifacts import inspect_bundle

            manifest = inspect_bundle(args.manifest, load_config(args.config))
            print(
                f"CHECK\tRESULT\nschema/files/SHA-256/parameters ({len(manifest.parameters)})"
                "\tPASS\nmodel readiness/tokenizer semantics\tNOT_VERIFIED"
            )
            return 0
        if args.command == "train" and args.validate_only:
            raw = read_json(args.config)
            if json.loads(raw).get("task") == "speech_adaptation":
                TrainingConfig.model_validate_json(raw)
            else:
                load_config(args.config)
            print("VALID_CONFIG: schema verified without backend; runtime not checked")
            return 0
        result = _remote_operation(args)
        if args.command == "evaluate" and args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except RemoteRuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        code = (
            "INVALID_BUNDLE"
            if args.command == "inspect"
            else (
                "INVALID_CONFIG"
                if args.command == "train" and args.validate_only
                else "OPERATION_FAILED"
            )
        )
        print(f"{code}: {exc}", file=sys.stderr)
        return 2
