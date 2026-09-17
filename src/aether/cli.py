"""CLI scaffold. Unimplemented operations fail without performing any model work."""

import argparse
import sys
from collections.abc import Sequence

from aether import __version__
from aether.config import load_config
from aether.training_gate import TrainingBlockedError, require_training_environment

EXIT_NOT_IMPLEMENTED = 3
EXIT_TRAINING_BLOCKED = 4


def build_parser() -> argparse.ArgumentParser:
    """Describe future commands without loading models or opening audio devices."""
    parser = argparse.ArgumentParser(
        prog="aether",
        description="aether — потоковый голосовой диалог (каркас проекта).",
        epilog="Локально: разработка и тесты. Обучение: только удалённый Google Colab.",
    )
    parser.add_argument("--version", action="version", version=f"aether {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "inspect": "Проверить локальный синтетический bundle (без backend).",
        "infer": "Обработать аудиофайл (ещё не реализовано).",
        "serve": "Запустить сервер (ещё не реализовано).",
        "talk": "Подключить аудиоклиент (ещё не реализовано).",
        "evaluate": "Оценить модель (ещё не реализовано).",
        "train": "Проверка конфигурации; обучение закрыто до подтверждения среды Colab.",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description)
        if name == "talk":
            command.add_argument("--url", required=True)
        else:
            command.add_argument("--config", required=True)
        if name == "inspect":
            command.add_argument("--manifest", required=True)
        if name == "infer":
            command.add_argument("--input", required=True)
            command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument(
                "--validate-only",
                action="store_true",
                help="Проверить конфигурацию без backend, загрузки весов и обучения.",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Return a nonzero status for unavailable work; never start training."""
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        from aether.artifacts import inspect_bundle

        try:
            manifest = inspect_bundle(args.manifest, load_config(args.config))
        except (OSError, ValueError) as exc:
            print(f"INVALID_BUNDLE: {exc}", file=sys.stderr)
            return 2
        print("CHECK\tRESULT")
        print("schema/config/codec\tPASS")
        print("files/size/SHA-256\tPASS")
        print(f"synthetic parameters ({len(manifest.parameters)})\tPASS")
        print("model readiness/tokenizer semantics\tNOT_VERIFIED")
        return 0
    if args.command == "train":
        if not args.validate_only:
            try:
                require_training_environment()
            except TrainingBlockedError as exc:
                print(str(exc), file=sys.stderr)
                return EXIT_TRAINING_BLOCKED
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            print(f"INVALID_CONFIG: {exc}", file=sys.stderr)
            return 2
        print(
            f"VALID_CONFIG: schema={config.schema_version} profile={config.profile}; "
            "только схема, готовность runtime и модели не проверена."
        )
        return 0
    print(
        f"NOT_IMPLEMENTED: {args.command} ещё не реализован; никаких действий не выполнено.",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED
