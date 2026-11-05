import argparse
import logging
import os
import sys
import warnings

# Configuring the environment and logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Disable unnecessary TensorFlow logs
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disabling Gradio analytics
logging.basicConfig(level=logging.WARNING)  # Disable all logs, except WARNING and above
warnings.filterwarnings("ignore")  # Disable all warnings

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

from assets.model_installer import check_and_install_models
from rvc.inference.infer import rvc_infer


def strtobool(val: str) -> bool:
    lower_val = val.lower()
    if lower_val in ("y", "yes", "t", "true", "on", "1"):
        return True

    if lower_val in ("n", "no", "f", "false", "off", "0"):
        return False

    raise ValueError(f"Недопустимое значение: {val!r}")


def create_parser():
    # Базовый парсер с общими аргументами
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument(
        "--rvc_model", type=str, required=True, help="Название RVC модели"
    )
    base.add_argument(
        "--f0_method", type=str, default="rmvpe", help="Метод извлечения F0"
    )
    base.add_argument("--f0_min", type=int, default=50, help="Минимальная частота F0")
    base.add_argument(
        "--f0_max", type=int, default=1100, help="Максимальная частота F0"
    )
    base.add_argument(
        "--rvc_pitch", type=float, default=0, help="Высота тона RVC модели"
    )
    base.add_argument("--protect", type=float, default=0.5, help="Защита согласных")
    base.add_argument("--index_rate", type=float, default=0, help="Коэффициент индекса")
    base.add_argument(
        "--volume_envelope", type=float, default=1, help="Огибающая громкости"
    )
    base.add_argument(
        "--output_format", type=str, default="mp3", help="Формат выходного файла"
    )

    # Главный парсер с субкомандами
    parser = argparse.ArgumentParser(
        description="Инструмент для замены голоса при помощи RVC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Субкоманда для RVC
    rvc = subparsers.add_parser("rvc", parents=[base], help="Конвертация аудио-файла")
    rvc.add_argument("--input_path", type=str, required=True, help="Путь к аудио-файлу")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    common_params = {
        "rvc_model": args.rvc_model,
        "f0_method": args.f0_method,
        "f0_min": args.f0_min,
        "f0_max": args.f0_max,
        "rvc_pitch": args.rvc_pitch,
        "protect": args.protect,
        "index_rate": args.index_rate,
        "volume_envelope": args.volume_envelope,
        "output_format": args.output_format,
    }

    if args.command == "rvc":
        rvc_infer(**common_params, input_path=args.input_path)

    print("\033[1;92m\nГолос успешно заменен!\n\033[0m")


if __name__ == "__main__":
    check_and_install_models()  # Checking and installing models
    main()
