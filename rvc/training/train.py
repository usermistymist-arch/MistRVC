import logging
import os
import random
import sys
import warnings

# Настройка окружения
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"

# Настройка логирования и подавление предупреждений
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")
# Страховка для режима warn_only=True: такие предупреждения называют операции
# без детерминированной реализации. При warn_only=False они не нужны — там падение.
warnings.filterwarnings("default", message=".*deterministic implementation.*")


# TensorBoard тянет предустановленный в Colab/Kaggle TensorFlow, чей импорт сыплет
# «Could not find cuda drivers» (TF_CPP_MIN_LOG_LEVEL эти строки не глушит).
# Заглушка notf переключает TensorBoard на tensorflow_stub. В ноутбуке это делает
# ячейка установки; здесь — страховка для запуска вне ноутбука (Kaggle, CLI).
def _disable_tensorflow():
    import importlib.util

    spec = importlib.util.find_spec("tensorboard")
    if spec and spec.origin:
        notf = os.path.join(os.path.dirname(spec.origin), "compat", "notf.py")
        if not os.path.exists(notf):
            try:
                with open(notf, "w", encoding="utf-8") as f:
                    f.write("# tensorflow_stub\n")
            except OSError:
                pass


_disable_tensorflow()


import argparse
import json
from collections import defaultdict
from random import randint
from time import time as ttime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from rvc._library.algorithm.commons import grad_norm, slice_segments
from rvc._library.algorithm.discriminators import MultiPeriodDiscriminator
from rvc._library.algorithm.synthesizers import Synthesizer
from rvc.training.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from rvc.training.mel_processing import (
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from rvc.training.utils.data_utils import (
    DistributedBucketSampler,
    TextAudioCollateMultiNSFsid,
    TextAudioLoaderMultiNSFsid,
)
from rvc.training.utils.probe import dataset_info, setup_probe
from rvc.training.utils.train_utils import (
    HParams,
    extract_model,
    load_checkpoint,
    save_checkpoint,
)
from rvc.training.visualization import (
    f0_cents_errors,
    f0_error_cents,
    mel_spectrogram_similarity,
    plot_spectrogram_to_numpy,
)

torch.backends.cudnn.deterministic = False
if (
    os.name == "nt"
):  # Включение только на Windows (в Kaggle и Colab снижает производительность)
    torch.backends.cudnn.benchmark = True

# Настройки TF32, в некоторых случаях должны повысить производительность
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception as e:
    print(f"Torch TF32: {e}")

global_step = 0


def _strtobool(val):
    return val.lower() in ("yes", "true", "t", "y", "on", "1")


def _format_duration(seconds):
    """Секунды -> '45.2 с' или '3 мин 06.0 с'."""
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes = int(seconds // 60)
    return f"{minutes} мин {seconds - minutes * 60:04.1f} с"


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--total_epoch", type=int, choices=range(1, 10001), default=300)
    parser.add_argument(
        "--save_every_epoch", type=int, choices=range(1, 201), default=25
    )
    parser.add_argument("--batch_size", type=int, choices=range(1, 129), default=8)
    parser.add_argument(
        "--sample_rate", type=int, choices=[32000, 40000, 48000], default=48000
    )
    parser.add_argument("--pretrain_g", type=str, default=None)
    parser.add_argument("--pretrain_d", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--save_to_zip", type=_strtobool, choices=[True, False], default=False
    )
    parser.add_argument(
        "--save_half", type=_strtobool, choices=[True, False], default=True
    )
    parser.add_argument("--d_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--d_grad_max_norm", type=float, default=0.0)
    parser.add_argument("--d_label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--probe", type=str, choices=["off", "fixed", "holdout"], default="fixed"
    )
    parser.add_argument("--probe_size", type=int, default=0)
    args = parser.parse_args()
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed должен быть целым числом от 0 до 4294967295")

    experiment_dir = os.path.join(args.experiment_dir, args.model_name)
    config_save_path = os.path.join(experiment_dir, "data", "config.json")

    # Генерация файла конфигурации
    config_existed = os.path.isfile(config_save_path)
    if not config_existed:
        from rvc._library.config import generate_config

        generate_config(config_save_path, args.sample_rate)

    # Загрузка файла конфигурации
    with open(config_save_path, encoding="utf-8") as f:
        config = json.load(f)

    saved_vocoder = config.get("model", {}).get("vocoder", "HiFi-GAN")
    if saved_vocoder != "HiFi-GAN":
        parser.error(
            f"Модель настроена на вокодер «{saved_vocoder}», который больше не поддерживается. "
            "MistRVC обучает только HiFi-GAN."
        )

    train_config = config.setdefault("train", {})
    saved_seed = int(train_config.get("seed", 1032003))
    seed_locked = (
        config_existed
        or bool(train_config.get("seed_locked", False))
        or os.path.isfile(os.path.join(experiment_dir, "checkpoint.pth"))
    )
    seed_ignored = None
    if seed_locked:
        # Конфиг мог приехать без поля seed: дописываем фолбэк, иначе
        # hparams.train.seed не будет существовать вовсе.
        if "seed" not in train_config:
            train_config["seed"] = saved_seed
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        if args.seed is not None and args.seed != saved_seed:
            seed_ignored = int(args.seed)
            print(
                f"⚠️ Сид модели «{args.model_name}» зафиксирован: {saved_seed}. "
                f"Передан {seed_ignored} — менять сид посреди тренировки нельзя.\n"
                f"Обучение принудительно продолжится с сидом {saved_seed} "
                f"(хранится в {config_save_path}, поле train.seed). "
                "Для другого сида создайте новую модель.",
                flush=True,
            )
    else:
        train_config["seed"] = saved_seed if args.seed is None else int(args.seed)
        train_config["seed_locked"] = True
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    hparams = HParams(**config)
    hparams.model_dir = experiment_dir
    hparams.model_name = args.model_name
    hparams.total_epoch = args.total_epoch
    hparams.save_every_epoch = args.save_every_epoch
    hparams.batch_size = args.batch_size
    hparams.pretrain_g = args.pretrain_g
    hparams.pretrain_d = args.pretrain_d
    hparams.gpus = args.gpus
    hparams.save_to_zip = args.save_to_zip
    hparams.save_half = args.save_half
    hparams.d_lr_multiplier = args.d_lr_multiplier
    hparams.d_grad_max_norm = args.d_grad_max_norm
    hparams.d_label_smoothing = args.d_label_smoothing
    hparams.deterministic = args.deterministic
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    train_files, eval_files, probe_note = setup_probe(
        experiment_dir,
        mode=args.probe,
        size=args.probe_size,
        batch_size=args.batch_size,
    )
    hparams.data.training_files = train_files
    hparams.data.eval_files = eval_files

    print(" \n\nПАРАМЕТРЫ ОБУЧЕНИЯ ")
    print("=" * 70)
    print(f"{'Папка сохранения:':<30} {hparams.model_dir}")
    print(f"{'Имя модели:':<30} {hparams.model_name}")
    total_sec, segment_sec = dataset_info(experiment_dir)
    dataset_note = _format_duration(total_sec) if total_sec is not None else "—"
    segment_note = f"{segment_sec:.1f} сек" if segment_sec is not None else "—"
    print(f"{'Длина датасета:':<30} {dataset_note}")
    print(f"{'Длина сегмента:':<30} {segment_note}")
    print(f"{'Эпох обучения:':<30} {hparams.total_epoch}")
    print(f"{'Сохранение каждые:':<30} {hparams.save_every_epoch} эпох")
    print(f"{'Размер батча:':<30} {hparams.batch_size}")
    print(f"{'Частота дискретизации:':<30} {hparams.data.sample_rate} Hz")
    print(f"{'Вокодер:':<30} HiFi-GAN")
    print(f"{'Оптимизатор:':<30} AdamW")
    if args.pretrain_g:
        print(f"{'Претрейн G:':<30} {hparams.pretrain_g}")
    if args.pretrain_d:
        print(f"{'Претрейн D:':<30} {hparams.pretrain_d}")
    print(f"{'Сохранение в ZIP:':<30} {'Да' if hparams.save_to_zip else 'Нет'}")
    print(f"{'Точность моделей:':<30} {'float16' if hparams.save_half else 'float32'}")
    print(f"{'Скорость обучения D:':<30} {hparams.d_lr_multiplier}")
    print(f"{'Потолок градиентов D:':<30} {hparams.d_grad_max_norm}")
    print(f"{'Сглаживание меток D:':<30} {hparams.d_label_smoothing}")
    seed_note = ""
    if seed_ignored is not None:
        seed_note = f" — переданный {seed_ignored} отклонён"
    print(f"{'Сид:':<30} {hparams.train.seed}{seed_note}")
    print(
        f"{'Детерминированное обучение:':<30} "
        f"{'Да' if hparams.deterministic else 'Нет'}"
    )
    print(f"{'Проверочный набор:':<30} {probe_note}")
    print("=" * 70 + "\n")
    return hparams


class MetricsAccumulator:
    """Среднее по батчам за интервал log_interval.

    Создаётся один раз в run() и живёт между эпохами; обнуляется методом
    reset() после записи метрик. При log_interval = 1 интервал равен эпохе.
    """

    def __init__(self):
        self.sums = defaultdict(float)
        self.weights = defaultdict(float)
        self.count = 0

    def update(self, **kwargs):
        self.count += 1
        for k, v in kwargs.items():
            self.sums[k] += v.item() if hasattr(v, "item") else v

    def update_weighted(self, key, value, weight):
        """Метрика со своим весом (например, KL — по числу валидных кадров)."""
        value = value.item() if hasattr(value, "item") else value
        weight = weight.item() if hasattr(weight, "item") else weight
        self.sums[key] += value * weight
        self.weights[key] += weight

    def reduce_across_ranks(self, device_id):
        """Суммирует накопленное по всем ранкам: метрики описывают все данные.

        Коллективная операция — вызывается на всех ранках одновременно.
        После неё average() считается по батчам всех GPU, а не только rank 0.
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        keys = sorted(self.sums)
        payload = (
            [self.sums[k] for k in keys]
            + [self.weights.get(k, 0.0) for k in keys]
            + [float(self.count)]
        )
        # NCCL требует тензоры на GPU, gloo работает с CPU
        target = (
            torch.device(f"cuda:{device_id}")
            if dist.get_backend().lower() == "nccl"
            else torch.device("cpu")
        )
        tensor = torch.tensor(payload, dtype=torch.float64, device=target)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        values = tensor.tolist()
        n = len(keys)
        for i, k in enumerate(keys):
            self.sums[k] = values[i]
            if k in self.weights:
                self.weights[k] = values[n + i]
        self.count = int(values[2 * n])

    def average(self):
        if not self.count:
            return {}
        # Взвешенные метрики делятся на свой вес, остальные — на число батчей
        return {
            k: v / (self.weights.get(k, 0) or self.count) for k, v in self.sums.items()
        }

    def reset(self):
        self.sums.clear()
        self.weights.clear()
        self.count = 0


def main():
    hps = get_hparams()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    )
    gpus = [int(item) for item in hps.gpus.split("-")] if device.type == "cuda" else [0]
    n_gpus = len(gpus)
    if device.type == "cpu":
        print("Обучение с использованием процессора займёт много времени.", flush=True)

    children = []
    for rank, device_id in enumerate(gpus):
        subproc = mp.Process(
            target=run,
            args=(hps, rank, n_gpus, device, device_id),
        )
        children.append(subproc)
        subproc.start()

    for subproc in children:
        subproc.join()
        # исключение в воркере до родителя не доходит, процесс может быть убит по сигналу
        if subproc.exitcode != 0:
            print(f"Обучение прервалось, код завершения {subproc.exitcode}", flush=True)
            sys.exit(1)

    sys.exit(0)


def run(hps, rank, n_gpus, device, device_id):
    global global_step

    try:
        writer_eval = (
            SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
            if rank == 0
            else None
        )
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=hps.data.sample_rate)

        dist.init_process_group(
            backend="gloo"
            if sys.platform == "win32" or device.type != "cuda"
            else "nccl",
            init_method="env://",
            world_size=n_gpus if device.type == "cuda" else 1,
            rank=rank if device.type == "cuda" else 0,
        )

        torch.manual_seed(hps.train.seed)
        random.seed(hps.train.seed)
        np.random.seed(hps.train.seed)
        if hps.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=False)
            if rank == 0:
                print(
                    "Детерминированный режим: TF32 выключен, cudnn.deterministic=True. "
                    "Обучение будет медленнее.",
                    flush=True,
                )
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

        collate_fn = TextAudioCollateMultiNSFsid()
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data)
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.batch_size,
            [50, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
            seed=hps.train.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            num_workers=2,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=8,
        )

        eval_batches = None
        if rank == 0 and hps.data.eval_files:
            eval_hps = HParams(**dict(hps.data.items()))
            eval_hps.training_files = hps.data.eval_files
            eval_dataset = TextAudioLoaderMultiNSFsid(eval_hps)
            if len(eval_dataset) > 0:
                eval_batches = [
                    collate_fn([eval_dataset[i]]) for i in range(len(eval_dataset))
                ]

        net_g = Synthesizer(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            sr=hps.data.sample_rate,
            checkpointing=False,
            randomized=True,
        )
        net_d = MultiPeriodDiscriminator(checkpointing=False)

        if device.type == "cuda":
            net_g = net_g.cuda(device_id)
            net_d = net_d.cuda(device_id)
        else:
            net_g = net_g.to(device)
            net_d = net_d.to(device)

        optim_g = torch.optim.AdamW(
            net_g.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            hps.train.learning_rate * hps.d_lr_multiplier,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

        if n_gpus > 1 and device.type == "cuda":
            net_g = DDP(net_g, device_ids=[device_id])
            net_d = DDP(net_d, device_ids=[device_id])

        # Загрузка чекпоинта
        epoch_str = None
        checkpoint_path = os.path.join(hps.model_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            try:
                epoch_str = load_checkpoint(
                    checkpoint_path, net_g, optim_g, net_d, optim_d
                )
            except Exception as e:
                print(f"Ошибка загрузки checkpoint.pth:\n{e}", flush=True)

        if epoch_str is not None:
            epoch_str += 1
            global_step = (epoch_str - 1) * len(train_loader)
        else:
            epoch_str = 1
            global_step = 0

            # Загрузка претрейнов если чекпоинт не найден
            if hps.pretrain_g not in ("", "None", None):
                if rank == 0:
                    print(
                        f"Загрузка претрейна генератора: '{hps.pretrain_g}'", flush=True
                    )
                g_model = net_g.module if hasattr(net_g, "module") else net_g
                g_model.load_state_dict(
                    torch.load(hps.pretrain_g, map_location="cpu", weights_only=True)[
                        "model"
                    ]
                )

                # Переинициализация строк emb_g
                # Default претрейн содержит одинаковые строки emb_g (0 и 1): такие спикеры
                # получают одинаковое условие и звучат одним голосом. Разводим
                # строки своих спикеров случайной инициализацией того же масштаба.
                if rank == 0:
                    print("Переинициализация emb_g...", flush=True)
                speakers_path = os.path.join(hps.model_dir, "speakers.txt")
                if os.path.exists(speakers_path):
                    with open(speakers_path, encoding="utf-8") as file:
                        lines = [line for line in file if line.strip()]
                    n_spk = len(lines) if lines else 1
                else:
                    n_spk = 1
                n_spk = min(
                    n_spk, g_model.emb_g.weight.shape[0]
                )  # защита от >109 спикеров

                with torch.no_grad():
                    weight = g_model.emb_g.weight.data
                    rows = weight[:n_spk]
                    # Масштаб = средняя норма строки претрейна / sqrt(gin_channels)
                    # Это сохраняет тот же порядок величин, к которому привык декодер
                    scale = float(rows.norm(dim=1).mean()) / (weight.shape[1] ** 0.5)
                    rows.normal_(0.0, scale)

                if rank == 0:
                    print(
                        f"emb_g переинициализирован для {n_spk} спикеров (масштаб: {scale:.4f}).\n",
                        flush=True,
                    )

            if hps.pretrain_d not in ("", "None", None):
                if rank == 0:
                    print(
                        f"Загрузка претрейна дискриминатора: '{hps.pretrain_d}'",
                        flush=True,
                    )
                d_model = net_d.module if hasattr(net_d, "module") else net_d
                d_model.load_state_dict(
                    torch.load(hps.pretrain_d, map_location="cpu", weights_only=True)[
                        "model"
                    ]
                )

        # Настройка scheduler
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )

        if epoch_str > 1:
            steps_done = epoch_str - 1
            for optim in (optim_g, optim_d):
                for group in optim.param_groups:
                    group["lr"] = group["initial_lr"] * hps.train.lr_decay**steps_done

        # Проверка: не превышает ли загруженная эпоха целевую
        if epoch_str > hps.total_epoch:
            if rank == 0:
                print(
                    f"\n⚠️  Загруженный чекпоинт (эпоха {epoch_str - 1}) уже превышает указанное количество эпох ({hps.total_epoch}).",
                    flush=True,
                )
            return

        print("\nЗапуск процесса обучения модели...", flush=True)
        # Аккумулятор живёт между эпохами: при log_interval > 1 метрики
        # копятся за весь интервал, а не выбрасываются через эпоху.
        metrics_acc = MetricsAccumulator()
        for epoch in range(epoch_str, hps.total_epoch + 1):
            train_and_evaluate(
                hps,
                rank,
                epoch,
                [net_g, net_d],
                [optim_g, optim_d],
                train_loader,
                writer_eval,
                fn_mel_loss,
                device,
                device_id,
                metrics_acc,
                eval_batches,
            )
            scheduler_g.step()
            scheduler_d.step()
    finally:
        # Уничтожение группы процессов для корректного закрытия программы
        if dist.is_initialized():
            dist.destroy_process_group()


# Имена субдискриминаторов (порядок из discriminators.py)
_DISC_NAMES = ["S", "P2", "P3", "P5", "P7", "P11", "P17", "P23", "P37"]
# Мел-масштабы (порядок n_mels в MultiScaleMelSpectrogramLoss) → группа.
# coarse — огибающая энергии, mid — форманты, fine — гармоническая деталь.
# Внутри группы кривые почти совпадают, отдельные теги дублировали сумму.
_MEL_SCALE_GROUPS = {
    "5": "coarse",
    "10": "coarse",
    "20": "mid",
    "40": "mid",
    "80": "mid",
    "160": "fine",
    "320": "fine",
    "480": "fine",
}
# Зерно проверочного набора: константа, чтобы замер был одинаковым каждую эпоху
_PROBE_SEED = 0


def evaluate_probe(hps, net_g, eval_batches, device, device_id):
    """Метрики на фиксированном проверочном наборе.

    Набор неизменен и без мьютов, сегмент считается целиком (а не случайные
    36 кадров). Батчи заранее загружены в RAM (см. run()), с диска здесь
    ничего не читается. RNG форкается и сеется константой: шум генератора
    одинаков каждую эпоху, а поток случайности обучения не сдвигается.
    """
    g_model = net_g.module if hasattr(net_g, "module") else net_g
    was_training = g_model.training
    was_randomized = g_model.randomized
    g_model.eval()
    g_model.randomized = False

    l1_sum, l1_num = 0.0, 0
    f0_parts = []
    first_mels = None
    rng_devices = [device_id] if device.type == "cuda" else []
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(_PROBE_SEED)
            for info in eval_batches:
                if device.type == "cuda":
                    info = [t.cuda(device_id, non_blocking=True) for t in info]
                else:
                    info = [t.to(device) for t in info]
                (
                    phone,
                    phone_lengths,
                    pitch,
                    pitchf,
                    spec,
                    spec_lengths,
                    _,
                    _,
                    sid,
                ) = info
                y_hat = g_model(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                )[0]
                real_mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sample_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                fake_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sample_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                frames = min(real_mel.size(-1), fake_mel.size(-1))
                real_mel = real_mel[..., :frames]
                fake_mel = fake_mel[..., :frames]
                l1_sum += (fake_mel - real_mel).abs().sum().item()
                l1_num += real_mel.numel()
                errors = f0_cents_errors(
                    y_hat,
                    pitchf,
                    torch.zeros_like(spec_lengths),
                    hps.data.sample_rate,
                    hps.data.hop_length,
                    frames,
                )
                if errors is not None:
                    f0_parts.append(errors)
                if first_mels is None:
                    first_mels = (real_mel[0].cpu().numpy(), fake_mel[0].cpu().numpy())
    finally:
        g_model.randomized = was_randomized
        if was_training:
            g_model.train()

    scalars, images = {}, {}
    if l1_num:
        scalars["eval/probe_mel_sim"] = max(
            0.0, min(100.0, 100.0 - 100.0 * l1_sum / l1_num)
        )
    if f0_parts:
        errors = torch.cat(f0_parts)
        scalars["eval/probe_f0_cents"] = errors.mean().item()
        scalars["eval/probe_f0_median"] = errors.median().item()
        scalars["eval/probe_f0_frames"] = int(errors.numel())
    if first_mels is not None:
        mel_axes = dict(
            sample_rate=hps.data.sample_rate,
            hop_length=hps.data.hop_length,
            mel_fmin=hps.data.mel_fmin,
            mel_fmax=hps.data.mel_fmax,
        )
        images["mel/probe/real"] = plot_spectrogram_to_numpy(first_mels[0], **mel_axes)
        images["mel/probe/fake"] = plot_spectrogram_to_numpy(first_mels[1], **mel_axes)
    return scalars, images


def train_and_evaluate(
    hps,
    rank,
    epoch,
    nets,
    optims,
    train_loader,
    writer_eval,
    fn_mel_loss,
    device,
    device_id,
    acc,
    eval_batches,
):
    global global_step

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()

    last_batch = None
    disc_names = _DISC_NAMES

    progress = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"{hps.model_name} ▸ Эпоха {epoch}/{hps.total_epoch}",
        unit="батч",
        dynamic_ncols=True,
        leave=True,
        disable=rank != 0,
        file=sys.stdout,
    )
    epoch_start = ttime()
    for _, info in enumerate(progress):
        if device.type == "cuda":
            info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
        else:
            info = [tensor.to(device) for tensor in info]

        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = info
        model_output = net_g(
            phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
        )
        y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = model_output
        wave = slice_segments(
            wave, ids_slice * hps.data.hop_length, hps.train.segment_size, dim=3
        )

        # Discriminator loss
        y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
        loss_disc, r_losses, g_losses = discriminator_loss(
            y_d_hat_r, y_d_hat_g, d_label_smoothing=hps.d_label_smoothing
        )
        loss_disc_r = sum(r_losses)
        loss_disc_g = sum(g_losses)
        optim_d.zero_grad()
        loss_disc.backward()

        # Gradient clipping. clip_grad_norm_ возвращает норму ДО обрезки:
        # берём её, чтобы видеть реальный масштаб градиентов D,
        # а не срезанный потолком.
        grad_norm_d_pre = None
        if hps.d_grad_max_norm > 0:
            grad_norm_d_pre = torch.nn.utils.clip_grad_norm_(
                net_d.parameters(), max_norm=hps.d_grad_max_norm
            )

        grad_norm_d = grad_norm(net_d.parameters())
        optim_d.step()

        # D diagnostics: уверенность D на реальных и фейковых сэмплах
        d_score_r = torch.stack([dr.float().mean() for dr in y_d_hat_r]).mean()
        d_score_g = torch.stack([dg.float().mean() for dg in y_d_hat_g]).mean()

        # Generator loss
        _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
        loss_mel_raw, mel_per_scale = fn_mel_loss(wave, y_hat)
        loss_mel = loss_mel_raw * hps.train.c_mel / 3.0
        mel_per_scale_scaled = [l * hps.train.c_mel / 3.0 for l in mel_per_scale]
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
        loss_fm, fm_per_disc = feature_loss(fmap_r, fmap_g)
        loss_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        loss_gen_all.backward()
        grad_norm_g = grad_norm(net_g.parameters())
        # Per-component gradient norms
        g_model = net_g.module if hasattr(net_g, "module") else net_g
        grad_norm_enc_p = grad_norm(g_model.enc_p.parameters())
        grad_norm_dec = grad_norm(g_model.dec.parameters())
        grad_norm_enc_q = grad_norm(g_model.enc_q.parameters())
        grad_norm_flow = grad_norm(g_model.flow.parameters())
        grad_norm_emb_g = grad_norm(g_model.emb_g.parameters())
        optim_g.step()

        # Аккумуляция метрик
        acc_update_dict = {
            "loss/avg/d": loss_disc,
            "loss/disc_r": loss_disc_r,
            "loss/disc_g": loss_disc_g,
            "loss/avg/g": loss_gen,
            "loss/g/fm": loss_fm,
            "loss/g/mel": loss_mel,
            "loss/g/total": loss_gen_all,
            "grad/norm_d": grad_norm_d,
            "grad/norm_g": grad_norm_g,
            "grad/norm_enc_p": grad_norm_enc_p,
            "grad/norm_dec": grad_norm_dec,
            "grad/norm_enc_q": grad_norm_enc_q,
            "grad/norm_flow": grad_norm_flow,
            "grad/norm_emb_g": grad_norm_emb_g,
            "eval/d_score_r": d_score_r,
            "eval/d_score_g": d_score_g,
        }
        # Тег пишем только при включённом клиппинге: без него норма до обрезки
        # равна grad/norm_d, и дублировать её в графиках незачем.
        if grad_norm_d_pre is not None:
            acc_update_dict["grad/norm_d_pre_clip"] = grad_norm_d_pre
        # FM группируем по типу входа: S — сырой waveform, P — периодная развёртка.
        fm_groups = {}
        for name, fml in zip(disc_names, fm_per_disc):
            group = name[0]
            fm_groups[group] = fm_groups.get(group, 0.0) + fml
        for group, value in fm_groups.items():
            acc_update_dict[f"loss/g/fm_{group}"] = value
        # Mel группируем по разрешению окна (см. _MEL_SCALE_GROUPS)
        mel_groups = {}
        for (name, group), ms in zip(_MEL_SCALE_GROUPS.items(), mel_per_scale_scaled):
            mel_groups[group] = mel_groups.get(group, 0.0) + ms
        for group, value in mel_groups.items():
            acc_update_dict[f"loss/g/mel_{group}"] = value
        acc.update(**acc_update_dict)
        # KL нормируется на число валидных кадров, а не на батч
        acc.update_weighted("loss/g/kl", loss_kl, z_mask.sum())

        if rank == 0:
            progress.set_postfix(
                **{
                    "loss/g/total": f"{loss_gen_all.item():.3f}",
                    "loss/g/mel": f"{loss_mel.item():.3f}",
                },
                refresh=False,
            )

        # Сохраняем данные последнего батча для визуализации и F0-метрики
        if rank == 0:
            last_batch = (spec, ids_slice, y_hat, pitchf)

        global_step += 1

    epoch_seconds = ttime() - epoch_start

    # Метрики должны описывать все данные, а не только шард rank 0.
    # Коллективная операция: вызывать на всех ранках, не под if rank == 0.
    if epoch % hps.train.log_interval == 0:
        acc.reduce_across_ranks(device_id)

    if rank == 0 and epoch % hps.train.log_interval == 0:
        avg = acc.average()
        scalar_dict = {
            **avg,
            "learning rate/G": optim_g.param_groups[0]["lr"],
            "learning rate/D": optim_d.param_groups[0]["lr"],
            "time/epoch_sec": epoch_seconds,
        }
        image_dict = {}
        if eval_batches is not None:
            probe_start = ttime()
            probe_scalars, probe_images = evaluate_probe(
                hps, net_g, eval_batches, device, device_id
            )
            scalar_dict["time/probe_sec"] = ttime() - probe_start
            scalar_dict.update(probe_scalars)
            image_dict.update(probe_images)
        elif last_batch is not None:
            # Режим «Случайный»: набора нет, замеряем последний батч эпохи.
            # Метрики шумные (в батч попадают мьюты), но иначе в этом режиме
            # не будет вообще никакой оценки качества.
            with torch.no_grad():
                spec, ids_slice, y_hat, pitchf = last_batch
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sample_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                segment_frames = hps.train.segment_size // hps.data.hop_length
                y_mel = slice_segments(mel, ids_slice, segment_frames, dim=3)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sample_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                scalar_dict["eval/mel_sim"] = mel_spectrogram_similarity(
                    y_hat_mel, y_mel
                )
                f0_mean, f0_median, f0_frames = f0_error_cents(
                    y_hat,
                    pitchf,
                    ids_slice,
                    hps.data.sample_rate,
                    hps.data.hop_length,
                    segment_frames,
                )
            # Нет voiced-кадров — метрика не посчиталась: пишем пропуск, а не 2400
            if f0_mean is not None:
                scalar_dict["eval/f0_error_cents"] = f0_mean
                scalar_dict["eval/f0_error_median"] = f0_median
                scalar_dict["eval/f0_frames"] = f0_frames
            mel_axes = dict(
                sample_rate=hps.data.sample_rate,
                hop_length=hps.data.hop_length,
                mel_fmin=hps.data.mel_fmin,
                mel_fmax=hps.data.mel_fmax,
            )
            image_dict["mel/slice/real"] = plot_spectrogram_to_numpy(
                y_mel[0].data.cpu().numpy(), **mel_axes
            )
            image_dict["mel/slice/fake"] = plot_spectrogram_to_numpy(
                y_hat_mel[0].data.cpu().numpy(), **mel_axes
            )
        for k, v in scalar_dict.items():
            writer_eval.add_scalar(k, v, epoch)
        for k, v in image_dict.items():
            writer_eval.add_image(k, v, epoch, dataformats="HWC")
        # Принудительный сброс на диск: иначе при остановке процесса
        # теряется хвост до flush_secs (120 с) незаписанных событий.
        writer_eval.flush()

    # Обнуляем накопленное за интервал на всех ранках, не только на rank 0
    if epoch % hps.train.log_interval == 0:
        acc.reset()

    # Сохранение моделей
    if rank == 0:
        is_final_epoch = epoch >= hps.total_epoch
        should_save_checkpoint = (epoch % hps.save_every_epoch == 0) or is_final_epoch

        if should_save_checkpoint:
            # Сохранение чекпоинта
            checkpoint_path = os.path.join(hps.model_dir, "checkpoint.pth")
            save_checkpoint(net_g, optim_g, net_d, optim_d, epoch, checkpoint_path)

            # Сохранение промежуточной модели
            weights_dir = os.path.join(hps.model_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)

            checkpoint_state = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            intermediate_path = os.path.join(
                weights_dir, f"{hps.model_name}_e{epoch}_s{global_step}.pth"
            )
            print(
                extract_model(
                    hps,
                    checkpoint_state,
                    epoch,
                    global_step,
                    intermediate_path,
                    hps.save_half,
                ),
                flush=True,
            )

            # Архивирование
            if is_final_epoch:
                if hps.save_to_zip:
                    import zipfile

                    zip_filename = os.path.join(hps.model_dir, f"{hps.model_name}.zip")
                    with zipfile.ZipFile(
                        zip_filename, "w", zipfile.ZIP_DEFLATED
                    ) as zipf:
                        # Добавляем модель последней эпохи из weights
                        if os.path.exists(intermediate_path):
                            zipf.write(
                                intermediate_path, os.path.basename(intermediate_path)
                            )
                        # Добавляем index
                        index_path = os.path.join(
                            hps.model_dir, f"{hps.model_name}.index"
                        )
                        if os.path.exists(index_path):
                            zipf.write(index_path, os.path.basename(index_path))
                        # Добавляем speakers.txt, если существует
                        speakers_path = os.path.join(hps.model_dir, "speakers.txt")
                        if os.path.exists(speakers_path):
                            zipf.write(speakers_path, "speakers.txt")
                    print(f"Файлы модели заархивированы в '{zip_filename}'", flush=True)

                print("\nОбучение успешно завершено!", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
