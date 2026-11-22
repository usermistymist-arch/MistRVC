"""Вкладка «Обучение»: секции как в официальном RVC WebUI.

Нарезка датасета → Признаки + индекс → Обучение.
Секции запускаются отдельными процессами с общим живым журналом.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
from urllib.parse import urlsplit

import gradio as gr
import requests

from assets.env_paths import require_training_logs_dir, training_logs_dir
from assets.model_installer import (
    PRETRAINS_DIR,
    download_with_progress,
    ensure_pretrains,
)
from assets.pretrains import NO_PRETRAIN, default_pretrain, pretrain_choices
from gradio_ui.jobs import command, request_stop, run_job
from rvc.training.utils.probe import MODE_NAMES, locked_probe

SAMPLE_RATES = [32000, 40000, 48000]
DEFAULT_GPUS = "0"
DEFAULT_SEED = 1032003
DEFAULT_PROBE = "fixed"
PROBE_CHOICES = [
    ("Случайный", "off"),
    ("Обучающие данные", "fixed"),
    ("Исключённые данные", "holdout"),
]
PROBE_INFO = (
    "Сегменты датасета, по которым каждую эпоху замеряется качество. "
    "Случайный: использует случайные сегменты каждую проверку, включая mute-файлы. "
    "Нестабильные графики, датасет не уменьшается. "
    "Обучающие данные: постоянный набор из данных для обучения, без mute-файлов. "
    "Стабильные графики, датасет не уменьшается. "
    "Исключённые данные: постоянный набор, исключённый из данных для обучения, "
    "без mute-файлов. Стабильные графики, датасет уменьшается, но видно: "
    "модель учится или запоминает датасет. Для датасетов от 40 сегментов, когда "
    "часть данных не жалко отдать. Сегменты распределяются по спикерам "
    "пропорционально их объёму данных."
)
PROBE_SIZE_INFO = (
    "Сколько сегментов датасета брать для проверок. "
    "0 — авто: ~8% датасета, но не меньше 4 и не больше 24. "
    "Больше сегментов — точнее замер, но в режиме "
    "«Исключённые данные» столько же +2 соседних на каждый блок спикера "
    "уходят из обучения."
)
F0_METHODS = ["rmvpe", "hpa-rmvpe"]
AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".webm",
    ".mp4",
)


def _section_hint(code1: str):
    """Центрированная подсказка под заголовком секции обучения."""
    return gr.HTML(
        "<div style='font-size:13px;opacity:0.85;text-align:center;padding:0 4px 8px;'>"
        f"Если в папке модели есть <code>{code1}</code>, обучение продолжится с него.</div>",
    )


def _exp_dir(model_name: str) -> str:
    return os.path.join(training_logs_dir(), model_name.strip())


def _require_logs_dir() -> str:
    """Корень экспериментов для действий; в Colab без Drive — понятная ошибка."""
    try:
        return require_training_logs_dir()
    except RuntimeError as error:
        raise gr.Error(str(error)) from error


def _require_name(model_name: str) -> str:
    if not model_name or not model_name.strip():
        raise gr.Error("Укажите имя модели.")
    return model_name.strip()


def _has_files(path: str) -> bool:
    try:
        return os.path.isdir(path) and bool(os.listdir(path))
    except OSError:
        return False


def _step_states(model_name):
    """Доступность кнопок 1/2/3 по фактическим артефактам модели.

    Пройденные шаги блокируются от повторного запуска; индекс доступен
    после извлечения признаков, пока нет файла `logs/<имя>/<имя>.index`.
    """
    if not model_name or not model_name.strip():
        return (
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    name = model_name.strip()
    exp_dir = _exp_dir(name)
    slices_done = _has_files(os.path.join(exp_dir, "data", "sliced_audios"))
    features_done = _has_files(os.path.join(exp_dir, "data", "features"))
    index_done = os.path.isfile(os.path.join(exp_dir, f"{name}.index"))
    return (
        gr.update(interactive=not slices_done),
        gr.update(interactive=slices_done and not features_done),
        gr.update(interactive=features_done and not index_done),
    )


def _refresh_pretrains(sample_rate, current):
    """Список претрейнов под выбранную частоту; чужой выбор сбрасывается."""
    choices = pretrain_choices(sample_rate)
    default = default_pretrain(sample_rate)
    value = current if current in choices else default
    return gr.update(choices=choices, value=value)


def _input_root(model_name: str, files, folder: str) -> str:
    """Папка входа для нарезки: указанная папка или загруженные файлы."""
    if folder and folder.strip():
        if not os.path.isdir(folder.strip()):
            raise gr.Error(f"Папка не найдена: {folder.strip()}")
        return folder.strip()
    if files:
        dataset_dir = tempfile.mkdtemp(prefix="mistrvc_dataset_")
        copied = 0
        for item in files:
            path = getattr(item, "name", None) or (
                item if isinstance(item, str) else None
            )
            if (
                path
                and os.path.isfile(path)
                and path.lower().endswith(AUDIO_EXTENSIONS)
            ):
                shutil.copyfile(path, os.path.join(dataset_dir, os.path.basename(path)))
                copied += 1
        if not copied:
            raise gr.Error("Среди прикреплённых файлов нет аудио.")
        return dataset_dir
    raise gr.Error("Укажите папку с датасетом или прикрепите аудиофайлы.")


def _slice_dataset(model_name, files, folder, sample_rate, segment_len, normalize):
    model_name = _require_name(model_name)
    _require_logs_dir()
    input_root = _input_root(model_name, files, folder)
    yield from run_job(
        f"Нарезка датасета «{model_name}»",
        command(
            "rvc.training.preprocess.preprocess",
            _exp_dir(model_name),
            input_root,
            float(segment_len),
            int(sample_rate),
            "True" if normalize else "False",
        ),
    )


def _extract_features(model_name, sample_rate, f0_method, include_mutes):
    model_name = _require_name(model_name)
    _require_logs_dir()
    sliced = os.path.join(_exp_dir(model_name), "data", "sliced_audios")
    if not os.path.isdir(sliced) or not os.listdir(sliced):
        raise gr.Error("Нет нарезанных сегментов — сначала выполните нарезку.")
    yield from run_job(
        f"Извлечение признаков «{model_name}»",
        command(
            "rvc.training.preprocess.preparing_data",
            _exp_dir(model_name),
            f0_method,
            int(sample_rate),
            int(include_mutes),
        ),
    )


def _train_index(model_name):
    model_name = _require_name(model_name)
    _require_logs_dir()
    features = os.path.join(_exp_dir(model_name), "data", "features")
    if not os.path.isdir(features) or not os.listdir(features):
        raise gr.Error("Нет признаков — сначала извлеките их.")
    yield from run_job(
        f"Индекс «{model_name}»",
        command("rvc.training.preprocess.extract_index", _exp_dir(model_name), "Auto"),
    )


def _check_config(model_name: str, exp_dir: str, sample_rate: int) -> None:
    """Сверяет частоту с уже созданной конфигурацией модели.

    `data/config.json` пишется один раз при первом запуске обучения, поэтому
    смена частоты у существующей модели иначе прошла бы молча:
    `rvc.training.train` берёт архитектуру и частоту именно из этого файла.
    Вокодер сверяется только как защита: поддерживается лишь HiFi-GAN.
    """
    config_path = os.path.join(exp_dir, "data", "config.json")
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise gr.Error(f"Не удалось прочитать {config_path}: {error}") from error

    saved_vocoder = config.get("model", {}).get("vocoder", "HiFi-GAN")
    if saved_vocoder != "HiFi-GAN":
        raise gr.Error(
            f"Модель «{model_name}» обучена с вокодером «{saved_vocoder}», "
            "который больше не поддерживается. MistRVC работает только с HiFi-GAN."
        )
    saved_rate = int(config.get("data", {}).get("sample_rate", sample_rate))
    if saved_rate != sample_rate:
        raise gr.Error(
            f"Модель «{model_name}» уже настроена на {saved_rate} Hz. "
            "Смена частоты возможна только для новой модели: удалите "
            f"{config_path} или возьмите другое имя."
        )


def _locked_seed(exp_dir: str):
    """Сид существующей модели или None, если сид ещё можно выбрать.

    Источник — `data/config.json`: он создаётся при первом запуске
    обучения и с этого момента хранит сид модели (`train.seed`).
    `checkpoint.pth` — страховка на случай утраченного конфига.
    """
    config_path = os.path.join(exp_dir, "data", "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as file:
                train_config = json.load(file).get("train", {})
            return int(train_config.get("seed", DEFAULT_SEED))
        except (OSError, ValueError, TypeError):
            return None
    if os.path.isfile(os.path.join(exp_dir, "checkpoint.pth")):
        return DEFAULT_SEED
    return None


def _seed_state(model_name):
    """Поле сида по имени модели: подставляет сид существующей и блокирует его.

    Вызывается тем же событием ввода имени, что и доступность кнопок 1/2/3
    (`_step_states`). Значение свободного поля не сбрасывается: ввод имени
    не должен затирать сид, выбранный пользователем для новой модели.
    """
    name = (model_name or "").strip()
    exp_dir = _exp_dir(name) if name else ""
    locked = _locked_seed(exp_dir) if name else None
    if locked is not None:
        return gr.update(
            value=locked,
            interactive=False,
            info=f"Сид модели «{name}» зафиксирован в data/config.json — менять его нельзя.",
        )
    if name and os.path.isdir(exp_dir):
        return gr.update(
            interactive=True,
            info="Модель существует, но обучение ещё не запускалось: сид можно выбрать "
            "до первого запуска, после он будет зафиксирован.",
        )
    return gr.update(
        interactive=True,
        info="Зерно случайности: инициализация emb_g, случайные срезы и шум, порядок батчей. "
        "Фиксируется при первом запуске обучения.",
    )


def _seed_on_start(model_name, seed):
    """Блокирует поле сида в момент нажатия «Запустить обучение».

    Обучение фиксирует сид в конфиге первым же действием, но основной
    обработчик завершится только через часы, поэтому поле блокируется
    заранее — и сразу с зафиксированным значением, если оно есть на диске.
    """
    name = (model_name or "").strip()
    locked = _locked_seed(_exp_dir(name)) if name else None
    value = locked if locked is not None else seed
    return gr.update(
        value=value if value is not None else DEFAULT_SEED,
        interactive=False,
        info="Обучение запущено: сид зафиксирован и не может быть изменён.",
    )


def _probe_state(model_name):
    """Поля набора по имени модели: подставляет зафиксированные и блокирует."""
    name = (model_name or "").strip()
    locked = locked_probe(_exp_dir(name)) if name else None
    if locked is None:
        return (
            gr.update(interactive=True, info=PROBE_INFO),
            gr.update(interactive=True, info=PROBE_SIZE_INFO),
        )
    mode, size = locked
    note = f"Набор модели «{name}» зафиксирован в data/meta.json — менять его нельзя."
    return (
        gr.update(value=mode, interactive=False, info=note),
        gr.update(value=size, interactive=False, info=note),
    )


def _probe_on_start(model_name, probe, probe_size):
    """Блокирует поля набора в момент нажатия кнопки, до конца обучения."""
    name = (model_name or "").strip()
    locked = locked_probe(_exp_dir(name)) if name else None
    mode, size = locked if locked is not None else (probe, probe_size)
    note = "Обучение запущено: проверочный набор зафиксирован."
    return (
        gr.update(value=mode, interactive=False, info=note),
        gr.update(value=size if size is not None else 0, interactive=False, info=note),
    )


def _lock_on_start(model_name, seed, probe, probe_size):
    """Один предшаг кнопки: блокирует сид и проверочный набор разом."""
    return (
        _seed_on_start(model_name, seed),
        *_probe_on_start(model_name, probe, probe_size),
    )


def _url_pretrain_path(kind: str, url: str) -> str:
    """Локальное имя для претрейна из URL: имя файла + хеш ссылки.

    Хеш в имени отличает файлы с одинаковыми именами из разных ссылок,
    поэтому скачанный раз претрейн переиспользуется без повторной загрузки.
    """
    name = os.path.basename(urlsplit(url).path)
    if not name.lower().endswith(".pth"):
        name = f"pretrain_{kind}.pth"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    safe_stem = re.sub(r"[^\w\-.]", "_", name[: -len(".pth")])
    return os.path.join(PRETRAINS_DIR, f"custom_{safe_stem}_{digest}_{kind}.pth")


def _prepare_manual_pretrain(kind: str, value: str):
    """Свой претрейн: локальный путь как есть, URL — скачивается.

    Генератор: отдаёт строки журнала, возвращает локальный путь к файлу.
    """
    if value.startswith(("http://", "https://")):
        local_path = _url_pretrain_path(kind, value)
        if os.path.isfile(local_path):
            yield f"{os.path.basename(local_path)} уже скачан"
            return local_path
        yield f"Скачиваю претрейн {kind}…"
        try:
            yield from download_with_progress(value, local_path)
        except (requests.RequestException, ValueError) as error:
            raise gr.Error(f"Не удалось скачать претрейн {kind}: {error}") from error
        return local_path
    if not os.path.isfile(value):
        raise gr.Error(
            f"Претрейн не найден: {value} — укажите существующий файл или прямую ссылку на него."
        )
    return value


def _resolve_pretrains(pretrain, pretrain_g, pretrain_d, sample_rate, exp_dir):
    """Пути претрейнов G/D: свои файлы или ссылки, встроенный набор или ничего.

    Генератор: отдаёт строки журнала, возвращает (путь G, путь D).
    Свои файлы имеют приоритет над встроенным набором. При продолжении
    обучения с чекпоинта претрейны не нужны — скачивание пропускается.
    """
    manual_g = (pretrain_g or "").strip()
    manual_d = (pretrain_d or "").strip()
    if manual_g or manual_d:
        if not (manual_g and manual_d):
            raise gr.Error("Укажите оба своих претрейна (G и D) или ни одного.")
        resolved_g = yield from _prepare_manual_pretrain("G", manual_g)
        resolved_d = yield from _prepare_manual_pretrain("D", manual_d)
        yield f"• Свои претрейны: {os.path.basename(resolved_g)}, {os.path.basename(resolved_d)}"
        return resolved_g, resolved_d
    if os.path.isfile(os.path.join(exp_dir, "checkpoint.pth")):
        yield "• Найден checkpoint.pth — обучение продолжится, претрейны не нужны."
        return None, None
    if pretrain == NO_PRETRAIN:
        yield "• Без претрейна: потребуется больше эпох и данных."
        return None, None
    try:
        return (yield from ensure_pretrains(pretrain, sample_rate))
    except ValueError as error:
        raise gr.Error(str(error)) from error


def _drain(generator):
    """Прогоняет генератор до конца, показывая накопленный журнал.

    Возвращает (строки, return-значение генератора).
    """
    lines = []
    try:
        while True:
            lines.append(next(generator))
            yield "\n".join(lines)
    except StopIteration as done:
        return lines, done.value


def _prefixed(note: str, generator):
    """Генератор с предупреждающей строкой в начале журнала.

    `_drain` накапливает её вместе с последующими строками, поэтому
    предупреждение видно сразу и не теряется на следующих этапах задачи.
    """
    if note:
        yield note
    return (yield from generator)


def _train_model(
    model_name,
    sample_rate,
    total_epoch,
    save_every_epoch,
    batch_size,
    pretrain,
    pretrain_g,
    pretrain_d,
    save_to_zip,
    save_half,
    d_lr_multiplier,
    d_grad_max_norm,
    d_label_smoothing,
    seed,
    deterministic,
    probe,
    probe_size,
):
    model_name = _require_name(model_name)
    logs_dir = _require_logs_dir()
    exp_dir = _exp_dir(model_name)
    if not os.path.isfile(os.path.join(exp_dir, "data", "filelist.txt")):
        raise gr.Error(
            "Нет filelist.txt — сначала выполните нарезку и извлечение признаков."
        )
    _check_config(model_name, exp_dir, int(sample_rate))
    locked_seed = _locked_seed(exp_dir)
    seed_note = ""
    if locked_seed is not None and seed is not None and int(seed) != locked_seed:
        # Сид посреди тренировки менять нельзя: запуск не прерываем, а
        # принудительно продолжаем с зафиксированным сидом и предупреждаем.
        seed_note = (
            f"Сид модели «{model_name}» зафиксирован: {locked_seed}. "
            f"Передан {int(seed)} — менять сид посреди тренировки нельзя. "
            f"Обучение принудительно продолжится с сидом {locked_seed}. "
            "Для другого сида создайте новую модель."
        )
    locked_set = locked_probe(exp_dir)
    probe_note = ""
    if locked_set is not None:
        saved_mode, saved_size = locked_set
        if probe != saved_mode or int(probe_size or 0) not in (0, saved_size):
            probe_note = (
                f"Проверочный набор модели «{model_name}» зафиксирован: "
                f"«{MODE_NAMES.get(saved_mode, saved_mode)}», {saved_size} сегм. "
                "Менять его посреди тренировки нельзя — "
                "обучение принудительно продолжится с прежним набором."
            )
        probe, probe_size = saved_mode, saved_size
    cmd = command(
        "rvc.training.train",
        "--experiment_dir",
        logs_dir,
        "--model_name",
        model_name,
        "--total_epoch",
        int(total_epoch),
        "--save_every_epoch",
        int(save_every_epoch),
        "--batch_size",
        int(batch_size),
        "--sample_rate",
        int(sample_rate),
        "--gpus",
        DEFAULT_GPUS,
        "--save_to_zip",
        "true" if save_to_zip else "false",
        "--save_half",
        "true" if save_half else "false",
        "--d_lr_multiplier",
        float(d_lr_multiplier),
        "--d_grad_max_norm",
        float(d_grad_max_norm),
        "--d_label_smoothing",
        float(d_label_smoothing),
    )
    if locked_seed is None:
        cmd += ["--seed", str(int(seed) if seed is not None else DEFAULT_SEED)]
    if deterministic:
        cmd.append("--deterministic")
    cmd += ["--probe", str(probe), "--probe_size", str(int(probe_size or 0))]

    notes, (resolved_g, resolved_d) = yield from _drain(
        _prefixed(
            "\n".join(part for part in (seed_note, probe_note) if part),
            _resolve_pretrains(
                pretrain, pretrain_g, pretrain_d, int(sample_rate), exp_dir
            ),
        ),
    )
    if resolved_g and resolved_d:
        cmd += ["--pretrain_g", resolved_g, "--pretrain_d", resolved_d]
    yield from run_job(f"Обучение «{model_name}»", cmd, prefix="\n".join(notes))


def training_tab():
    model_name = gr.Textbox(label="Имя модели", placeholder="MyVoice")

    with (
        gr.Accordion("Подготовка данных", open=True),
        gr.Group(),
        gr.Row(equal_height=True),
    ):
        with gr.Column(scale=1):
            dataset_folder = gr.Textbox(
                label="Папка с датасетом", placeholder="/путь/к/аудио"
            )
            dataset_files = gr.File(
                label="…или загрузите файлы", file_count="multiple", height=260
            )
        with gr.Column(scale=1):
            with gr.Row(equal_height=True):
                sample_rate = gr.Dropdown(
                    SAMPLE_RATES, value=48000, label="Частота (Hz)"
                )
                f0_method = gr.Dropdown(F0_METHODS, value="hpa-rmvpe", label="Метод F0")
                normalize = gr.Checkbox(value=True, label="Нормализация")
            with gr.Row(equal_height=True):
                segment_len = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=3.0,
                    label="Сегмент (сек)",
                )
                include_mutes = gr.Slider(
                    minimum=0, maximum=10, step=1, value=2, label="Мьют-файлов"
                )
            with gr.Row(equal_height=True):
                slice_btn = gr.Button("1. Нарезать на сегменты", variant="primary")
                extract_btn = gr.Button(
                    "2. Извлечь признаки", variant="primary", interactive=False
                )
                index_btn = gr.Button(
                    "3. Построить индекс", variant="primary", interactive=False
                )

    with gr.Group():
        _section_hint("checkpoint.pth")
        with gr.Row(equal_height=True):
            total_epoch = gr.Slider(
                minimum=1, maximum=5000, step=1, value=500, label="Всего эпох"
            )
            save_every_epoch = gr.Slider(
                minimum=1, maximum=200, step=1, value=20, label="Сохранять каждые N"
            )
            batch_size = gr.Slider(
                minimum=1, maximum=128, step=1, value=16, label="Батч"
            )
        with gr.Row(equal_height=True):
            pretrain = gr.Dropdown(
                pretrain_choices(48000),
                value=default_pretrain(48000),
                label="Претрейн",
                scale=1,
            )
            pretrain_g = gr.Textbox(
                label="Свой претрейн G",
                placeholder="Путь или URL к .pth — вместо встроенного",
                scale=2,
            )
            pretrain_d = gr.Textbox(
                label="Свой претрейн D",
                placeholder="Путь или URL к .pth — вместо встроенного",
                scale=2,
            )
        with gr.Row(equal_height=True):
            save_to_zip = gr.Checkbox(False, label="Собрать ZIP в конце")
            save_half = gr.Checkbox(False, label="Половинная точность моделей")
        with gr.Accordion("Стабилизация обучения", open=False):
            gr.Markdown(
                "Настройки для опытных пользователей - изменяйте, только если знаете что делаете."
            )
            with gr.Row():
                d_lr_multiplier = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=1.0,
                    label="Скорость обучения D",
                    info="Множитель к скорости G: 1.0 = как у G, 0.25 = в 4 раза медленнее.",
                )
                d_grad_max_norm = gr.Slider(
                    minimum=0.0,
                    maximum=500.0,
                    step=5.0,
                    value=0.0,
                    label="Потолок градиентов D",
                    info="Потолок суммарной нормы градиентов D. 0.0 = выкл.",
                )
                d_label_smoothing = gr.Slider(
                    minimum=0.0,
                    maximum=0.2,
                    step=0.01,
                    value=0.0,
                    label="Сглаживание меток D",
                    info="Цели D: real = 1-s, fake = s - D менее самоуверен. 0.0 = выкл.",
                )
        with gr.Accordion("Воспроизводимость", open=False):
            with gr.Group():
                with gr.Row(equal_height=True):
                    seed = gr.Number(
                        value=DEFAULT_SEED,
                        precision=0,
                        minimum=0,
                        maximum=2**31 - 1,
                        label="Сид обучения",
                        info="Зерно случайности: инициализация emb_g, случайные срезы и шум, порядок батчей.",
                    )
                    deterministic = gr.Checkbox(
                        value=False,
                        label="Детерминировать обучение",
                        info="Повторяемость прогонов при одинаковом сиде. Обучение будет медленнее.",
                    )
                with gr.Row(equal_height=True):
                    probe = gr.Dropdown(
                        PROBE_CHOICES,
                        value=DEFAULT_PROBE,
                        label="Проверочный набор",
                        info=PROBE_INFO,
                    )
                    probe_size = gr.Number(
                        value=0,
                        precision=0,
                        minimum=0,
                        maximum=24,
                        label="Сегментов в наборе",
                        info=PROBE_SIZE_INFO,
                    )
        with gr.Row(equal_height=True):
            train_btn = gr.Button("Запустить обучение", variant="primary")
            stop_btn = gr.Button("Завершить процесс", variant="stop")
        stop_status = gr.Markdown("")

    with gr.Group():
        log = gr.Textbox(label="Журнал", lines=24, max_lines=24)

    step_buttons = [slice_btn, extract_btn, index_btn]
    model_name.change(
        _step_states, inputs=model_name, outputs=step_buttons, api_name=False
    )
    model_name.change(_seed_state, inputs=model_name, outputs=seed, api_name=False)
    model_name.change(
        _probe_state, inputs=model_name, outputs=[probe, probe_size], api_name=False
    )
    # Список претрейнов зависит от частоты.
    sample_rate.change(
        _refresh_pretrains,
        inputs=[sample_rate, pretrain],
        outputs=pretrain,
        api_name=False,
    )
    slice_btn.click(
        _slice_dataset,
        inputs=[
            model_name,
            dataset_files,
            dataset_folder,
            sample_rate,
            segment_len,
            normalize,
        ],
        outputs=log,
    ).then(_step_states, inputs=model_name, outputs=step_buttons, api_name=False)
    extract_btn.click(
        _extract_features,
        inputs=[model_name, sample_rate, f0_method, include_mutes],
        outputs=log,
    ).then(_step_states, inputs=model_name, outputs=step_buttons, api_name=False)
    index_btn.click(
        _train_index,
        inputs=model_name,
        outputs=log,
    ).then(_step_states, inputs=model_name, outputs=step_buttons, api_name=False)
    train_btn.click(
        _lock_on_start,
        inputs=[model_name, seed, probe, probe_size],
        outputs=[seed, probe, probe_size],
        queue=False,
        api_name=False,
    ).then(
        _train_model,
        inputs=[
            model_name,
            sample_rate,
            total_epoch,
            save_every_epoch,
            batch_size,
            pretrain,
            pretrain_g,
            pretrain_d,
            save_to_zip,
            save_half,
            d_lr_multiplier,
            d_grad_max_norm,
            d_label_smoothing,
            seed,
            deterministic,
            probe,
            probe_size,
        ],
        outputs=log,
        queue=True,
    ).then(_seed_state, inputs=model_name, outputs=seed, api_name=False).then(
        _probe_state,
        inputs=model_name,
        outputs=[probe, probe_size],
        api_name=False,
    )
    stop_btn.click(request_stop, outputs=stop_status, queue=False, api_name=False)
