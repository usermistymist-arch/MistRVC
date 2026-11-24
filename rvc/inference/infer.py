import gc
import os
import threading

import gradio as gr
import numpy as np
import torch
from scipy import signal
from scipy.ndimage import uniform_filter1d

from rvc._library.algorithm.synthesizers import Synthesizer
from rvc._library.audio import load_audio, save_audio
from rvc._library.config import Config
from rvc._library.embedders.fairseq import load_model
from rvc.inference.pipeline import VC

# Определяем пути к папкам и файлам (константы).
# Корень репозитория, а не текущий каталог: обучение запускается отдельными
# процессами с cwd = корень проекта, поэтому папки должны совпадать.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RVC_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "RVC_models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "RVC_output")
EMBEDDER_PATH = os.path.join(
    PROJECT_ROOT, "assets", "models", "embedders", "contentvec_base.pt"
)

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".aiff",
}

# Создаем папки, если их нет
os.makedirs(RVC_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Инициализация конфигурации
config = Config()


def _process_silence(
    audio: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 2.0,
    buffer_sec: float = 1.0,
    threshold_db: float = -60.0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Вырезает середину длительных тишин, оставляя буфер по краям.

    Возвращает:
        processed_audio: аудио с вырезанными серединами тишин.
        silence_regions: список кортежей (insert_pos, silence_len) в координатах processed_audio,
                         указывающих, куда и сколько тишины нужно вставить обратно.
    """
    threshold_linear = 10 ** (threshold_db / 20.0)
    window_size = max(1, int(0.01 * sample_rate))  # 10 мс
    min_silence_samples = int(min_duration_sec * sample_rate)
    buffer_samples = int(buffer_sec * sample_rate)

    if len(audio) < window_size:
        return audio, []

    # Детекция по копии без DC и гула <48 Гц: HPF пайплайна применяется позже,
    # а здесь низкочастотный мусор иначе удерживал бы RMS выше порога.
    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sample_rate)
    analysis = signal.filtfilt(bh, ah, audio).astype(np.float64)
    # uniform_filter1d ведёт скользящую сумму с плавающей точкой: после громких
    # участков в цифровой тишине остаётся крошечный ОТРИЦАТЕЛЬНЫЙ остаток.
    # sqrt(отрицательного) = NaN, а NaN < порог = False — тишина не детектировалась.
    # Поэтому обрезаем снизу нулём ДО sqrt.
    mean_sq = uniform_filter1d(analysis**2, size=window_size)
    rms = np.sqrt(np.maximum(mean_sq, 0.0))
    is_silent = rms < threshold_linear

    # Находим все участки тишины
    silence_regions_raw = []
    in_silence = False
    start_idx = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            start_idx = i
        elif not silent and in_silence:
            in_silence = False
            end_idx = i
            if end_idx - start_idx >= min_silence_samples:
                silence_regions_raw.append((start_idx, end_idx))

    if in_silence and len(is_silent) - start_idx >= min_silence_samples:
        silence_regions_raw.append((start_idx, len(is_silent)))

    if not silence_regions_raw:
        return audio, []

    # Центрированное RMS-окно сжимает регион на пол-окна с каждой стороны —
    # расширяем границы обратно, чтобы якоря таймингов были точными до сэмпла.
    silence_regions_raw = [
        (max(0, s - window_size // 2), min(len(audio), e + window_size // 2))
        for s, e in silence_regions_raw
    ]

    # Обрабатываем участки: оставляем буфер, вырезаем середину
    segments = []
    regions_to_restore = []
    current_pos = 0
    processed_len = 0

    for start, end in silence_regions_raw:
        duration = end - start
        if duration <= 2 * buffer_samples:
            continue  # Пауза слишком короткая, чтобы вырезать середину

        # Аудио до начала тишины
        if start > current_pos:
            segments.append(audio[current_pos:start])
            processed_len += start - current_pos

        # Начальный буфер тишины
        segments.append(audio[start : start + buffer_samples])
        processed_len += buffer_samples

        # Вырезаемая середина
        cut_len = duration - 2 * buffer_samples
        regions_to_restore.append((processed_len, cut_len))

        # Пропускаем вырезанную середину, current_pos сдвигаем на конец буфера
        current_pos = end - buffer_samples

    # Оставшееся аудио
    if current_pos < len(audio):
        segments.append(audio[current_pos:])

    if not segments:
        return audio, []

    return np.concatenate(segments), regions_to_restore


def _restore_silence(
    audio_opt: np.ndarray,
    silence_regions: list[tuple[int, int]],
    processed_len: int,
    tgt_sr: int,
    crossfade_ms: float = 10.0,
) -> np.ndarray:
    """Вставляет вырезанные середины тишин обратно с кроссфейдом.

    Координаты регионов — сэмплы 16 кГц обработанного аудио. Выход модели имеет
    частоту tgt_sr, а его длина может отличаться от идеальной на ±окно на каждой
    внутренней склейке пайплайна, поэтому:
      * точка вставки берётся по фактическому масштабу len(audio_opt)/processed_len;
      * длина тишины подбирается так, чтобы её конец попал ровно на идеальную
        границу оригинала — вокал после каждой паузы начинается в свой тайминг.
    """
    if not silence_regions or processed_len <= 0:
        return audio_opt

    rate = tgt_sr / 16000.0
    model_len = len(audio_opt)
    scale = model_len / processed_len
    cf_samples = max(1, int((crossfade_ms / 1000.0) * tgt_sr))

    result_segments = []
    model_pos = 0  # курсор по audio_opt
    inserted = 0  # сколько нулей уже вставлено
    cut_before = 0  # сумма вырезанных сэмплов (16 кГц) до текущего региона
    after_zeros = False  # предыдущий шаг вставлял тишину -> нужен fade-in

    for insert_pos, cut_len in silence_regions:
        m = max(model_pos, min(int(round(insert_pos * scale)), model_len))
        ideal_end = int(round((insert_pos + cut_before + cut_len) * rate))
        zeros_len = max(0, ideal_end - m - inserted)

        if m > model_pos:
            segment = audio_opt[model_pos:m].copy()
            fade = min(cf_samples, len(segment))
            segment[-fade:] *= np.linspace(1.0, 0.0, fade)  # fade-out перед тишиной
            if after_zeros:
                fi = min(cf_samples, len(segment))
                segment[:fi] *= np.linspace(0.0, 1.0, fi)  # fade-in после тишины
            result_segments.append(segment)

        if zeros_len > 0:
            result_segments.append(np.zeros(zeros_len, dtype=audio_opt.dtype))
            inserted += zeros_len
        after_zeros = zeros_len > 0

        model_pos = m
        cut_before += cut_len

    if model_pos < model_len:
        segment = audio_opt[model_pos:].copy()
        fade = min(cf_samples, len(segment))
        segment[:fade] *= np.linspace(0.0, 1.0, fade)
        result_segments.append(segment)

    return np.concatenate(result_segments)


# Отображает прогресс выполнения задачи.
def display_progress(percent, message, is_print, progress=gr.Progress()):
    if is_print:
        print(message)
    progress(percent, desc=message)


# Загружает модель RVC и индекс по имени модели.
def load_rvc_model(rvc_model):
    model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Папка модели {rvc_model} не найдена в {RVC_MODELS_DIR}"
        )

    # Находим файл модели с расширением .pth
    rvc_model_path = next(
        (
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".pth")
        ),
        None,
    )
    # Находим файл индекса с расширением .index
    rvc_index_path = next(
        (
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if f.endswith(".index")
        ),
        None,
    )

    # Проверяем, существует ли файл модели
    if not rvc_model_path:
        raise FileNotFoundError(f"Модель {rvc_model} не содержит .pth файла!")

    return rvc_model_path, rvc_index_path


# Загружает семантический эмбеддер
def load_embedder(model_path):
    embedder = load_model(model_path).to(config.device).eval()
    return embedder


# Получает конвертер голоса
def get_vc(model_path):
    # Загружаем состояние модели
    cpt = torch.load(model_path, map_location="cpu", weights_only=True)
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(
            f"Некорректный формат модели {model_path}. Используйте модель RVC."
        )

    # MistRVC поддерживает только v2-модели с питчем
    model_name = os.path.basename(model_path)
    if cpt.get("version") != "v2":
        raise ValueError(
            f"Модель '{model_name}' не v2 (version={cpt.get('version')!r}). MistRVC поддерживает только v2-модели.",
        )
    if not cpt.get("f0", 1):
        raise ValueError(
            f"Модель '{model_name}' обучена без питча (f0=False). MistRVC поддерживает только v2-модели с питчем.",
        )

    # Извлекаем параметры модели
    tgt_sr = cpt["config"][-1]
    vocoder = cpt.get("vocoder", "HiFi-GAN")
    if vocoder != "HiFi-GAN":
        raise ValueError(
            f"Модель '{model_name}' обучена с вокодером '{vocoder}'. "
            "MistRVC поддерживает только HiFi-GAN."
        )

    # Инициализируем синтезатор (единственная архитектура — HiFi-GAN NSF)
    net_g = Synthesizer(*cpt["config"], text_enc_hidden_dim=768)

    # Удаляем ненужный слой
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float().eval()

    # Инициализируем объект конвертера голоса
    vc = VC(tgt_sr, config)
    return cpt, net_g, tgt_sr, vc


# Gradio 5.x пропускает в имени файла только буквы/цифры и "._-, ",
# поэтому обычный "+" при скачивании вырезается. Для положительного питча
# берём символ-букву, который фильтр не трогает:
#   "十" (U+5341) — полноразмерный плюс, "ᐩ" (U+1429) — компактный приподнятый.
PLUS_SIGN = "十"


# Собирает путь выходного файла по шаблону из настроек
def build_output_path(
    input_path,
    output_dir,
    rvc_model,
    output_format,
    speaker_name=None,
    filename_template="{PITCH_SHIFT}{ }{NAME} - {MODEL}{_}{SPEAKER}",
    rvc_pitch=0,
):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if len(base_name) > 120:
        gr.Warning("Имя файла превышает 120 символов и будет сокращено для удобства.")
        base_name = f"{base_name[:61]}..."
    os.makedirs(output_dir, exist_ok=True)

    speaker_str = speaker_name if speaker_name else ""
    pitch_shift_str = ""
    if rvc_pitch:
        sign = PLUS_SIGN if rvc_pitch > 0 else "-"
        pitch_shift_str = f"{sign}{abs(rvc_pitch):g}"
    filename = (
        filename_template.replace("{NAME}", base_name)
        .replace("{MODEL}", rvc_model)
        .replace("{SPEAKER}", speaker_str)
        .replace("{PITCH_SHIFT}", pitch_shift_str)
    )

    # Опциональные разделители: { } -> пробел, {_} -> "_".
    # Если после подстановки ключей они оказались по краям имени — убираются:
    # пустой {PITCH_SHIFT} не оставляет ведущий пробел, а пустой {SPEAKER} — висящее "_".
    filename = filename.replace("{ }", " ").replace("{_}", "_").strip(" _")
    filename = filename.replace("+", PLUS_SIGN)
    if not filename:
        filename = base_name

    return os.path.join(output_dir, f"{filename}.{output_format}")


# Загружает эмбеддер, модель и конвертер голоса одним набором
def load_pipeline(rvc_model):
    display_progress(0.1, "Загружаем семантический эмбеддер...", False)
    embedder_model = load_embedder(EMBEDDER_PATH)
    display_progress(0.2, f"Загружаем модель '{rvc_model}'...", False)
    model_path, index_path = load_rvc_model(rvc_model)
    display_progress(0.3, "Получаем конвертер голоса...", False)
    cpt, net_g, tgt_sr, vc = get_vc(model_path)

    num_speakers = cpt["weight"]["emb_g.weight"].shape[0]

    return {
        "embedder": embedder_model,
        "cpt": cpt,
        "net_g": net_g,
        "tgt_sr": tgt_sr,
        "vc": vc,
        "index_path": index_path,
        "num_speakers": num_speakers,
    }


# Кэш последнего пайплайна: повторные запуски той же модели не перезагружают
# эмбеддер и веса. Хранится один пайплайн — при смене модели предыдущий выгружается.
_pipeline_cache = {"name": None, "pipe": None, "stamp": None}
_pipeline_lock = threading.Lock()


def _file_stamp(path):
    """Отпечаток файла модели: перезапись .pth должна сбрасывать кэш."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _free_memory():
    """Отпускает ссылки сборщику мусора и чистит кэш CUDA."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Выгружает пайплайн из памяти (когда он больше никому не нужен)
def free_pipeline(pipe):
    if pipe is not None:
        pipe.clear()
    _free_memory()


def _forget_pipeline(pipe):
    """Забывает пайплайн, не трогая его словарь: им может пользоваться другой запрос."""
    if pipe is not None:
        del pipe
    _free_memory()


def get_pipeline(rvc_model):
    """Пайплайн модели: готовый из памяти, если она не менялась, иначе с диска."""
    # Листинг папки вместо полной загрузки: файлы модели могли заменить на диске.
    model_path, index_path = load_rvc_model(rvc_model)
    stamp = _file_stamp(model_path)
    with _pipeline_lock:
        cached = _pipeline_cache
        if (
            cached["name"] == rvc_model
            and cached["pipe"] is not None
            and cached["stamp"] == stamp
        ):
            # Индекс читается из файла на каждом запуске (pipeline.py) — обновляем путь,
            # чтобы появившийся позже .index подхватился без перезагрузки модели.
            cached["pipe"]["index_path"] = index_path
            display_progress(0.3, f"[⚡] Модель '{rvc_model}' уже загружена", True)
            return cached["pipe"]
        previous = cached["pipe"]
        pipe = load_pipeline(rvc_model)
        cached["name"] = rvc_model
        cached["pipe"] = pipe
        cached["stamp"] = stamp
    _forget_pipeline(previous)
    return pipe


def release_pipeline():
    """Выгружает кэш инференса; нужна перед обучением, UVR и апскейлом.

    Возвращает имя выгруженной модели или None, если память уже свободна.
    """
    with _pipeline_lock:
        name = _pipeline_cache["name"]
        pipe = _pipeline_cache["pipe"]
        _pipeline_cache["name"] = None
        _pipeline_cache["pipe"] = None
        _pipeline_cache["stamp"] = None
    if pipe is None:
        return None
    _forget_pipeline(pipe)
    return name


# Конвертирует один файл уже загруженным пайплайном
def convert_one(
    pipe,
    input_path,
    output_path,
    f0_method="rmvpe",
    f0_min=61,
    f0_max=2218,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0.25,
    volume_envelope=1.0,
    output_format="wav",
    speaker_id=0,
):
    display_progress(0.4, "Загружаем аудио...", False)
    audio = load_audio(input_path, 16000)

    # --- Обработка тишины ---
    processed_audio, silence_regions = _process_silence(
        audio,
        sample_rate=16000,
        min_duration_sec=2.0,
        buffer_sec=1.0,
        threshold_db=-60.0,
    )
    if silence_regions:
        display_progress(
            0.45, f"[✂️] Вырезано {len(silence_regions)} участков тишины...", True
        )
        audio_to_convert = processed_audio
    else:
        audio_to_convert = audio

    display_progress(
        0.5, f"[🌌] Преобразуем аудио '{os.path.basename(input_path)}'...", True
    )
    audio_opt = pipe["vc"].pipeline(
        model=pipe["embedder"],
        net_g=pipe["net_g"],
        sid=speaker_id,
        audio=audio_to_convert,
        pitch=rvc_pitch,
        f0_min=f0_min,
        f0_max=f0_max,
        f0_method=f0_method,
        file_index=pipe["index_path"],
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
    )

    # --- Возврат тишины ---
    if silence_regions:
        display_progress(0.75, "[↩️] Возвращаем тишину...", True)
        audio_opt = _restore_silence(
            audio_opt,
            silence_regions,
            len(processed_audio),
            pipe["tgt_sr"],
            crossfade_ms=10.0,
        )

    display_progress(0.8, "[💫] Сохраняем результат...", True)
    save_audio(audio_opt, pipe["tgt_sr"], output_path, output_format)
    return output_path


# Выполнение инференса с использованием RVC
def rvc_infer(
    rvc_model=None,
    input_path=None,
    f0_method="rmvpe",
    f0_min=61,
    f0_max=2218,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0.25,
    volume_envelope=1.0,
    output_format="wav",
    speaker_id=0,
    speaker_name="",
    filename_template="{PITCH_SHIFT}{ }{NAME} - {MODEL}{_}{SPEAKER}",
    progress=gr.Progress(track_tqdm=True),
):
    if not rvc_model:
        raise ValueError("Не выбрана модель для RVC-инференса")
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл '{input_path}' не найден!")

    display_progress(0, "\n[⚙️] Запуск конвейера генерации...", True)
    # Пайплайн остаётся в памяти: следующий запуск этой же модели стартует сразу.
    pipe = get_pipeline(rvc_model)
    output_path = build_output_path(
        input_path,
        OUTPUT_DIR,
        rvc_model,
        output_format,
        speaker_name,
        filename_template,
        rvc_pitch,
    )
    convert_one(
        pipe,
        input_path,
        output_path,
        f0_method=f0_method,
        f0_min=f0_min,
        f0_max=f0_max,
        rvc_pitch=rvc_pitch,
        protect=protect,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        output_format=output_format,
        speaker_id=speaker_id,
    )

    display_progress(1.0, f"[✅] Преобразование завершено — {output_path}", True)
    return gr.Audio(output_path, label=os.path.basename(output_path))


# Собирает список аудиофайлов: папка и/или загруженные файлы
def collect_batch_inputs(dir_input, files):
    inputs = []
    if dir_input and str(dir_input).strip():
        folder = str(dir_input).strip()
        if not os.path.isdir(folder):
            raise ValueError(f"Папка не найдена: {folder}")
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if (
                os.path.isfile(path)
                and os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS
            ):
                inputs.append(path)
    for item in files or []:
        path = getattr(item, "name", None) or (item if isinstance(item, str) else None)
        if path and os.path.isfile(path):
            inputs.append(path)
    # Убираем дубликаты, сохраняя порядок
    return list(dict.fromkeys(inputs))


# Пакетная конвертация: модель грузится один раз, файлы идут по очереди
def rvc_batch_infer(
    rvc_model=None,
    dir_input=None,
    files=None,
    output_dir=None,
    f0_method="rmvpe",
    f0_min=61,
    f0_max=2218,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0.25,
    volume_envelope=1.0,
    output_format="wav",
    speaker_id=0,
    speaker_name="",
    filename_template="{PITCH_SHIFT}{ }{NAME} - {MODEL}{_}{SPEAKER}",
    progress=gr.Progress(track_tqdm=True),
):
    if not rvc_model:
        raise ValueError("Не выбрана модель для RVC-инференса")
    inputs = collect_batch_inputs(dir_input, files)
    if not inputs:
        raise ValueError("Нет аудиофайлов: укажите папку или прикрепите файлы.")

    target_dir = (
        str(output_dir).strip()
        if output_dir and str(output_dir).strip()
        else OUTPUT_DIR
    )
    os.makedirs(target_dir, exist_ok=True)

    print(f"\n[⚙️] Пакетная конвертация: {len(inputs)} файл(ов), модель '{rvc_model}'")
    pipe = get_pipeline(rvc_model)
    done, failed = [], []
    for num, input_path in enumerate(inputs, 1):
        progress(
            num / len(inputs),
            desc=f"[{num}/{len(inputs)}] {os.path.basename(input_path)}",
        )
        try:
            output_path = build_output_path(
                input_path,
                target_dir,
                rvc_model,
                output_format,
                speaker_name,
                filename_template,
                rvc_pitch,
            )
            convert_one(
                pipe,
                input_path,
                output_path,
                f0_method=f0_method,
                f0_min=f0_min,
                f0_max=f0_max,
                rvc_pitch=rvc_pitch,
                protect=protect,
                index_rate=index_rate,
                volume_envelope=volume_envelope,
                output_format=output_format,
                speaker_id=speaker_id,
            )
            done.append(output_path)
        except Exception as error:
            failed.append(f"{os.path.basename(input_path)}: {error}")

    lines = [f"Готово: {len(done)}/{len(inputs)}. Папка: {target_dir}"]
    lines.extend(f"✓ {os.path.basename(path)}" for path in done)
    lines.extend(f"✗ {item}" for item in failed)
    print(f"[✅] Пакетная конвертация завершена — {len(done)}/{len(inputs)}")
    return "\n".join(lines)
