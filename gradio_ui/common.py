"""Общие константы и компоненты интерфейса MistRVC."""

import os
import re

import gradio as gr

from gradio_ui.settings import filename_template_accordion

# Пути считаются от корня репозитория, а не от текущего каталога: обучение запускается
# отдельными процессами с cwd = корень проекта (gradio_ui/jobs.py), поэтому интерфейс
# должен смотреть в те же папки, откуда его ни запустили бы.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RVC_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "RVC_models")
RVC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "RVC_output")
UVR_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "UVR_models")
UVR_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "UVR_output")
LOADED_AUDIO_DIR = os.path.join(PROJECT_ROOT, "models", "loaded_audio")

OUTPUT_FORMATS = ["wav", "flac", "mp3", "ogg", "m4a"]
F0_METHODS = ["hpa-rmvpe", "rmvpe", "fcpe", "crepe", "crepe-tiny"]


def natural_key(text: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def list_rvc_models() -> list:
    """Имена папок голосовых моделей, отсортированные естественным порядком."""
    if not os.path.isdir(RVC_MODELS_DIR):
        return []
    return sorted(
        (
            name
            for name in os.listdir(RVC_MODELS_DIR)
            if os.path.isdir(os.path.join(RVC_MODELS_DIR, name))
        ),
        key=natural_key,
    )


# Списки моделей во вкладках: обновляются разом после загрузки модели,
# при открытии вкладки и по единственной кнопке «Обновить список».
_MODEL_DROPDOWNS = []


def model_dropdowns() -> list:
    """Все созданные списки голосовых моделей (для вывода обновлений разом)."""
    return list(_MODEL_DROPDOWNS)


def refresh_models() -> list:
    """Обновления для всех списков моделей; подходит как `outputs=model_dropdowns()`."""
    choices = list_rvc_models()
    return [gr.update(choices=choices) for _ in _MODEL_DROPDOWNS]


def model_select() -> gr.Dropdown:
    """Список голосовых моделей; обновляется автоматически (см. `model_dropdowns`)."""
    with gr.Group():
        model = gr.Dropdown(
            label="Голосовая модель", choices=list_rvc_models(), value=None
        )
    _MODEL_DROPDOWNS.append(model)
    return model


def pitch_slider():
    """Слайдер регулировки высоты тона."""
    return gr.Slider(
        minimum=-24,
        maximum=24,
        step=1,
        value=0,
        label="Сдвиг высоты тона",
        info="-12 — на октаву ниже, 12 — на октаву выше",
    )


def process_file_upload(file):
    return file, gr.update(value=file)


def swap_visibility():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=None),
    )


def swap_buttons():
    return gr.update(visible=False), gr.update(visible=True)


def list_loaded_audio() -> "dict[str, str]":
    """{отображаемое имя: абсолютный путь} для всех аудио в папке тестов."""
    from collections import OrderedDict

    found = OrderedDict()
    if not os.path.isdir(LOADED_AUDIO_DIR):
        return found
    for name in sorted(os.listdir(LOADED_AUDIO_DIR), key=natural_key):
        path = os.path.join(LOADED_AUDIO_DIR, name)
        if os.path.isfile(path):
            stem = os.path.splitext(name)[0]
            label = stem.replace("_", " ").strip() or name
            found[label] = path
    return found


def conversion_settings():
    """Настройки конвертации."""
    with gr.Accordion("Настройки конвертации", open=True):
        with gr.Group():
            f0_method = gr.Dropdown(
                value="hpa-rmvpe",
                label="Метод выделения тона",
                choices=F0_METHODS,
            )
            index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=0.1,
                label="Влияние индекса",
                info="Влияние .index файла модели на выходное аудио. Чем выше значение, тем ближе в мелких деталях звучание модели к звучанию датасета, на котором она обучалась, но тем больше артефактов в аудио. [По умолчанию: 0,1]",
            )
            volume_envelope = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.05,
                value=0.5,
                label="Баланс громкости",
                info="Влияние громкости модели на громкость выходного аудио. Чем выше значение, тем ближе к оригинальной громкости модели, чем ниже - тем ближе к громкости входного аудио. [По умолчанию: 0,5]",
            )
            protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                step=0.05,
                value=0.4,
                label="Защита согласных",
                info="Защита согласных и звуков дыхания, во избежание артефактов. Может влиять на акцент модели. Значение 0,5 - полная защита. [По умолчанию: 0,4]",
            )
    name_template = filename_template_accordion()
    return {
        "f0_method": f0_method,
        "index_rate": index_rate,
        "protect": protect,
        "volume_envelope": volume_envelope,
        "name_template": name_template,
    }
