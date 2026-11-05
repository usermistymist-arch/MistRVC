"""Общие константы и компоненты интерфейса MistRVC."""

import os
import re

import gradio as gr

# Пути считаются от корня репозитория, а не от текущего каталога: обучение запускается
# отдельными процессами с cwd = корень проекта (gradio_ui/jobs.py), поэтому интерфейс
# должен смотреть в те же папки, откуда его ни запустили бы.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RVC_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "RVC_models")
RVC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "RVC_output")
UVR_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "UVR_models")
UVR_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "UVR_output")

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
        model = gr.Dropdown(label="Голосовые модели:", choices=list_rvc_models())
    _MODEL_DROPDOWNS.append(model)
    return model


def pitch_slider():
    """Слайдер регулировки высоты тона."""
    return gr.Slider(
        minimum=-24,
        maximum=24,
        step=1,
        value=0,
        label="Регулировка высоты тона",
        info="-24 — Мужская модель | 24 — Женская модель",
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


def conversion_settings():
    """Аккордеон тонких настроек конвертации (классический вид).

    Возвращает словарь компонентов; ключи соответствуют именованным
    аргументам `rvc_infer` / `rvc_batch_infer`
    (кроме модели, входа, тона и формата — они лежат в основной части).
    """
    with gr.Accordion("Настройки преобразования", open=True):
        with gr.Column(variant="panel"):
            with gr.Accordion("Стандартные настройки", open=True):
                with gr.Group():
                    with gr.Column(variant="panel"):
                        f0_method = gr.Dropdown(
                            value="hpa-rmvpe",
                            label="Метод выделения тона",
                            choices=F0_METHODS,
                        )
                    with gr.Column(variant="panel"):
                        index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.1,
                            label="Влияние индекса",
                            info="Влияние, оказываемое индексным файлом; Чем выше значение, тем больше влияние. Однако выбор более низких значений может помочь смягчить артефакты, присутствующие в аудио.",
                        )
                        volume_envelope = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,
                            label="Скорость смешивания RMS",
                            info="Заменить или смешать с огибающей громкости выходного сигнала. Чем ближе значение к 1, тем больше используется огибающая выходного сигнала.",
                        )
                        protect = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            step=0.05,
                            value=0.4,
                            label="Защита согласных",
                            info="Защитить согласные и звуки дыхания, чтобы избежать электроакустических разрывов и артефактов. Максимальное значение параметра 0.5 обеспечивает полную защиту. Уменьшение этого значения может снизить защиту, но уменьшить эффект индексирования.",
                        )

            with gr.Accordion("Дополнительные настройки", open=False):
                with gr.Group():
                    with gr.Row(variant="panel"):
                        f0_min = gr.Slider(
                            minimum=1,
                            maximum=120,
                            step=1,
                            value=62,
                            label="Минимальный диапазон тона",
                            info="Определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                        )
                        f0_max = gr.Slider(
                            minimum=380,
                            maximum=16000,
                            step=1,
                            value=1108,
                            label="Максимальный диапазон тона",
                            info="Определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                        )

            return {
                "f0_method": f0_method,
                "index_rate": index_rate,
                "protect": protect,
                "volume_envelope": volume_envelope,
                "f0_min": f0_min,
                "f0_max": f0_max,
            }
