"""Вкладка «Конвертация»: одиночная и пакетная замена голоса (классический вид)."""

import os
from datetime import datetime

import gradio as gr

from gradio_ui.common import (
    OUTPUT_FORMATS,
    RVC_OUTPUT_DIR,
    conversion_settings,
    model_select,
    pitch_slider,
    process_file_upload,
    swap_buttons,
    swap_visibility,
)
from gradio_ui.settings import load_settings, DEFAULT_SETTINGS, update_filename_template


def _get_speaker_update(model_name: str):
    """Читает speakers.txt и возвращает gr.update для dropdown speaker_id."""
    from gradio_ui.common import RVC_MODELS_DIR

    if not model_name:
        return gr.update(visible=False, value=None)

    model_dir = os.path.join(RVC_MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        return gr.update(visible=False, value=None)

    speakers_path = os.path.join(model_dir, "speakers.txt")
    if not os.path.exists(speakers_path):
        return gr.update(visible=False, value=None)

    try:
        speaker_names = []
        with open(speakers_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    name = line.split("=", 1)[1].strip()
                    speaker_names.append(name)

        if len(speaker_names) > 1:
            return gr.update(
                visible=True, choices=speaker_names, value=speaker_names[0]
            )
        else:
            return gr.update(visible=False, value=None)
    except Exception:
        return gr.update(visible=False, value=None)


def _unique_batch_dir() -> str:
    """Новая подпапка пакета внутри общей папки вывода; повторы исключены."""
    stamp = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    target = os.path.join(RVC_OUTPUT_DIR, stamp)
    num = 2
    while os.path.exists(target):
        target = os.path.join(RVC_OUTPUT_DIR, f"{stamp}_{num}")
        num += 1
    return target


def _convert(
    rvc_model,
    input_path,
    f0_method,
    rvc_pitch,
    output_format,
    index_rate,
    protect,
    volume_envelope,
    f0_min,
    f0_max,
    speaker_id,
    progress=gr.Progress(track_tqdm=True),
):
    from rvc.inference.infer import rvc_infer

    settings = load_settings()
    filename_template = settings.get("infer_name_template", "{NAME}_({MODEL}_{SPEAKER})")

    # Преобразуем имя спикера в ID и получаем его текстовое имя
    speaker_id_int = 0
    speaker_name = ""
    if speaker_id is not None and isinstance(speaker_id, str):
        from gradio_ui.common import RVC_MODELS_DIR
        model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
        speakers_path = os.path.join(model_dir, "speakers.txt")
        if os.path.exists(speakers_path):
            with open(speakers_path, encoding="utf-8") as f:
                names = [line.split("=", 1)[1].strip() for line in f if "=" in line]
            if speaker_id in names:
                speaker_id_int = names.index(speaker_id)
                speaker_name = speaker_id
    elif speaker_id is not None:
        speaker_id_int = int(speaker_id)

    return rvc_infer(
        rvc_model=rvc_model,
        input_path=input_path,
        f0_method=f0_method,
        f0_min=f0_min,
        f0_max=f0_max,
        rvc_pitch=rvc_pitch,
        protect=protect,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        output_format=output_format,
        speaker_id=speaker_id_int,
        speaker_name=speaker_name,
        filename_template=filename_template,
        progress=progress,
    )


def _convert_batch(
    rvc_model,
    dir_input,
    files,
    f0_method,
    rvc_pitch,
    output_format,
    index_rate,
    protect,
    volume_envelope,
    f0_min,
    f0_max,
    speaker_id,
    progress=gr.Progress(track_tqdm=True),
):
    from rvc.inference.infer import rvc_batch_infer

    settings = load_settings()
    filename_template = settings.get("infer_name_template", "{NAME}_({MODEL}_{SPEAKER})")

    # Преобразуем имя спикера в ID и получаем его текстовое имя
    speaker_id_int = 0
    speaker_name = ""
    if speaker_id is not None and isinstance(speaker_id, str):
        from gradio_ui.common import RVC_MODELS_DIR
        model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
        speakers_path = os.path.join(model_dir, "speakers.txt")
        if os.path.exists(speakers_path):
            with open(speakers_path, encoding="utf-8") as f:
                names = [line.split("=", 1)[1].strip() for line in f if "=" in line]
            if speaker_id in names:
                speaker_id_int = names.index(speaker_id)
                speaker_name = speaker_id
    elif speaker_id is not None:
        speaker_id_int = int(speaker_id)

    output_dir = _unique_batch_dir()
    summary = rvc_batch_infer(
        rvc_model=rvc_model,
        dir_input=dir_input,
        files=files,
        output_dir=output_dir,
        f0_method=f0_method,
        f0_min=f0_min,
        f0_max=f0_max,
        rvc_pitch=rvc_pitch,
        protect=protect,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        output_format=output_format,
        speaker_id=speaker_id_int,
        speaker_name=speaker_name,
        filename_template=filename_template,
        progress=progress,
    )
    results = sorted(
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, name))
    )
    if not results:
        raise gr.Error(summary)
    return results


def _single_conversion_tab():
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(variant="panel"):
                with gr.Group():
                    rvc_model = model_select()

                    speaker_id = gr.Dropdown(
                        choices=[0],
                        value=0,
                        label="Спикер",
                        visible=False,
                    )

                with gr.Group():
                    rvc_pitch = pitch_slider()

            rvc_model.change(
                fn=_get_speaker_update,
                inputs=rvc_model,
                outputs=speaker_id,
                api_name=False,
            )

        with gr.Column(scale=2, variant="panel"):
            with gr.Column() as upload_file:
                local_file = gr.Audio(label="Аудио", type="filepath")

            with gr.Column(visible=False) as enter_local_file:
                song_input = gr.Textbox(
                    label="Путь к файлу:",
                    info="Введите полный путь к файлу.",
                )

            with gr.Column():
                show_upload_button = gr.Button(
                    value="Загрузить файл с устройства",
                    visible=False,
                )
                show_enter_button = gr.Button(value="Ввести путь к файлу")

    with gr.Group(), gr.Row(equal_height=True):
        generate_btn = gr.Button(
            value="Генерировать",
            variant="primary",
            scale=2,
        )
        converted_voice = gr.Audio(
            label="Преобразованный голос",
            interactive=False,
            scale=9,
        )
        with gr.Column(min_width=160):
            output_format = gr.Dropdown(
                value="flac",
                label="Формат файла",
                choices=OUTPUT_FORMATS,
            )

    settings = conversion_settings()

    # Загрузка файлов
    local_file.change(
        process_file_upload,
        inputs=local_file,
        outputs=[song_input, local_file],
        api_name=False,
    )

    # Обновление кнопок
    show_upload_button.click(
        swap_visibility,
        outputs=[upload_file, enter_local_file, song_input, local_file],
        api_name=False,
    )
    show_enter_button.click(
        swap_visibility,
        outputs=[enter_local_file, upload_file, song_input, local_file],
        api_name=False,
    )
    show_upload_button.click(
        swap_buttons, outputs=[show_upload_button, show_enter_button], api_name=False
    )
    show_enter_button.click(
        swap_buttons, outputs=[show_enter_button, show_upload_button], api_name=False
    )

    # Обновление списка моделей — внутри model_select().

    # Запуск процесса преобразования
    generate_btn.click(
        _convert,
        inputs=[
            rvc_model,
            song_input,
            settings["f0_method"],
            rvc_pitch,
            output_format,
            settings["index_rate"],
            settings["protect"],
            settings["volume_envelope"],
            settings["f0_min"],
            settings["f0_max"],
            speaker_id,
        ],
        outputs=[converted_voice],
    )


def _batch_conversion_tab():
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(variant="panel"):
                with gr.Group():
                    rvc_model = model_select()

                    speaker_id = gr.Dropdown(
                        choices=[0],
                        value=0,
                        label="Спикер",
                        visible=False,
                    )

                with gr.Group():
                    rvc_pitch = pitch_slider()

            rvc_model.change(
                fn=_get_speaker_update,
                inputs=rvc_model,
                outputs=speaker_id,
                api_name=False,
            )

        with gr.Column(scale=2, variant="panel"):
            dir_input = gr.Textbox(
                label="Папка с аудио:",
                info="Введите полный путь к папке с аудиофайлами.",
            )
            batch_files = gr.File(
                label="…или прикрепите файлы", file_count="multiple", height=180
            )

    with gr.Group(), gr.Row(equal_height=True):
        batch_btn = gr.Button(
            value="Генерировать пакет",
            variant="primary",
            scale=2,
        )
        batch_result = gr.File(
            label="Результат",
            file_count="multiple",
            height=190,
            interactive=False,
            scale=9,
        )
        with gr.Column(min_width=160):
            output_format = gr.Dropdown(
                value="flac",
                label="Формат файла",
                choices=OUTPUT_FORMATS,
            )

    settings = conversion_settings()

    batch_btn.click(
        _convert_batch,
        inputs=[
            rvc_model,
            dir_input,
            batch_files,
            settings["f0_method"],
            rvc_pitch,
            output_format,
            settings["index_rate"],
            settings["protect"],
            settings["volume_envelope"],
            settings["f0_min"],
            settings["f0_max"],
            speaker_id,
        ],
        outputs=[batch_result],
    )

def _conversion_settings_tab():
    """Подвкладка с настройками, специфичными для конвертации."""
    settings = load_settings()
    current_template = settings.get("infer_name_template", DEFAULT_SETTINGS["infer_name_template"])
    
    with gr.Column():
        gr.Markdown("### 📁 Именование файлов")
        gr.Markdown(
            "Шаблон имени выходного файла при конвертации. Доступные ключи: `{NAME}`, `{MODEL}`, `{SPEAKER}`."
        )
        template_input = gr.Textbox(
            label="Шаблон имени файла",
            value=current_template,
            info="Пример: {NAME}_({MODEL}_{SPEAKER})",
        )
        save_template_btn = gr.Button("Сохранить настройки именования", variant="primary")
        status_text = gr.Textbox(label="Статус", interactive=False)

    save_template_btn.click(
        fn=update_filename_template,
        inputs=[template_input],
        outputs=[status_text],
    )

def conversion_tab():
    with gr.Tab("Одиночная конвертация"):
        _single_conversion_tab()
    with gr.Tab("Пакетная конвертация"):
        _batch_conversion_tab()
    with gr.Tab("Настройки"):
        _conversion_settings_tab()