"""Вкладка «Инструменты»: проверка моделей, разделение чекпоинта, апскейл аудио."""

import os

import gradio as gr

from gradio_ui.common import PROJECT_ROOT

UPSCALE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "Upscale_output")
DEFAULT_SPLIT_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "pretrains")

MODEL_INFO_KEYS = [
    ("model_name", "Модель"),
    ("version", "Версия"),
    ("vocoder", "Вокодер"),
    ("sr", "Частота"),
    ("epoch", "Эпоха"),
    ("step", "Шаг"),
    ("f0", "С питчем"),
]


def _uploaded_path(item):
    if item is None:
        return None
    return getattr(item, "name", None) or (item if isinstance(item, str) else None)


def _inspect_model(model_file) -> str:
    path = _uploaded_path(model_file)
    if not path or not os.path.isfile(path):
        raise gr.Error("Прикрепите .pth файл модели.")
    try:
        import torch
    except ImportError as error:
        raise gr.Error("PyTorch не установлен.") from error
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise gr.Error(f"Не удалось прочитать файл: {error}") from error
    if not isinstance(checkpoint, dict):
        raise gr.Error("Файл не похож на модель.")

    if "generator" in checkpoint and "discriminator" in checkpoint:
        generator = checkpoint.get("generator") or {}
        return "\n".join(
            [
                "Единый чекпоинт обучения (G + D), не модель для инференса.",
                f"Эпоха: {checkpoint.get('epoch', '—')}",
                f"Learning rate: {generator.get('lr', checkpoint.get('learning_rate', '—'))}",
                "",
                "Разделите его ниже, чтобы получить претрейны G/D.",
            ],
        )

    lines = [
        f"Файл: {os.path.basename(path)} ({os.path.getsize(path) / 1024 / 1024:.1f} МБ)"
    ]
    lines.extend(
        f"{label}: {checkpoint[key]}"
        for key, label in MODEL_INFO_KEYS
        if key in checkpoint
    )
    weights = checkpoint.get("weight")
    if isinstance(weights, dict):
        lines.append(f"Слоёв: {len(weights)}")
    if checkpoint.get("version") != "v2" or not checkpoint.get("f0", 1):
        lines.append("")
        lines.append("MuXVS поддерживает только v2-модели с питчем.")
    return "\n".join(lines)


def _split_checkpoint(checkpoint_file, output_dir) -> str:
    from rvc.utils.split_checkpoint import split_checkpoint

    path = _uploaded_path(checkpoint_file)
    if not path:
        raise gr.Error("Укажите путь или прикрепите checkpoint.pth.")
    try:
        return split_checkpoint(path, output_dir.strip() or DEFAULT_SPLIT_DIR)
    except (FileNotFoundError, ValueError) as error:
        raise gr.Error(str(error)) from error


def _upscale_audio(
    audio_file, overlap, progress=gr.Progress(track_tqdm=True)
) -> gr.Audio:
    path = _uploaded_path(audio_file)
    if not path:
        raise gr.Error("Прикрепите аудиофайл.")
    try:
        from rvc._library.config import Config
        from rvc.inference.modules.audio_upscaler import upscale
    except ImportError as error:
        raise gr.Error(f"Нет зависимостей апскейла: {error}") from error

    os.makedirs(UPSCALE_OUTPUT_DIR, exist_ok=True)
    progress(0.1, desc="Улучшаем качество…")
    try:
        # FlashSR грузит свои веса на то же устройство — кэш конвертации не нужен.
        from rvc.inference.infer import release_pipeline

        release_pipeline()
    except Exception:  # noqa: BLE001 — без torch/зависимостей выгружать нечего
        pass
    try:
        upscale(path, UPSCALE_OUTPUT_DIR, int(overlap), Config().device)
    except Exception as error:
        raise gr.Error(f"Апскейл не удался: {error}") from error
    output_path = os.path.join(UPSCALE_OUTPUT_DIR, os.path.basename(path))
    if not os.path.isfile(output_path):
        raise gr.Error(f"Результат не найден: {output_path}")
    return gr.Audio(output_path, label=os.path.basename(output_path))


def _convert_tb_to_txt(tb_files) -> list[str]:
    """Читает файлы событий TensorBoard и возвращает список путей к .txt файлам."""
    if not tb_files:
        raise gr.Error("Прикрепите хотя бы один файл событий TensorBoard (.0, .1 или events.out.tfevents).")
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise gr.Error("Библиотека tensorboard не установлена.") from error

    import tempfile
    import shutil

    temp_dir = os.path.join(PROJECT_ROOT, "output", "tb_converted")
    os.makedirs(temp_dir, exist_ok=True)
    
    output_paths = []
    skipped_files = []
    
    if not isinstance(tb_files, list):
        tb_files = [tb_files]
        
    for tb_file in tb_files:
        path = _uploaded_path(tb_file)
        if not path or not os.path.isfile(path):
            skipped_files.append(f"{os.path.basename(str(tb_file))} (файл не найден)")
            continue
            
        temp_tb_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_tb_dir, "events.out.tfevents.0")
        try:
            shutil.copy2(path, temp_file_path)
            event_acc = EventAccumulator(temp_file_path)
            event_acc.Reload()
        except Exception as error:
            skipped_files.append(f"{os.path.basename(path)} (ошибка чтения: {error})")
            shutil.rmtree(temp_tb_dir, ignore_errors=True)
            continue
        finally:
            shutil.rmtree(temp_tb_dir, ignore_errors=True)
        
        tags = event_acc.Tags()
        scalar_tags = tags.get('scalars', [])
        
        if not scalar_tags:
            all_tags_info = {k: (len(v) if isinstance(v, list) else v) for k, v in tags.items()}
            skipped_files.append(
                f"{os.path.basename(path)} (нет скаляров, найдено: {all_tags_info})"
            )
            continue
            
        lines = []
        lines.append(f"Файл: {os.path.basename(path)}")
        lines.append("=" * 60)
        
        for tag in sorted(scalar_tags):
            events = event_acc.Scalars(tag)
            if not events:
                continue
            
            lines.append(f"\n{tag}:")
            for e in events:
                lines.append(f"     [Шаг {e.step}] - {e.value:.6f}")
                
        txt_content = "\n".join(lines)
        
        base_name = os.path.splitext(os.path.basename(path))[0]
        if "events.out.tfevents" in base_name:
            clean_name = base_name.replace("events.out.tfevents", "metrics").replace(".", "_")
        else:
            clean_name = base_name
            
        txt_filename = f"{clean_name}_converted.txt"
        txt_path = os.path.join(temp_dir, txt_filename)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)
            
        output_paths.append(txt_path)
        
    if skipped_files:
        warning_text = "Пропущены файлы:\n" + "\n".join(f"• {name}" for name in skipped_files)
        gr.Warning(warning_text)
        
    if not output_paths:
        raise gr.Error(
            "Не удалось обработать ни один файл. "
            "Проверьте формат и содержимое. "
            "Смотрите предупреждение выше для подробностей."
        )
        
    return output_paths


def tools_tab():
    with gr.Accordion("Проверка модели (.pth)", open=False):
        model_file = gr.File(
            label="Файл модели", file_types=[".pth"], file_count="single"
        )
        inspect_btn = gr.Button("Проверить", variant="primary")
        model_info = gr.Textbox(label="Сведения", lines=10, interactive=False)
        inspect_btn.click(_inspect_model, inputs=model_file, outputs=model_info)

    with gr.Accordion("Разделение чекпоинта (G / D)", open=False):
        gr.Markdown("Единый `checkpoint.pth` → `G_pretrain.pth` + `D_pretrain.pth`.")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Column() as upload_file:
                    checkpoint_file = gr.File(
                        label="Чекпоинт", file_types=[".pth"], file_count="single"
                    )
                with gr.Column(visible=False) as enter_local_file:
                    checkpoint_path = gr.Textbox(
                        label="Путь к чекпоинту:",
                        info="Введите полный путь к файлу .pth",
                    )
                with gr.Row():
                    show_upload_button = gr.Button(
                        value="Загрузить файл с устройства",
                        visible=False,
                    )
                    show_enter_button = gr.Button(value="Ввести путь к файлу")
            split_dir = gr.Textbox(DEFAULT_SPLIT_DIR, label="Папка результата", scale=3)

        split_btn = gr.Button("Разделить", variant="primary")
        split_message = gr.Textbox(label="Результат", lines=4, interactive=False)

        # Переключение видимости между загрузкой и вводом пути
        show_upload_button.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[upload_file, enter_local_file],
            api_name=False,
        )
        show_enter_button.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[upload_file, enter_local_file],
            api_name=False,
        )
        show_upload_button.click(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[show_upload_button, show_enter_button],
            api_name=False,
        )
        show_enter_button.click(
            lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[show_upload_button, show_enter_button],
            api_name=False,
        )

        # Функция для получения пути из любого источника
        def _resolve_checkpoint_path(file_obj, path_str):
            if file_obj is not None:
                return getattr(file_obj, "name", None) or (
                    file_obj if isinstance(file_obj, str) else None
                )
            if path_str and path_str.strip():
                return path_str.strip()
            return None

        split_btn.click(
            lambda file_obj, path_str, out_dir: _split_checkpoint(
                _resolve_checkpoint_path(file_obj, path_str), out_dir
            ),
            inputs=[checkpoint_file, checkpoint_path, split_dir],
            outputs=split_message,
        )

    with gr.Accordion("Апскейл аудио (FlashSR)", open=False):
        with gr.Row():
            upscale_in = gr.Audio(label="Исходное аудио", type="filepath")
            upscale_out = gr.Audio(label="Улучшенное аудио", interactive=False)
        with gr.Row(equal_height=True):
            overlap = gr.Slider(
                minimum=1, maximum=8, step=1, value=2, label="Перекрытие"
            )
            upscale_btn = gr.Button("Улучшить", variant="primary")
        upscale_btn.click(
            _upscale_audio, inputs=[upscale_in, overlap], outputs=upscale_out
        )
    
    with gr.Accordion("Конвертация TensorBoard eval файлов .0 в .txt", open=False):
        gr.Markdown(
            "Загрузите один или несколько файлов событий TensorBoard (например, `events.out.tfevents...0`), "
            "чтобы преобразовать их метрики (loss, lr и др.) в удобный читаемый текстовый формат."
        )
        with gr.Row():
            tb_files = gr.File(
                label="Файлы TensorBoard", 
                file_types=[".0", ".1", ".tfevents"], 
                file_count="multiple"
            )
            tb_file_output = gr.File(
                label="Скачать результаты (.txt)", 
                file_count="multiple",
                interactive=False
            )
        convert_tb_btn = gr.Button("Конвертировать", variant="primary")
            
        convert_tb_btn.click(
            _convert_tb_to_txt,
            inputs=[tb_files],
            outputs=[tb_file_output]
        )