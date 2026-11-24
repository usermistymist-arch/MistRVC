"""Вкладка «Модели»: загрузка RVC-моделей и установка эмбеддеров."""

import os
import shutil
import urllib.parse
import urllib.request

import gradio as gr

from gradio_ui.common import (
    PROJECT_ROOT,
    model_dropdowns,
    refresh_models,
    swap_buttons,
    swap_visibility,
)

EMBEDDERS_DIR = os.path.join(PROJECT_ROOT, "assets", "models", "embedders")
EMBEDDER_PATH = os.path.join(EMBEDDERS_DIR, "contentvec_base.pt")
EMBEDDER_BASE_URL = (
    "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"
)
EMBEDDER_MODELS = [
    "contentvec_base.pt",
    "hubert_base.pt",
    "korean_hubert_base.pt",
    "chinese_hubert_base.pt",
    "portuguese_hubert_base.pt",
    "japanese_hubert_base.pt",
]


def _download_from_url(url, dir_name, progress=gr.Progress()):
    from rvc.inference.modules.model_manager import download_from_url

    return download_from_url(url, dir_name, progress=progress)


def _upload_zip(zip_file, dir_name, progress=gr.Progress()):
    from rvc.inference.modules.model_manager import upload_zip_file

    return upload_zip_file(zip_file, dir_name, progress=progress)


def _resolve_file(item):
    """Извлекает путь из объекта gr.File или из текстового поля."""
    if item is None:
        return None
    if isinstance(item, str):
        return item.strip() if item.strip() else None
    return getattr(item, "name", None)


def _upload_files_with_paths(
    pth_file,
    pth_path,
    index_file,
    index_path,
    speakers_file,
    speakers_path,
    dir_name,
    progress=gr.Progress(),
):
    import types

    from rvc.inference.modules.model_manager import upload_separate_files

    # Создаём обёртки для совместимости с upload_separate_files (ожидает .name)
    def wrap_path(file_obj, path_str):
        resolved = _resolve_file(path_str)
        if resolved:
            if not os.path.isfile(resolved):
                raise gr.Error(f"Файл не найден: {resolved}")
            return types.SimpleNamespace(name=resolved)
        return file_obj

    pth_final = wrap_path(pth_file, pth_path)
    index_final = wrap_path(index_file, index_path)
    speakers_final = wrap_path(speakers_file, speakers_path)

    if not pth_final:
        raise gr.Error("Файл .pth обязателен: прикрепите его или укажите путь.")
    if not (dir_name or "").strip():
        raise gr.Error("Укажите имя модели.")

    return upload_separate_files(
        pth_final, index_final, speakers_final, dir_name, progress=progress
    )


def _install_embedder(model_name, custom_url, progress=gr.Progress()):
    try:
        if custom_url:
            if not custom_url.endswith((".pt", "?download=true")):
                return "Ошибка: URL должен вести к .pt файлу или заканчиваться на '?download=true'."
            if urllib.parse.urlparse(custom_url).netloc != "huggingface.co":
                return "Ошибка: разрешён только huggingface.co."
            model_url, model_name = (
                custom_url,
                os.path.basename(urllib.parse.urlparse(custom_url).path),
            )
        else:
            model_url = EMBEDDER_BASE_URL + model_name

        tmp_path = os.path.join(EMBEDDERS_DIR, "tmp_model.pt")
        progress(0.4, desc=f'[~] Установка "{model_name}"…')
        with urllib.request.urlopen(model_url) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)

        progress(0.8, desc="[~] Замена эмбеддера…")
        if os.path.exists(EMBEDDER_PATH):
            os.remove(EMBEDDER_PATH)
        os.rename(tmp_path, EMBEDDER_PATH)
        return f'Модель "{model_name}" установлена.'
    except Exception as error:
        return f'Ошибка установки "{model_name}": {error}'


def _toggle_custom_embedder(enabled: bool):
    if enabled:
        return gr.update(visible=True, value=""), gr.update(visible=False)
    return gr.update(visible=False, value=""), gr.update(
        visible=True, value="contentvec_base.pt"
    )


def models_tab(offline: bool):
    # Единственная кнопка обновления списка: остальные вкладки подхватывают
    # изменения сами — после загрузки модели и при открытии вкладки.
    with gr.Group(), gr.Row(equal_height=True):
        refresh_btn = gr.Button("Обновить список моделей", variant="secondary", scale=1)
        gr.Markdown(
            "Список голосовых моделей обновляется сам: после загрузки и при открытии вкладки «Конвертация»."
        )
    refresh_btn.click(refresh_models, outputs=model_dropdowns(), api_name=False)

    if not offline:
        with gr.Accordion("По ссылке (ZIP)", open=True):
            gr.Markdown(
                "HuggingFace · Pixeldrain · Google Drive · Mega · Яндекс Диск · Dropbox"
            )
            with gr.Row():
                zip_link = gr.Textbox(label="Ссылка", scale=3)
                url_name = gr.Textbox(label="Имя модели", scale=2)
            url_btn = gr.Button("Скачать", variant="primary")

    with gr.Accordion("ZIP-файл", open=False):
        zip_file = gr.File(
            label="ZIP", file_types=[".zip"], file_count="single", scale=3
        )
        zip_name = gr.Textbox(label="Имя модели", scale=2)
        zip_btn = gr.Button("Загрузить", variant="primary")

    with gr.Accordion("Раздельно .pth, .index, speakers.txt", open=False):
        gr.Markdown("Файл speakers.txt опционален, нужен лишь мультиспикерным моделям.")
        with gr.Row():
            with gr.Column():
                with gr.Column() as pth_upload:
                    pth_file = gr.File(
                        label=".pth", file_types=[".pth"], file_count="single"
                    )
                with gr.Column(visible=False) as pth_enter:
                    pth_path = gr.Textbox(
                        label="Путь к .pth:",
                        info="Введите полный путь к файлу .pth",
                    )
                with gr.Row():
                    pth_show_upload = gr.Button(
                        value="Загрузить .pth", visible=False, scale=1
                    )
                    pth_show_enter = gr.Button(value="Ввести путь к .pth", scale=1)
            with gr.Column():
                with gr.Column() as index_upload:
                    index_file = gr.File(
                        label=".index", file_types=[".index"], file_count="single"
                    )
                with gr.Column(visible=False) as index_enter:
                    index_path = gr.Textbox(
                        label="Путь к .index:",
                        info="Введите полный путь к файлу .index",
                    )
                with gr.Row():
                    index_show_upload = gr.Button(
                        value="Загрузить .index", visible=False, scale=1
                    )
                    index_show_enter = gr.Button(value="Ввести путь к .index", scale=1)
            with gr.Column():
                with gr.Column() as speakers_upload:
                    speakers_file = gr.File(
                        label="speakers.txt",
                        file_types=[".txt"],
                        file_count="single",
                    )
                with gr.Column(visible=False) as speakers_enter:
                    speakers_path = gr.Textbox(
                        label="Путь к speakers.txt:",
                        info="Введите полный путь к файлу speakers.txt",
                    )
                with gr.Row():
                    speakers_show_upload = gr.Button(
                        value="Загрузить speakers.txt", visible=False, scale=1
                    )
                    speakers_show_enter = gr.Button(
                        value="Ввести путь к speakers.txt", scale=1
                    )
        files_name = gr.Textbox(label="Имя модели")
        files_btn = gr.Button("Загрузить", variant="primary")

    if not offline:
        with gr.Accordion("Эмбеддер", open=False):
            gr.Markdown("Менять нужно, только если обучали модель с другим эмбеддером.")
            with gr.Row():
                custom_check = gr.Checkbox(False, label="Свой URL")
                custom_url = gr.Textbox(label="URL модели", visible=False, scale=3)
                embedder_drop = gr.Dropdown(
                    EMBEDDER_MODELS,
                    value="contentvec_base.pt",
                    label="Эмбеддер",
                    scale=3,
                )
            embedder_btn = gr.Button("Установить", variant="primary")
            custom_check.change(
                _toggle_custom_embedder,
                inputs=custom_check,
                outputs=[custom_url, embedder_drop],
                api_name=False,
            )

    # Показываем результат после всех полей и кнопок, чтобы его не путали с URL.
    message = gr.Textbox(label="Результат", interactive=False)

    if not offline:
        url_btn.click(
            _download_from_url, inputs=[zip_link, url_name], outputs=message
        ).then(
            refresh_models,
            outputs=model_dropdowns(),
            api_name=False,
        )
    zip_btn.click(_upload_zip, inputs=[zip_file, zip_name], outputs=message).then(
        refresh_models,
        outputs=model_dropdowns(),
        api_name=False,
    )
    # Переключение видимости для .pth
    pth_show_upload.click(
        swap_visibility,
        outputs=[pth_upload, pth_enter, pth_path, pth_file],
        api_name=False,
    )
    pth_show_enter.click(
        swap_visibility,
        outputs=[pth_enter, pth_upload, pth_path, pth_file],
        api_name=False,
    )
    pth_show_upload.click(
        swap_buttons, outputs=[pth_show_upload, pth_show_enter], api_name=False
    )
    pth_show_enter.click(
        swap_buttons, outputs=[pth_show_enter, pth_show_upload], api_name=False
    )

    # Переключение видимости для .index
    index_show_upload.click(
        swap_visibility,
        outputs=[index_upload, index_enter, index_path, index_file],
        api_name=False,
    )
    index_show_enter.click(
        swap_visibility,
        outputs=[index_enter, index_upload, index_path, index_file],
        api_name=False,
    )
    index_show_upload.click(
        swap_buttons, outputs=[index_show_upload, index_show_enter], api_name=False
    )
    index_show_enter.click(
        swap_buttons, outputs=[index_show_enter, index_show_upload], api_name=False
    )

    # Переключение видимости для speakers.txt
    speakers_show_upload.click(
        swap_visibility,
        outputs=[speakers_upload, speakers_enter, speakers_path, speakers_file],
        api_name=False,
    )
    speakers_show_enter.click(
        swap_visibility,
        outputs=[speakers_enter, speakers_upload, speakers_path, speakers_file],
        api_name=False,
    )
    speakers_show_upload.click(
        swap_buttons,
        outputs=[speakers_show_upload, speakers_show_enter],
        api_name=False,
    )
    speakers_show_enter.click(
        swap_buttons,
        outputs=[speakers_show_enter, speakers_show_upload],
        api_name=False,
    )
    files_btn.click(
        _upload_files_with_paths,
        inputs=[
            pth_file,
            pth_path,
            index_file,
            index_path,
            speakers_file,
            speakers_path,
            files_name,
        ],
        outputs=message,
    ).then(
        refresh_models,
        outputs=model_dropdowns(),
        api_name=False,
    )
    if not offline:
        embedder_btn.click(
            _install_embedder, inputs=[embedder_drop, custom_url], outputs=message
        )
