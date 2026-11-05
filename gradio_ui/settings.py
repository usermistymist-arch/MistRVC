import json
import os

import gradio as gr

# Путь к файлу настроек на Google Drive (для Colab)
SETTINGS_DIR = os.environ.get("MISTRV_SETTINGS_DIR", "/content/drive/MyDrive/MistRVC")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")

# Дефолтные настройки
DEFAULT_SETTINGS = {
    "theme_color": "teal",
    "infer_name_template": "{NAME}_({MODEL}_{SPEAKER})",
}

# Список поддерживаемых цветов Gradio (отображаемое имя, внутреннее значение)
# Задача 5 выполнена здесь: "Серый" -> "Металлический", "Нейтральный" -> "Серый"
AVAILABLE_COLORS = [
    ("Индиго", "indigo"),
    ("Синий", "blue"),
    ("Фиолетовый", "purple"),
    ("Розовый", "pink"),
    ("Красный", "red"),
    ("Оранжевый", "orange"),
    ("Жёлтый", "yellow"),
    ("Зелёный", "green"),
    ("Бирюзовый", "teal"),
    ("Металлический", "slate"),
    ("Серый", "neutral"),
]


def load_settings():
    """Загружает настройки из файла на Google Drive."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                settings = json.load(f)
                # Добавляем дефолтное значение шаблона, если его нет в старом файле
                if "infer_name_template" not in settings:
                    settings["infer_name_template"] = DEFAULT_SETTINGS["infer_name_template"]
                return settings
        except Exception:
            pass
    return DEFAULT_SETTINGS.copy()


def save_theme_color(color):
    """Сохраняет только цвет темы, не трогая другие настройки."""
    settings = load_settings()
    settings["theme_color"] = color
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

def update_filename_template(template: str):
    """Обновляет только шаблон именования, сохраняя остальные настройки."""
    settings = load_settings()
    settings["infer_name_template"] = template
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    return f"✅ Шаблон сохранён: '{template}'"

def apply_theme(color):
    """Сохраняет цвет и возвращает сообщение о статусе."""
    save_theme_color(color)
    return f"✅ Цвет темы сохранён: '{color}'. Перезапустите интерфейс для применения."

def settings_tab():
    """Создаёт содержимое вкладки 'Настройки' (только глобальные)."""
    settings = load_settings()
    current_color = settings.get("theme_color", DEFAULT_SETTINGS["theme_color"])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 Цвет интерфейса")
            gr.Markdown(
                "Изменения вступят в силу после перезапуска интерфейса (ячейки кода в Google Colab)."
            )
            color_dropdown = gr.Dropdown(
                label="Цвет темы",
                choices=AVAILABLE_COLORS,
                value=current_color,
            )
            apply_btn = gr.Button("Сохранить цвет", variant="primary")
            status_text = gr.Textbox(label="Статус", interactive=False)

    apply_btn.click(
        fn=apply_theme,
        inputs=[color_dropdown],
        outputs=[status_text],
    )