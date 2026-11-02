# MistRVC

Форк проекта [MuXVS](https://github.com/MuXolotl/MuXVS), адаптированный для удобного использования в Google Colab и локально. 

Конвертация голоса (RVCv2), синтез речи (TTS), разделение аудио (UVR) и обучение моделей в одном минималистичном веб-интерфейсе на Gradio. Работает на Python 3.13.

## Основные возможности

- **Конвертация** — замена голоса с тонкими настройками и пакетной обработкой.
- **TTS** — синтез речи из текста (EdgeTTS) с последующей конвертацией в целевой голос.
- **UVR** — разделение аудио на стемы (Roformer, MDX23C, MDX-NET, VR Arch, Demucs).
- **Обучение** — полный цикл: нарезка, извлечение признаков, обучение и создание индекса.
- **Модели** — загрузка RVC-моделей по ссылке, из ZIP или локальных файлов.
- **Инструменты** — проверка моделей, разделение чекпоинтов, апскейл аудио.

## Быстрый старт

### Google Colab
Откройте ноутбук и следуйте инструкциям в ячейках:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usermistymist-arch/MistRVC/blob/main/assets/MistRVC_Colab.ipynb)

### Локальный запуск (Windows)
1. Запустите `run-MistRVC-installer.bat` (установит зависимости).
2. Запустите `run-MistRVC.bat` (запустит веб-интерфейс).

### Локальный запуск (вручную)
```bash
python app.py --port 4000
```

## CLI
```bash
python -m rvc.inference.infer_cli rvc --rvc_model MyModel --input_path in.wav
python -m rvc.inference.infer_cli tts --rvc_model MyModel --tts_voice ru-RU-SvetlanaNeural --tts_text "Привет"
```

## Примечание о разработке
Этот форк активно разрабатывается с использованием AI-ассистента (Qwen). Как и в любом проекте, создаваемом в парадигме vibe coding, здесь возможны неточности или ошибки, характерные для автоматической генерации кода. Проверяйте изменения и используйте на свой страх и риск.

## Благодарности
Этот проект основан на коде [MuXVS](https://github.com/MuXolotl/MuXVS) от MuXolotl. Спасибо автору за отличную базу и реализацию fairseq для Python 3.13.
