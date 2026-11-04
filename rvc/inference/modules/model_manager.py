import os
import shutil
import sys
import zipfile

import gradio as gr

from rvc.inference.modules.download_source import download_file

# Путь к директории, где будут храниться модели RVC (от корня репозитория)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
rvc_models_dir = os.path.join(PROJECT_ROOT, "models", "RVC_models")
os.makedirs(rvc_models_dir, exist_ok=True)


# Распаковывает zip-файл в указанную директорию и находит файлы модели (.pth и .index)
def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath, speakers_filepath = None, None, None
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            file_path = os.path.join(root, name)
            if name.endswith(".index") and os.stat(file_path).st_size > 1024 * 100:
                index_filepath = file_path
            if name.endswith(".pth") and os.stat(file_path).st_size > 1024 * 1024 * 40:
                model_filepath = file_path
            if name == "speakers.txt":
                speakers_filepath = file_path

    if not model_filepath:
        raise gr.Error(f"Не найден файл модели .pth в распакованном zip-файле. Проверьте содержимое в {extraction_folder}.")

    rename_and_cleanup(extraction_folder, model_filepath, index_filepath, speakers_filepath)


# Функция для переименования файлов и удаления пустых папок
def rename_and_cleanup(extraction_folder, model_filepath, index_filepath, speakers_filepath=None):
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))
    if speakers_filepath:
        os.rename(speakers_filepath, os.path.join(extraction_folder, "speakers.txt"))

    for filepath in os.listdir(extraction_folder):
        full_path = os.path.join(extraction_folder, filepath)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)


# Основная функция для скачивания модели по ссылке и распаковки zip-файла
def download_from_url(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f"[~] Загрузка голосовой модели {dir_name}...")
        zip_name = os.path.join(rvc_models_dir, dir_name + ".zip")
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            # Проверка на наличие директории с таким именем
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели."
            )

        download_file(url, zip_name, progress)  # Скачивание файла
        progress(0.8, desc="[~] Распаковка zip-файла...")
        extract_zip(extraction_folder, zip_name)  # Распаковка zip-файла
        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        # Обработка ошибок при загрузке модели
        raise gr.Error(f"Ошибка при загрузке модели: {e!s}")


# Функция для загрузки и распаковки zip-файла модели через интерфейс
def upload_zip_file(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели."
            )

        zip_name = zip_path.name
        progress(0.8, desc="[~] Распаковка zip-файла...")
        extract_zip(extraction_folder, zip_name)  # Распаковка zip-файла
        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        # Обработка ошибок при загрузке и распаковке
        raise gr.Error(f"Ошибка при загрузке модели: {e!s}")


# Функция для загрузки отдельных файлов модели (.pth и .index)
def upload_separate_files(pth_file, index_file, speakers_file, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f"Директория голосовой модели {dir_name} уже существует! Выберите другое имя.")

        os.makedirs(extraction_folder, exist_ok=True)

        progress(0.3, desc="[~] Загрузка .pth файла...")
        if pth_file:
            pth_path = os.path.join(extraction_folder, os.path.basename(pth_file.name))
            shutil.copyfile(pth_file.name, pth_path)

        progress(0.6, desc="[~] Загрузка .index файла...")
        if index_file:
            index_path = os.path.join(extraction_folder, os.path.basename(index_file.name))
            shutil.copyfile(index_file.name, index_path)

        progress(0.9, desc="[~] Загрузка файла спикеров...")
        if speakers_file:
            speakers_path = os.path.join(extraction_folder, "speakers.txt")
            shutil.copyfile(speakers_file.name, speakers_path)

        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        raise gr.Error(f"Ошибка при загрузке модели: {e!s}")


# Основная функция для вызова из командной строки
def main():
    if len(sys.argv) != 3:
        print(
            '\nИспользование:\npython3 -m rvc.inference.modules.model_manager "url" "dir_name"\n'
        )
        sys.exit(1)

    url = sys.argv[1]
    dir_name = sys.argv[2]

    try:
        # Создаем "фейковый" progress-объект для консоли
        class ConsoleProgress:
            def __call__(self, progress, desc=""):
                print(desc)

        result = download_from_url(url, dir_name, progress=ConsoleProgress())
        print(result)
    except gr.Error as e:
        print(f"Ошибка: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
