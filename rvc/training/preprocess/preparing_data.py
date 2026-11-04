import logging
import os
import sys
import traceback
import warnings
from random import shuffle

# Установка переменных окружения и отключение предупреждений
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Корень репозитория: скрипт может запускаться из любого каталога
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(PROJECT_ROOT)
from rvc._library.audio import load_audio
from rvc._library.embedders.fairseq import load_model
from rvc._library.predictors.f0 import RMVPEF0

import re

def natural_sort_key(s):
    """Ключ для натуральной сортировки: числа сравниваются как числа."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

exp_dir = str(
    sys.argv[1]
)  # Директория с данными, подготовленными скриптом `preprocess.py`
f0_method = str(sys.argv[2])  # Метод извлечения F0 / rmvpe, rmvpe+, hpa-rmvpe
sample_rate = int(sys.argv[3])  # Частота дискретизации для генерации filelist.txt
include_mutes = int(
    sys.argv[4]
)  # Количество мьют файлов на одного спикера / По умолчанию = 2


class DataPreprocessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Рабочая частота F0 и HuBERT — обе модели требуют 16 кГц
        self.sample_rate = 16000
        self.hop_size = 160
        self.f0_bin = 256
        self.f0_min = 50.0
        self.f0_max = 2218.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Инициализация моделей
        self.model_rmvpe = RMVPEF0(self.device)
        self.hubert_model = self._load_hubert_model()
        # process_files переключит на медленный путь, если 16 кГц-кэша нет
        self.use_16k_cache = True

    def _load_hubert_model(self):
        """Загрузка модели HuBERT"""
        hubert_model_path = os.path.join(
            PROJECT_ROOT, "assets", "models", "embedders", "contentvec_base.pt"
        )
        return load_model(hubert_model_path).to(self.device).eval()

    def _read_16k(self, path):
        """Моно 16 кГц для F0/HuBERT: из кэша нарезки — напрямую, иначе — ресемпл."""
        if self.use_16k_cache:
            wav, sr = sf.read(path)
            assert sr == 16000
            if wav.ndim == 2:
                wav = wav.mean(-1)
            return np.ascontiguousarray(wav, dtype=np.float32)
        return load_audio(path, self.sample_rate)

    def compute_f0(self, path, f0_method):
        """Вычисление F0"""
        audio = self._read_16k(path)
        return self.model_rmvpe.get_f0(audio, self.f0_min, self.f0_max, f0_method)

    def coarse_f0(self, f0):
        """Квантование F0"""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def extract_features(self, wav_path):
        """Извлечение признаков HuBERT"""
        audio = self._read_16k(wav_path)
        source = torch.from_numpy(audio).float().view(1, -1).to(self.device)
        padding_mask = torch.zeros(source.shape, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits = self.hubert_model.extract_features(
                source=source, padding_mask=padding_mask, output_layer=12
            )
            return logits[0].squeeze(0).float().cpu().numpy()

    def process_files(self):
        cache_root = f"{exp_dir}/data/sliced_audios_16k"
        gt_root = f"{exp_dir}/data/sliced_audios"
        
        # Проверяем, есть ли подпапки (мульти-спикер)
        speaker_dirs = [d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))]
        
        inp_root = cache_root

        f0_quant_path = f"{exp_dir}/data/f0_quantized"
        f0_voiced_path = f"{exp_dir}/data/f0_voiced"
        features_path = f"{exp_dir}/data/features"

        os.makedirs(f0_quant_path, exist_ok=True)
        os.makedirs(f0_voiced_path, exist_ok=True)
        os.makedirs(features_path, exist_ok=True)

        # Сбор файлов для обработки (рекурсивно)
        files_to_process = []
        if speaker_dirs:
            for speaker_dir in speaker_dirs:
                speaker_inp = os.path.join(inp_root, speaker_dir)
                if os.path.isdir(speaker_inp):
                    for f in os.listdir(speaker_inp):
                        if f.endswith(".wav"):
                            files_to_process.append((speaker_dir, f))
        else:
            for f in os.listdir(inp_root):
                if f.endswith(".wav"):
                    files_to_process.append(("", f))
                    
        if not files_to_process:
            self._raise_no_files_error()

        print("\n[2/3] - Запуск процесса извлечения признаков...")

        # Обработка F0
        for speaker_dir, file in tqdm(
            files_to_process,
            desc="Извлечение F0 (фундаментальной частоты)",
            bar_format="{desc}: {n}/{total}",
            ncols=80,
        ):
            try:
                inp_path = os.path.join(inp_root, speaker_dir, file) if speaker_dir else os.path.join(inp_root, file)
                
                f0_quant_dir = os.path.join(f0_quant_path, speaker_dir) if speaker_dir else f0_quant_path
                f0_voiced_dir = os.path.join(f0_voiced_path, speaker_dir) if speaker_dir else f0_voiced_path
                
                os.makedirs(f0_quant_dir, exist_ok=True)
                os.makedirs(f0_voiced_dir, exist_ok=True)
                
                opt_path1 = os.path.join(f0_quant_dir, file)
                opt_path2 = os.path.join(f0_voiced_dir, file)

                if not (os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy")):
                    featur_pit = self.compute_f0(inp_path, f0_method)
                    np.save(opt_path2, featur_pit, allow_pickle=False)
                    coarse_pit = self.coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)
            except Exception:
                raise RuntimeError(f"Ошибка извлечения тона!\nФайл - {inp_path}\n{traceback.format_exc()}")

        # Обработка HuBERT
        for speaker_dir, file in tqdm(
            files_to_process,
            desc="Извлечение семантических признаков HuBERT",
            bar_format="{desc}: {n}/{total}",
            ncols=80,
        ):
            try:
                wav_path = os.path.join(inp_root, speaker_dir, file) if speaker_dir else os.path.join(inp_root, file)
                
                features_dir = os.path.join(features_path, speaker_dir) if speaker_dir else features_path
                os.makedirs(features_dir, exist_ok=True)
                
                out_path = os.path.join(features_dir, file.replace(".wav", ".npy"))

                if not os.path.exists(out_path):
                    feats = self.extract_features(wav_path)
                    if np.isnan(feats).sum() > 0:
                        raise TypeError(f"Файл {file} содержит некорректные значения (NaN).")
                    np.save(out_path, feats, allow_pickle=False)
            except Exception:
                raise RuntimeError(f"Ошибка извлечения признаков!\nФайл - {wav_path}\n{traceback.format_exc()}")

        print("✓ Извлечение акустических признаков успешно завершено!")
        
    def _raise_no_files_error(self):
        error_message = (
            "ОШИБКА: Не найдено ни одного фрагмента для обработки.\n"
            "Возможные причины:\n"
            "1. Датасет не имеет звука.\n"
            "2. Датасет слишком тихий.\n"
            "3. Датасет слишком короткий (менее 3 секунд).\n"
            "4. Датасет слишком длинный (более 1 часа одним файлом).\n\n"
            "Попробуйте увеличить громкость или объем датасета. Если у вас один большой файл, можно разделить его на несколько более мелких."
        )
        raise FileNotFoundError(error_message)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    mute_base_path = os.path.join(PROJECT_ROOT, "rvc", "training", "mute")

    gt_wavs_dir = os.path.join(model_path, "data", "sliced_audios")
    feature_dir = os.path.join(model_path, "data", "features")
    f0_dir = os.path.join(model_path, "data", "f0_quantized")
    f0nsf_dir = os.path.join(model_path, "data", "f0_voiced")

    options = []
    
    # Получаем список папок спикеров
    speaker_dirs = [d for d in os.listdir(gt_wavs_dir) if os.path.isdir(os.path.join(gt_wavs_dir, d))]
    speaker_dirs.sort(key=natural_sort_key) # Для детерминированного порядка
    
    if not speaker_dirs:
        # Fallback на случай, если папок нет (старая структура)
        speaker_dirs = [""]

    for speaker_id_str in speaker_dirs:
        speaker_gt_dir = os.path.join(gt_wavs_dir, speaker_id_str) if speaker_id_str else gt_wavs_dir
        speaker_feature_dir = os.path.join(feature_dir, speaker_id_str) if speaker_id_str else feature_dir
        speaker_f0_dir = os.path.join(f0_dir, speaker_id_str) if speaker_id_str else f0_dir
        speaker_f0nsf_dir = os.path.join(f0nsf_dir, speaker_id_str) if speaker_id_str else f0nsf_dir
        
        if not os.path.isdir(speaker_gt_dir):
            continue
            
        gt_wavs_files = set(name.split(".")[0] for name in os.listdir(speaker_gt_dir) if name.endswith(".wav"))
        
        valid_names = []
        for name in gt_wavs_files:
            if (os.path.exists(os.path.join(speaker_feature_dir, f"{name}.npy")) and
                os.path.exists(os.path.join(speaker_f0_dir, f"{name}.wav.npy")) and
                os.path.exists(os.path.join(speaker_f0nsf_dir, f"{name}.wav.npy"))):
                valid_names.append(name)
                
        speaker_id = int(speaker_id_str) if speaker_id_str else 0
        
        for name in valid_names:
            options.append(
                f"{os.path.join(speaker_gt_dir, name)}.wav|"
                f"{os.path.join(speaker_feature_dir, name)}.npy|"
                f"{os.path.join(speaker_f0_dir, name)}.wav.npy|"
                f"{os.path.join(speaker_f0nsf_dir, name)}.wav.npy|{speaker_id}"
            )
            
        # Добавляем мьют-файлы для каждого спикера
        if include_mutes > 0:
            mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
            mute_feature_path = os.path.join(mute_base_path, "features", "mute.npy")
            mute_f0_path = os.path.join(mute_base_path, "f0_quantized", "mute.wav.npy")
            mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

            for _ in range(include_mutes):
                options.append(
                    f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{speaker_id}"
                )

    shuffle(options)

    with open(
        os.path.join(model_path, "data", "filelist.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(options))
        
    # Сохраняем имена спикеров для инференса
    speakers_path = os.path.join(model_path, "speakers.txt")
    if speaker_dirs:
        with open(speakers_path, "w", encoding="utf-8") as f:
            for idx, speaker_dir in enumerate(speaker_dirs):
                f.write(f"speaker_{idx} = {speaker_dir}\n")
    else:
        with open(speakers_path, "w", encoding="utf-8") as f:
            f.write("speaker_0 = Speaker 0\n")


if __name__ == "__main__":
    try:
        preprocessor = DataPreprocessor()
        preprocessor.process_files()

        generate_filelist(exp_dir, sample_rate, include_mutes)
    except Exception as e:
        print(f"Критическая ошибка: {e!s}")
        print(traceback.format_exc())
        sys.exit(1)
