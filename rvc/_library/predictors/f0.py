import os

import numpy as np
import torch
import torchcrepe
from scipy.signal import resample_poly
from torchfcpe import spawn_bundled_infer_model

from rvc._library.predictors.rmvpe import RMVPE


def median_interp_pitch(f0):
    f0 = np.where(f0 == 0, np.nan, f0)
    return float(
        np.median(
            np.interp(np.arange(len(f0)), np.where(~np.isnan(f0))[0], f0[~np.isnan(f0)])
        )
    )


def calc_pitch_shift(f0, target_f0=155.0, limit_f0=12):
    return max(
        -limit_f0,
        min(limit_f0, int(np.round(12 * np.log2(target_f0 / median_interp_pitch(f0))))),
    )


class RMVPEF0:
    def __init__(self, device, sample_rate=16000):
        self.device = device
        self.sample_rate = sample_rate
        self.model_og = RMVPE(
            os.path.join("assets", "models", "predictors", "rmvpe.pt"),
            self.device,
            hpa=False,
        )
        self.model_hpa = RMVPE(
            os.path.join("assets", "models", "predictors", "hpa-rmvpe.pt"),
            self.device,
            hpa=True,
        )

    def get_f0(self, audio, f0_min=61, f0_max=2218, type_rmvpe="rmvpe", extended=False):
        # f0_min/f0_max здесь не используются: диапазон применяется к результату
        # в pipeline.py (только инференс), чтобы данные обучения не менялись.
        if type_rmvpe == "rmvpe":
            model = self.model_og
        elif type_rmvpe == "hpa-rmvpe":
            model = self.model_hpa
        else:
            raise ValueError(
                f"Недопустимое значение: {type_rmvpe!r}. Доступные варианты: 'rmvpe' и 'hpa-rmvpe'."
            )

        f0, conf_nat = model.infer_from_audio_conf(audio, thred=0.03)
        if extended:
            f0_ext, conf_ext = self._f0_octave_down(model, audio)
            f0 = self._merge_extended(f0, f0_ext, conf_nat, conf_ext, f0_max)
        return f0

    def _f0_octave_down(self, model, audio, thred=0.03):
        """Второй проход: растяжение сигнала x2 (питч -12 ст, sr тот же),
        поэтому анализатор видит f/2; результат x2 = исходная f + уверенность.
        Кадры и conf возвращаются на родную сетку 100 fps."""
        y = resample_poly(audio, 2, 1)
        f0_half, conf_half = model.infer_from_audio_conf(y, thred=thred)
        f0_half = f0_half * 2.0
        n_full = 1 + len(audio) // 160
        if n_full < 2:
            return np.zeros(n_full, np.float32), np.zeros(n_full, np.float32)
        t_half = np.linspace(0.0, 1.0, len(f0_half))
        t_full = np.linspace(0.0, 1.0, n_full)
        conf_full = np.interp(t_full, t_half, conf_half).astype(np.float32)
        voiced_idx = np.flatnonzero(f0_half > 0)
        if voiced_idx.size == 0:
            return np.zeros(n_full, np.float32), conf_full
        mask = np.interp(t_full, t_half, (f0_half > 0).astype(float)) >= 0.5
        log_f0 = np.interp(t_full, t_half[voiced_idx], np.log2(f0_half[voiced_idx]))
        f0_full = np.where(mask, 2.0**log_f0, 0.0).astype(np.float32)
        conf_full[f0_full == 0] = 0.0
        return f0_full, conf_full

    @staticmethod
    def _merge_extended(f0_nat, f0_ext, conf_nat, conf_ext, f0_max=2218):
        """Трекинг с инерцией: на каждом кадре кандидаты = {native, ext},
        score = уверенность - штраф за скачок относительно предыдущего кадра.
        Октавные провалы/выбросы проигрывают инерции, полутоновые переходы
        между нотами — почти бесплатны. Никаких покадровых масок."""
        n = min(len(f0_nat), len(f0_ext))
        nat, ext = f0_nat[:n], f0_ext[:n]
        cn, ce = conf_nat[:n], conf_ext[:n]
        ext_ok = (ext > 0) & (ext <= f0_max)
        lam = 0.5  # штраф за октаву скачка от предыдущего кадра
        floor = 0.25  # минимальная уверенность кандидата
        out = np.zeros(n, np.float32)
        prev = 0.0
        for k in range(n):
            best, bs = 0.0, -1e9
            if nat[k] > 0 and cn[k] >= floor:
                s = cn[k] - (lam * abs(np.log2(nat[k] / prev)) if prev > 0 else 0.0)
                if s > bs:
                    best, bs = nat[k], s
            if ext_ok[k] and ce[k] >= floor:
                s = ce[k] - (lam * abs(np.log2(ext[k] / prev)) if prev > 0 else 0.0)
                if s > bs:
                    best, bs = ext[k], s
            out[k] = best
            if best > 0:
                prev = best
        return out


class CREPEF0:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size

    def get_f0(self, audio, f0_min=61, f0_max=2218, p_len=None, model="full"):
        if p_len is None:
            p_len = audio.shape[0] // self.hop_size

        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        f0, pd = torchcrepe.predict(
            audio=audio.float().to(self.device).unsqueeze(dim=0),
            sample_rate=self.sample_rate,
            hop_length=self.hop_size,
            fmin=f0_min,
            fmax=f0_max,
            model=model,
            decoder=torchcrepe.decode.weighted_argmax,
            return_periodicity=True,
            batch_size=512,
            device=self.device,
            pad=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.median(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()

        return f0


class FCPEF0:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_bundled_infer_model(self.device)

    def get_f0(
        self, audio, f0_min=61, f0_max=2218, p_len=None, use_interpolation=False
    ):
        if p_len is None:
            p_len = audio.shape[0] // self.hop_size

        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)

        f0 = (
            self.model.infer(
                wav=audio.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=0.006,
                f0_min=f0_min,
                f0_max=f0_max,
                interp_uv=use_interpolation,
                output_interp_target_length=p_len,
                return_uv=False,
                test_time_augmentation=True,
                tta_uv_penalty=12.0,
                tta_key_shifts=[-24, -12, 0, 12, 24],
                tta_use_origin_uv=True,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0
