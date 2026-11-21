import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import Normalize


def plot_spectrogram_to_numpy(
    spectrogram,
    figsize=(10, 4),
    cmap="viridis",
    sample_rate=None,
    hop_length=None,
    mel_fmin=0.0,
    mel_fmax=None,
):
    """Визуализация Mel-спектрограммы.

    spectrogram — натуральный логарифм мел-амплитуды (spec_to_mel_torch).
    Если переданы sample_rate и hop_length, оси подписываются в Гц и секундах;
    иначе — старая разметка в каналах/кадрах.
    """
    # ln(amplitude) -> dB: те же цвета, что при Normalize(-10, 0) на ln-шкале
    db_per_ln = 20.0 / math.log(10.0)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        spectrogram * db_per_ln,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=Normalize(vmin=-10.0 * db_per_ln, vmax=0.0),
    )
    cbar = fig.colorbar(im, ax=ax, format="%+.0f dB")
    cbar.set_label("Уровень, дБ")

    n_mels, n_frames = spectrogram.shape
    if sample_rate and hop_length:
        from librosa import mel_frequencies

        # Тот же slaney-мел, что и librosa.filters.mel в spec_to_mel_torch
        freqs = mel_frequencies(
            n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax or sample_rate / 2
        )
        ticks, labels = [], []
        for hz in (0, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000):
            if hz > (mel_fmax or sample_rate / 2):
                continue
            ch = int(np.argmin(np.abs(freqs - hz)))
            if ch in ticks:  # на 32k низкие тики могут совпасть каналом
                continue
            ticks.append(ch)
            labels.append(f"{hz // 1000}k" if hz >= 1000 else str(hz))
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Частота, Гц (mel-шкала)")

        step = max(1, round(0.25 * sample_rate / hop_length))  # тик ~0.25 с
        xs = np.arange(0, n_frames, step)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{i * hop_length / sample_rate:.2f}" for i in xs])
        ax.set_xlabel("Время, с")
    else:
        ax.set_xlabel("Кадры")
        ax.set_ylabel("Частотные каналы")
    plt.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf, dtype=np.uint8)
    plt.close(fig)
    return data


def mel_spectrogram_similarity(y_hat_mel, y_mel):
    """Сходство между сгенерированной и реальной мел-спектрограммами"""
    device = y_hat_mel.device
    y_mel = y_mel.to(device)

    if y_hat_mel.shape != y_mel.shape:
        trimmed_shape = tuple(
            min(dim_a, dim_b) for dim_a, dim_b in zip(y_hat_mel.shape, y_mel.shape)
        )
        y_hat_mel = y_hat_mel[..., : trimmed_shape[-1]]
        y_mel = y_mel[..., : trimmed_shape[-1]]

    loss_mel = F.l1_loss(y_hat_mel, y_mel)
    mel_spec_similarity = 100.0 - (loss_mel * 100.0)
    return mel_spec_similarity.clamp(0.0, 100.0)


def _estimate_f0_autocorr(
    wav, sample_rate, hop_length, f0_min=50.0, f0_max=1100.0, frame_ms=40.0
):
    """Быстрая оценка F0 через автокорреляцию для онлайн-метрики."""
    wav = wav.squeeze(1) if wav.dim() == 3 else wav
    original_device = wav.device

    # FFT всегда на CPU: на CUDA cuFFT выделяет буферы мимо аллокатора PyTorch
    # и при занятой VRAM падает с CUFFT_INTERNAL_ERROR; на MPS torch.fft нестабилен.
    # Объём крошечный (сотни кадров) — на CPU миллисекунды; результат вернём на исходное устройство.
    wav = wav.detach().float().cpu()

    frame_length = int(sample_rate * frame_ms / 1000.0)
    frame_length = max(frame_length, int(sample_rate / f0_min) * 2, hop_length * 2)
    frame_length = min(frame_length, wav.size(-1))
    if frame_length <= 4 or wav.size(-1) < frame_length:
        return wav.new_zeros((wav.size(0), 1)).to(original_device)

    # Центрируем кадры: кадр k должен описывать момент k*hop, как pitchf датасета
    # (RMVPE считает мел с center=True). Без дополнения центр смещён на +20 мс.
    # Нулями, а не reflect: краевые кадры получат низкий rms и будут отброшены.
    pad = frame_length // 2
    wav = F.pad(wav, (pad, pad))

    frames = wav.unfold(-1, frame_length, hop_length)
    frames = frames - frames.mean(dim=-1, keepdim=True)
    window = torch.hann_window(frame_length, device=wav.device, dtype=wav.dtype)
    frames = frames * window

    rms = frames.pow(2).mean(dim=-1).sqrt()
    n_fft = 1 << int(math.ceil(math.log2(frame_length * 2 - 1)))
    spectrum = torch.fft.rfft(frames, n=n_fft)
    autocorr = torch.fft.irfft(spectrum * spectrum.conj(), n=n_fft)[..., :frame_length]

    min_lag = max(1, int(sample_rate / f0_max))
    max_lag = min(frame_length - 1, int(sample_rate / f0_min))
    if max_lag <= min_lag:
        return wav.new_zeros(rms.shape).to(original_device)

    peak_region = autocorr[..., min_lag : max_lag + 1]
    peak_values, peak_indices = peak_region.max(dim=-1)

    # Целый лаг квантует частоту с шагом f0^2/sr: на 1100 Гц это ~40 центов,
    # на 500 Гц ~18. Парабола по трём отсчётам даёт дробный лаг и убирает
    # этот шумовой пол измерителя.
    idx = peak_indices.unsqueeze(-1)
    last = peak_region.size(-1) - 1
    left = peak_region.gather(-1, (idx - 1).clamp_min(0)).squeeze(-1)
    right = peak_region.gather(-1, (idx + 1).clamp_max(last)).squeeze(-1)
    # В вершине знаменатель неположительный; clamp_max страхует от деления на 0
    denom = (left - 2.0 * peak_values + right).clamp_max(-1e-12)
    delta = (0.5 * (left - right) / denom).clamp(-0.5, 0.5)
    # На краях диапазона соседей нет — интерполировать нечего
    inner = (peak_indices > 0) & (peak_indices < last)
    delta = torch.where(inner, delta, torch.zeros_like(delta))

    lag = (peak_indices + min_lag).float() + delta
    f0 = sample_rate / lag.clamp_min(1.0)

    periodicity = peak_values / autocorr[..., 0].clamp_min(1e-8)
    voiced = (rms > 0.003) & (periodicity > 0.25)
    return torch.where(voiced, f0, torch.zeros_like(f0)).to(original_device)


def _slice_pitch(pitchf, ids_slice, segment_frames, device):
    """Берет тот же срез pitchf, что и аудио-сегмент для y_hat."""
    target = pitchf.unsqueeze(1)
    target_slices = []
    for idx, start in enumerate(ids_slice.tolist()):
        target_slice = target[idx : idx + 1, :, start : start + segment_frames]
        if target_slice.size(-1) < segment_frames:
            target_slice = F.pad(
                target_slice, (0, segment_frames - target_slice.size(-1))
            )
        target_slices.append(target_slice)
    return torch.cat(target_slices, dim=0).squeeze(1).to(device).float()


def f0_cents_errors(y_hat, pitchf, ids_slice, sample_rate, hop_length, segment_frames):
    """Ошибки F0 в центах по voiced-кадрам: 1D тензор или None, если их нет."""
    fake_f0 = _estimate_f0_autocorr(y_hat, sample_rate, hop_length)
    target_f0 = _slice_pitch(pitchf, ids_slice, segment_frames, fake_f0.device)

    frames = min(fake_f0.size(-1), target_f0.size(-1))
    fake_f0 = fake_f0[..., :frames]
    target_f0 = target_f0[..., :frames]

    voiced_mask = (fake_f0 > 0.0) & (target_f0 > 0.0)
    if not bool(voiced_mask.any()):
        return None

    cents_error = 1200.0 * torch.abs(
        torch.log2(
            fake_f0[voiced_mask].clamp_min(1e-5)
            / target_f0[voiced_mask].clamp_min(1e-5)
        ),
    )
    return cents_error.clamp(max=2400.0)


def f0_error_cents(y_hat, pitchf, ids_slice, sample_rate, hop_length, segment_frames):
    """Ошибка F0 в центах: (среднее, медиана, число кадров).

    Меньше = лучше. 100 cents = 1 полутон. Считается только на voiced кадрах,
    где и y_hat, и target pitchf имеют ненулевой F0.

    Медиана устойчива к единичным октавным промахам: если среднее заметно
    больше медианы, метрику тянут выбросы, а не общий строй. Если сравнивать
    нечего, возвращаем None — в графике будет пропуск, а не «катастрофа».
    """
    errors = f0_cents_errors(
        y_hat, pitchf, ids_slice, sample_rate, hop_length, segment_frames
    )
    if errors is None:
        return None, None, 0
    return errors.mean(), errors.median(), int(errors.numel())
