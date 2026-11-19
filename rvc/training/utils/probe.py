"""Фиксированный зонд оценки: выбор сегментов и списки файлов для обучения.

Зонд — небольшой неизменный набор сегментов для сравнения эпох и моделей.
Мьюты в зонд не берутся никогда: тишина воспроизводится почти идеально и
завышает метрики. Соседние окна нарезки перекрываются на 0.3 с, поэтому
в зонд берётся сплошной блок, а соседи блока при изъятии тоже уходят.
"""

import hashlib
import json
import os
import re

MODES = ("off", "fixed", "holdout")
# Имена для журнала; в CLI и probe.json остаются коды режимов.
MODE_NAMES = {
    "off": "Случайный",
    "fixed": "Обучающие данные",
    "holdout": "Исключённые данные",
}
_SHARE = 0.08
_MIN_PROBE = 4
_MAX_PROBE = 24
_MIN_SEGMENTS = 40


def _wav(line):
    return line.split("|")[0]


def _is_mute(line):
    return os.path.basename(_wav(line)).startswith("mute")


def _key(line):
    """(спикер, исходный файл, номер окна) из имени `{idx0}_{idx1}.wav`."""
    parts = line.split("|")
    stem = os.path.splitext(os.path.basename(parts[0]))[0]
    found = re.search(r"(\d+)_(\d+)$", stem)
    if found:
        return (parts[-1], int(found.group(1)), int(found.group(2)))
    return (parts[-1], stem, 0)


def _choose(candidates, size):
    """Сплошной блок из `size` сегментов и его соседи (отступ).

    Блок берётся из середины самого большого исходника: так в зонд не
    попадают обрезанные края записи, а соседи по idx1 не остаются в обучении.
    """
    groups = {}
    for line in candidates:
        sid, src, _ = _key(line)
        groups.setdefault((sid, src), []).append(line)
    for lines in groups.values():
        lines.sort(key=_key)

    chosen, margin = [], []
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        lines = groups[key]
        take = min(size - len(chosen), len(lines))
        if take <= 0:
            break
        start = (len(lines) - take) // 2
        chosen += lines[start : start + take]
        margin += lines[max(0, start - 1) : start]
        margin += lines[start + take : start + take + 1]
    return chosen, margin


def _digest(lines):
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def locked_probe(exp_dir):
    """Зафиксированный набор модели: (режим, число сегментов) или None.

    None — набор ещё можно выбрать: probe.json нет либо filelist.txt
    изменился, и при следующем запуске набор будет пересобран.
    """
    data_dir = os.path.join(exp_dir, "data")
    probe_path = os.path.join(data_dir, "probe.json")
    filelist = os.path.join(data_dir, "filelist.txt")
    if not os.path.isfile(probe_path) or not os.path.isfile(filelist):
        return None
    try:
        with open(probe_path, encoding="utf-8") as file:
            saved = json.load(file)
        with open(filelist, encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
        if saved.get("filelist") != _digest(lines):
            return None
        mode = saved.get("mode")
        if mode not in MODES:
            return None
        return mode, int(saved.get("size") or len(saved.get("eval", [])))
    except (OSError, ValueError, TypeError):
        return None


def setup_probe(exp_dir, mode="fixed", size=0, batch_size=16):
    """Готовит зонд и списки; возвращает (train-список, eval-список или None, описание).

    Выбор сохраняется в data/probe.json и переиспользуется при продолжении
    обучения: менять зонд посреди пути нельзя, как и сид.
    """
    data_dir = os.path.join(exp_dir, "data")
    filelist = os.path.join(data_dir, "filelist.txt")
    if mode == "off":
        probe_path = os.path.join(data_dir, "probe.json")
        saved_mode = None
        if os.path.isfile(probe_path):
            try:
                with open(probe_path, encoding="utf-8") as file:
                    saved_mode = json.load(file).get("mode")
            except (OSError, ValueError):
                saved_mode = None
        if saved_mode != "holdout":
            return filelist, None, MODE_NAMES["off"]
        # Изъятые сегменты нельзя вернуть в обучение посреди пути
        print(
            f"Проверочный набор зафиксирован как «{MODE_NAMES['holdout']}» — "
            f"запрос «{MODE_NAMES['off']}» проигнорирован.",
            flush=True,
        )
        mode = "holdout"
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим зонда: {mode}")

    with open(filelist, encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    candidates = [line for line in lines if not _is_mute(line)]
    if not candidates:
        return filelist, None, "случайный (нет сегментов)"

    digest = _digest(lines)
    probe_path = os.path.join(data_dir, "probe.json")
    saved = None
    if os.path.isfile(probe_path):
        try:
            with open(probe_path, encoding="utf-8") as file:
                saved = json.load(file)
        except (OSError, ValueError):
            saved = None

    if saved and saved.get("filelist") == digest:
        if saved.get("mode") != mode:
            was = MODE_NAMES.get(saved["mode"], saved["mode"])
            asked = MODE_NAMES.get(mode, mode)
            print(
                f"Проверочный набор зафиксирован как «{was}» — запрос «{asked}» "
                "проигнорирован: менять набор посреди обучения нельзя.",
                flush=True,
            )
        mode = saved["mode"]
        eval_wavs = set(saved["eval"])
        margin_wavs = set(saved.get("margin", []))
        saved_size = int(saved.get("size") or len(eval_wavs))
        if size and int(size) != saved_size:
            print(
                f"Проверочный набор зафиксирован: {saved_size} сегм. — "
                f"запрос {int(size)} проигнорирован: менять набор посреди обучения нельзя.",
                flush=True,
            )
    else:
        if saved:
            print("• filelist.txt изменился — зонд пересобран.", flush=True)
        want = max(0, int(size)) or max(
            _MIN_PROBE, min(_MAX_PROBE, round(_SHARE * len(candidates)))
        )
        want = min(want, len(candidates))
        chosen, margin = _choose(candidates, want)
        if mode == "holdout":
            rest = len(candidates) - len(chosen) - len(margin)
            if len(candidates) < _MIN_SEGMENTS or rest < 2 * batch_size:
                print(
                    f"Набор исключённых из датасета данных недоступен: "
                    f"{len(candidates)} сегментов, "
                    f"после изъятия осталось бы {rest}. "
                    "Выбран режим 'Обучающие данные'.",
                    flush=True,
                )
                mode = "fixed"
        eval_wavs = {_wav(line) for line in chosen}
        margin_wavs = {_wav(line) for line in margin}
        with open(probe_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "filelist": digest,
                    "mode": mode,
                    "size": len(eval_wavs),
                    "eval": sorted(eval_wavs),
                    "margin": sorted(margin_wavs),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    removed = (eval_wavs | margin_wavs) if mode == "holdout" else set()
    train_lines = [line for line in lines if _wav(line) not in removed]
    eval_lines = [line for line in lines if _wav(line) in eval_wavs]
    train_path = os.path.join(data_dir, "train_filelist.txt")
    eval_path = os.path.join(data_dir, "eval_filelist.txt")
    with open(train_path, "w", encoding="utf-8") as file:
        file.write("\n".join(train_lines))
    with open(eval_path, "w", encoding="utf-8") as file:
        file.write("\n".join(eval_lines))

    note = f"{MODE_NAMES[mode]}, {len(eval_lines)} сегм."
    if mode == "holdout":
        note += f"; обучение: {len(train_lines)} из {len(lines)} строк"
    return train_path, eval_path, note
