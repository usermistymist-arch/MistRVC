"""Запуск долгих задач (шаги обучения) с живым журналом и кнопкой остановки.

Скрипты пайплайна читают аргументы из `sys.argv` на уровне модуля,
поэтому запускаются отдельным процессом — как в официальном RVC WebUI.

Вывод процесса читает фоновый поток: он же печатает эхо в консоль (ячейку
Colab) и копит последние строки журнала. Отдача журнала в интерфейс —
отдельный генератор-наблюдатель, поэтому обрыв соединения с браузером
прекращает только показ в UI, но не задачу, не эхо и не чтение вывода.
"""

import io
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque

import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAIL_LINES = 200  # Сколько последних строк журнала показывать
POLL_INTERVAL = 0.3  # Как часто наблюдатель отдаёт снимок журнала в интерфейс

_lock = threading.Lock()
_process = None
_process_group_id = None
_stop_requested = False
_job_active = False
_job = None  # Текущая или последняя завершённая задача (_Job)


class _Job:
    """Состояние одной задачи: журнал, живой tqdm-бар и признак завершения.

    Живёт в модуле, а не в генераторе обработчика: переоткрытие страницы
    подключается к тому же состоянию и видит накопленный журнал.
    """

    def __init__(self, title: str, prefix: str = ""):
        self.lock = threading.Lock()
        self.lines = deque(maxlen=TAIL_LINES)
        self.lines.append(f"> {title}")
        if prefix:
            self.lines.extend(prefix.split("\n"))
        self.live = ""
        self.finished = False

    def append(self, text: str) -> None:
        with self.lock:
            self.lines.append(text)

    def set_live(self, text: str) -> None:
        with self.lock:
            self.live = text

    def snapshot(self) -> str:
        with self.lock:
            return "\n".join([*self.lines, self.live] if self.live else self.lines)

    def finish(self, message: str) -> None:
        with self.lock:
            if message:
                self.lines.append(message)
            self.live = ""
            self.finished = True


def busy() -> bool:
    with _lock:
        if _job_active:
            return True
        proc = _process
        group_id = _process_group_id
    if proc is None:
        return False
    if proc.poll() is None:
        return True
    if os.name != "nt" and group_id is not None:
        try:
            os.killpg(group_id, 0)
            return True
        except (OSError, ProcessLookupError):
            pass
    return False


def command(*args) -> list:
    """Команда запуска модуля проекта тем же интерпретатором, что и интерфейс."""
    return [sys.executable or "python", "-m", *[str(arg) for arg in args]]


def _kill_process_tree(proc: subprocess.Popen, sig: int, process_group_id=None) -> None:
    """Останавливает процесс и его потомков, даже если лидер уже завершился."""
    if os.name == "nt":
        # Popen.terminate() завершает только родителя; taskkill /T нужен,
        # чтобы вместе с ним остановить multiprocessing/DataLoader workers.
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        # Не вызывать os.getpgid(proc.pid) здесь: после выхода лидера это
        # невозможно, хотя его дочерние процессы ещё могут оставаться живы.
        os.killpg(process_group_id if process_group_id is not None else proc.pid, sig)
    except (OSError, ProcessLookupError):
        pass


def request_stop() -> str:
    """Обработчик кнопки «Остановить». Вызывается с queue=False."""
    global _stop_requested
    with _lock:
        _stop_requested = True
        proc = _process
        process_group_id = _process_group_id
        active = _job_active
    if proc is None:
        if active:
            return "Остановка запрошена: задача завершится перед следующей командой."
        return "Нет запущенного процесса."

    _kill_process_tree(proc, signal.SIGTERM, process_group_id)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    # Не считаем задачу остановленной лишь потому, что вышел родительский
    # Python-процесс: воркеры DataLoader могут пережить его и держать ОЗУ.
    _kill_process_tree(
        proc, signal.SIGKILL if os.name != "nt" else signal.SIGTERM, process_group_id
    )
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return "Не удалось остановить процесс — проверьте оставшиеся дочерние процессы."
    return "Процесс и его дочерние процессы остановлены."


def _release_inference_cache() -> None:
    """Освобождает кэш инференса: обучающему процессу нужна вся память GPU.

    Импорт ленивый — `rvc.inference.infer` тянет torch, а интерфейс запускается без него.
    """
    try:
        from rvc.inference.infer import release_pipeline
    except Exception:  # noqa: BLE001 — без torch/зависимостей выгружать нечего
        return
    released = release_pipeline()
    if released:
        print(
            f"[i] Модель '{released}' выгружена из памяти перед запуском задачи.",
            flush=True,
        )


def _spawn(cmd: list) -> subprocess.Popen:
    global _process, _process_group_id
    _release_inference_cache()
    kwargs = {
        "cwd": PROJECT_ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with _lock:
        # Команда собрана из констант, не из пользовательского ввода.
        _process = subprocess.Popen(cmd, **kwargs)  # noqa: S603
        _process_group_id = _process.pid if os.name != "nt" else None
        # Текстовая обёртка без трансляции \r: это живой прогресс tqdm, а не конец строки.
        _process.stdout = io.TextIOWrapper(
            _process.stdout, encoding="utf-8", errors="replace", newline=""
        )
        return _process


def _run_commands(job: _Job, commands: list) -> None:
    """Тело задачи в фоновом потоке: команды по очереди, чтение вывода, эхо.

    Не зависит от соединения с браузером: даже если все наблюдатели Gradio
    отключились, поток продолжает читать вывод (иначе pipe переполнился бы
    и обучающий процесс завис на записи) и печатать его в консоль.
    """
    global _process, _process_group_id, _job_active
    try:
        for cmd in commands:
            with _lock:
                if _stop_requested:
                    job.finish("[!] Остановлено пользователем.")
                    return
            # Аргументы могли дописаться в cmd после command(): приводим к str,
            # иначе join и Popen падают на числах.
            cmd = [str(part) for part in cmd]
            job.append(f"$ {' '.join(cmd)}")
            proc = _spawn(cmd)
            buffer = ""
            live = ""
            while True:
                # Читаем посимвольно: tqdm минутами шлёт только \r без \n,
                # построчное чтение на это время глухо виснет.
                chunk = proc.stdout.read(1)
                if chunk:
                    if chunk in "\r\n":
                        text = buffer.strip()
                        buffer = ""
                        if chunk == "\n":
                            # Colab/Jupyter перерисовывают \r в выдаче ячейки,
                            # поэтому живой бар печатается и там: завершённую
                            # строку выводим поверх бара через \r + перевод строки.
                            console_text = text or live
                            if console_text:
                                job.append(console_text)
                                if live:
                                    print("\r" + console_text, flush=True)
                                else:
                                    print(console_text, flush=True)
                            live = ""
                            job.set_live("")
                        elif text:  # \r — живое обновление tqdm-бара
                            live = text
                            job.set_live(live)
                            print("\r" + live, end="", flush=True)
                    else:
                        buffer += chunk
                elif proc.poll() is not None:
                    break
                else:
                    time.sleep(0.05)
            tail = buffer.strip()
            if tail:
                print(tail, flush=True)
                job.append(tail)
            if live:
                print("\r" + live, flush=True)
                job.append(live)
                job.set_live("")
            proc.wait()
            with _lock:
                stopped = _stop_requested
            if stopped:
                job.finish("[!] Остановлено пользователем.")
                return
            if proc.returncode != 0:
                job.finish(f"[X] Ошибка (код {proc.returncode}). Смотрите журнал выше.")
                return
        job.finish("Готово.")
    except Exception as error:  # noqa: BLE001 — сбой задачи пишем в журнал, не роняем UI
        message = f"[X] Внутренняя ошибка задачи: {error}"
        print(message, flush=True)
        job.finish(message)
    finally:
        # Сбрасываем только по реальному завершению задачи, а не по обрыву
        # соединения: иначе osиротевший процесс стал бы неуправляемым.
        with _lock:
            _job_active = False
            _process = None
            _process_group_id = None


def start_job(title: str, commands: list, prefix: str = "") -> _Job:
    """Запускает задачу в фоновом потоке и сразу возвращает её состояние."""
    global _job, _stop_requested, _job_active
    if commands and isinstance(commands[0], str):
        commands = [commands]
    with _lock:
        if _job_active:
            raise gr.Error(
                "Другая задача ещё выполняется — дождитесь её или остановите."
            )
        _stop_requested = False
        _job_active = True
        _job = _Job(title, prefix)
        job = _job
    threading.Thread(
        target=_run_commands, args=(job, commands), daemon=True, name="mistrvc-job"
    ).start()
    return job


def watch_job(job: _Job | None = None):
    """Наблюдатель: отдаёт снимок журнала каждые 0.3 с до завершения задачи.

    Без аргумента подключается к текущей/последней задаче. Обрыв соединения
    закрывает только этот генератор; сама задача продолжает работать.
    """
    if job is None:
        with _lock:
            job = _job
    if job is None:
        yield ""
        return
    while True:
        yield job.snapshot()
        if job.finished:
            return
        time.sleep(POLL_INTERVAL)


def run_job(title: str, commands: list, prefix: str = ""):
    """Генератор для кнопки запуска: стартует задачу и отдаёт её журнал.

    Совместим со старым вызовом `yield from run_job(...)`, но выполнение
    теперь живёт в фоновом потоке (start_job), а не в этом генераторе.
    """
    job = start_job(title, commands, prefix)
    yield from watch_job(job)
