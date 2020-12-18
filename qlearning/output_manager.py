import os
import shutil
import sys

from threading import Thread, Event
from typing import Optional
from pathlib import Path
from time import strftime

import torch
from torch_spread import NetworkManager, mp_ctx
from tqdm import tqdm

from qlearning.parameters import Parameters


class Logger:
    def __init__(self, logfile: str):
        """ A simple logger class that redirects an input stream to both the console and an file.

        Parameters
        ----------
        logfile: str
            Output file of any print commands.
        """
        self.terminal = sys.stdout
        self.log = open(logfile, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class OutputManager:
    def __init__(self, hparams: Parameters, reloaded: bool = False):
        """ A small helper class that collects a bunch of methods for saving and displaying results for q-learning.

        Parameters
        ----------
        hparams: Parameters
            List of parameters. These will be pickled and saved along with the model.
        reloaded: bool
            Whether or not this output manager is being reloaded from the save directory.
            Should only be used by the script.
        """
        self.hparams = hparams
        self.reloaded = reloaded
        self.output_directory = hparams.output_directory
        self.start_time = None

        if hparams.append_time_to_output:
            self.start_time = strftime('%y-%m-%d_%H-%M-%S')
            self.output_directory = f"{self.output_directory}_{self.start_time}"

        self.logfile = Path(f"{self.output_directory}/results.txt")
        self.hparams_file = Path(f"{self.output_directory}/hparams.json")
        self.weights_file = Path(f"{self.output_directory}/weights/current.torch")
        self.backup_weights_template = f"{self.output_directory}/weights/epoch_{'{}'}.torch"

        os.makedirs(self.weights_file.parent, exist_ok=True)

    def print_parameters(self):
        self.print_with_border("Configuration")

        for key, val in sorted(self.hparams.__dict__.items()):
            self.print_value(key, val)
        print()

        if self.reloaded:
            print("Note")
            print("-" * 60)
            self.print_value("Loading existing parameters", self.hparams_file, end="\n\n")

    @staticmethod
    def print_with_border(string: str, before: str = "-", after: str = "=", length: int = 60):
        if before is not None:
            print(before * length)

        print(string)

        if after is not None:
            print(after * length)

    @staticmethod
    def print_value(name, value, start: str = "", end: str = "\n"):
        print(start, end="")
        if isinstance(value, float):
            print(f"{name:30} : {value:.3f}", end=end)
        else:
            print(f"{name:30} : {value}", end=end)

    @property
    def logger(self) -> Logger:
        return Logger(self.logfile)

    @property
    def weights_exist(self) -> bool:
        return self.weights_file.exists()

    @property
    def hparams_exist(self) -> bool:
        return self.hparams_file.exists()

    def load_weights(self, manager: NetworkManager) -> bool:
        initial_iteration = True
        if self.weights_exist:
            self.print_value("Loading network from", self.weights_file)
            manager.load_state_dict(torch.load(self.weights_file))
            initial_iteration = False

        return initial_iteration

    def save_weights(self, manager: NetworkManager, iteration: int):
        self.print_value("\nSaving network weights to", self.weights_file)
        torch.save(manager.state_dict, self.weights_file)
        if (iteration % self.hparams.weights_save_iterations) == 0:
            shutil.copy(self.weights_file, self.backup_weights_file(iteration))

    def backup_weights_file(self, iteration: int) -> Path:
        return Path(self.backup_weights_template.format(iteration))

    def save_hparams(self):
        self.hparams.save(self.hparams_file)


class AsyncStatusBar(Thread):
    UPDATE = b"UPDATE"
    BEGIN = b"BEGIN"
    KILL = b"KILL"
    END = b"END"

    def __init__(self, enable: bool = True):
        """ Create an asynchronous thread for managing a status bar that is shared by many workers.

        Parameters
        ----------
        enable: bool
            Boolean flag to enable or disable the status bar display. Simplifies code slightly.
        """
        super(AsyncStatusBar, self).__init__()
        self.request_queue = mp_ctx.Queue()
        self.finished = Event()
        self.started = False
        self.enable = enable

        self.desc: Optional[str] = None
        self.total: Optional[int] = None

    def begin(self, desc: Optional[str] = None, total: Optional[int] = None):
        if self.enable:
            self.request_queue.put((self.BEGIN, desc, total))

    def end(self):
        if self.enable:
            self.request_queue.put((self.END, ))
            self.finished.wait()
            print()

    def update(self, amount: int):
        if self.enable:
            self.request_queue.put((self.UPDATE, amount))

    def kill(self, timeout: Optional[float] = None):
        if self.enable:
            self.request_queue.put((self.KILL, ))
            self.join(timeout)

    @property
    def updater(self):
        return StatusBarUpdater(self)

    def run(self) -> None:
        progress_bar: Optional[tqdm] = None
        self.started = True

        while True:
            command, *data = self.request_queue.get()

            if command == self.KILL:
                return

            elif command == self.BEGIN:
                progress_bar = tqdm(desc=data[0], total=data[1])
                self.finished.clear()
                continue

            elif progress_bar is None:
                continue

            elif command == self.END:
                progress_bar.close()
                progress_bar = None
                self.finished.set()
                continue

            else:
                progress_bar.update(data[0])

    def __call__(self, desc: Optional[str] = None, total: Optional[int] = None):
        self.desc = desc
        self.total = total

        return self

    def __enter__(self):
        if not self.enable:
            return self

        if not self.started:
            self.start()

        self.begin(self.desc, self.total)
        self.desc = None
        self.total = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()


class StatusBarUpdater:
    def __init__(self, progress_bar: AsyncStatusBar):
        """ Small container class that is passed to Process-based workers.

        Notes
        -----
        This is necessary because the main class cannot be pickled.
        """
        self.request_queue = progress_bar.request_queue
        self.enable = progress_bar.enable

    def update(self, amount: int):
        if self.enable:
            self.request_queue.put((AsyncStatusBar.UPDATE, amount))
