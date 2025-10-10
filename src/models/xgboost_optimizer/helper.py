from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ==================== TRAINING CONTROLLER (keyboard listener) ====================

class TrainingController:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.pause_lock = threading.Lock()
        self.listener_thread = None
        self.running = False

    def start_listener(self):
        if self.listener_thread is not None and self.listener_thread.is_alive():
            return
        print("\n" + "=" * 70)
        print("  TRAINING CONTROLS ACTIVE")
        print("  Type 'p' + ENTER to PAUSE | 'r' to RESUME | 'q' to QUIT after current op")
        print("=" * 70 + "\n")
        self.running = True
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()

    def _keyboard_listener(self):
        if os.name == "nt":
            try:
                import msvcrt  # type: ignore
            except ImportError:
                while self.running:
                    time.sleep(0.2)
                return
            buf = []
            while self.running:
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ("\r", "\n"):
                            cmd = "".join(buf).strip().lower()
                            buf.clear()
                            self._dispatch(cmd)
                        elif ch == "\x03":
                            break
                        else:
                            buf.append(ch)
                    else:
                        time.sleep(0.1)
                except Exception:
                    break
        else:
            import select
            line = []
            while self.running:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch in ("\n", "\r"):
                            cmd = "".join(line).strip().lower()
                            line.clear()
                            self._dispatch(cmd)
                        else:
                            line.append(ch)
                except Exception:
                    break

    def _dispatch(self, cmd: str):
        if cmd == "p":
            with self.pause_lock:
                if not self.paused:
                    self.paused = True
                    print("\n=== ⏸️  TRAINING PAUSED — 'r' to resume, 'q' to quit ===\n")
        elif cmd == "r":
            with self.pause_lock:
                if self.paused:
                    self.paused = False
                    print("\n=== ▶️  TRAINING RESUMED ===\n")
        elif cmd == "q":
            with self.pause_lock:
                if not self.should_stop:
                    self.should_stop = True
                    print("\n=== 🛑 QUIT REQUESTED — will stop after current operation ===\n")
        elif cmd:
            print(f"[Unknown '{cmd}'] Valid: p, r, q")

    def check_pause(self):
        while True:
            with self.pause_lock:
                if self.should_stop:
                    raise KeyboardInterrupt("Training stopped by user")
                if not self.paused:
                    break
            time.sleep(0.3)

    def stop(self):
        self.running = False
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=1.0)


# ==================== PLOTTING (graph creation) ====================

def annotated_trial_plot(
    evals_result: Dict[str, Dict[str, list]],
    title: str,
    best_idx: int,
    gap: float,
    save_path_png: Optional[Path],
    show_plots: bool,
    save_plots_as_png: bool,
):
    """
    Render a 2-panel chart: (logloss) and (accuracy), annotate best iteration & gap.
    This function is backend-agnostic; caller can set Agg if desired before calling.
    """
    if not evals_result:
        return

    tr_ll = evals_result.get("validation_0", {}).get("logloss", None)
    va_ll = evals_result.get("validation_1", {}).get("logloss", None)
    tr_er = evals_result.get("validation_0", {}).get("error", None)
    va_er = evals_result.get("validation_1", {}).get("error", None)
    if tr_ll is None or va_ll is None or tr_er is None or va_er is None:
        return

    iters = np.arange(1, len(tr_ll) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Loss
    axes[0].plot(iters, tr_ll, linewidth=2, label="Train Logloss")
    axes[0].plot(iters, va_ll, linewidth=2, label="Val Logloss")
    axes[0].axvline(best_idx + 1, linestyle="--", linewidth=1)
    axes[0].annotate(f"best@{best_idx+1}\ngap={gap:.3f}",
                     xy=(best_idx + 1, va_ll[best_idx]),
                     xytext=(best_idx + 1, max(va_ll) * 0.9),
                     arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    # Accuracy
    axes[1].plot(iters, 1.0 - np.asarray(tr_er), linewidth=2, label="Train Accuracy")
    axes[1].plot(iters, 1.0 - np.asarray(va_er), linewidth=2, label="Val Accuracy")
    axes[1].axvline(best_idx + 1, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Boosting Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_plots_as_png and save_path_png is not None:
        fig.savefig(save_path_png, dpi=120)

    if show_plots:
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)


def set_matplotlib_backend(show_plots: bool):
    """
    Utility to switch to Agg when plots are not shown (avoid GUI deps).
    Call this once at startup in main.
    """
    if not show_plots:
        matplotlib.use("Agg")
