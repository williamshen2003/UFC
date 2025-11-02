"""
Training utilities for XGBoost optimization.
Provides keyboard controls and trial plotting.
"""
from __future__ import annotations

import os, sys, threading, time
from pathlib import Path
from typing import Optional, Dict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ==================== TRAINING CONTROLLER ====================

class TrainingController:
    """Handles pause/resume/quit controls for training."""

    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.pause_lock = threading.Lock()
        self.listener_thread = None
        self.running = False

    def start_listener(self):
        """Start keyboard listener thread."""
        if self.listener_thread is not None and self.listener_thread.is_alive():
            return
        print("\n" + "=" * 70)
        print("  TRAINING CONTROLS: p(pause) | r(resume) | q(quit)")
        print("=" * 70 + "\n")
        self.running = True
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()

    def _keyboard_listener(self):
        """Listen for keyboard input in OS-specific way."""
        if os.name == "nt":
            self._listen_windows()
        else:
            self._listen_unix()

    def _listen_windows(self):
        """Windows keyboard listener."""
        try:
            import msvcrt
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
                        self._dispatch("".join(buf).strip().lower())
                        buf.clear()
                    elif ch != "\x03":
                        buf.append(ch)
                    else:
                        break
                else:
                    time.sleep(0.1)
            except Exception:
                break

    def _listen_unix(self):
        """Unix keyboard listener."""
        import select
        line = []
        while self.running:
            try:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch in ("\n", "\r"):
                        self._dispatch("".join(line).strip().lower())
                        line.clear()
                    else:
                        line.append(ch)
            except Exception:
                break

    def _dispatch(self, cmd: str):
        """Dispatch keyboard command."""
        if cmd == "p":
            with self.pause_lock:
                if not self.paused:
                    self.paused = True
                    print("\n⏸️  PAUSED — 'r' to resume, 'q' to quit\n")
        elif cmd == "r":
            with self.pause_lock:
                if self.paused:
                    self.paused = False
                    print("\n▶️  RESUMED\n")
        elif cmd == "q":
            with self.pause_lock:
                if not self.should_stop:
                    self.should_stop = True
                    print("\n🛑 QUIT REQUESTED\n")
        elif cmd:
            print(f"[Unknown '{cmd}'] Valid: p, r, q")

    def check_pause(self):
        """Check and handle pause state."""
        while True:
            with self.pause_lock:
                if self.should_stop:
                    raise KeyboardInterrupt("Training stopped by user")
                if not self.paused:
                    break
            time.sleep(0.3)

    def stop(self):
        """Stop listening."""
        self.running = False
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=1.0)


# ==================== PLOTTING ====================

def annotated_trial_plot(evals_result: Dict, title: str, best_idx: int, gap: float,
                        save_path_png: Optional[Path], show_plots: bool, save_plots_as_png: bool):
    """Render training progress plot with loss and accuracy."""
    if not evals_result:
        return

    tr_ll = evals_result.get("validation_0", {}).get("logloss")
    va_ll = evals_result.get("validation_1", {}).get("logloss")
    tr_er = evals_result.get("validation_0", {}).get("error")
    va_er = evals_result.get("validation_1", {}).get("error")
    if not all([tr_ll, va_ll, tr_er, va_er]):
        return

    iters = np.arange(1, len(tr_ll) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Loss
    axes[0].plot(iters, tr_ll, linewidth=2, label="Train LL")
    axes[0].plot(iters, va_ll, linewidth=2, label="Val LL")
    axes[0].axvline(best_idx + 1, linestyle="--", linewidth=1)
    axes[0].annotate(f"@{best_idx+1}\n{gap:.3f}", xy=(best_idx + 1, va_ll[best_idx]),
                    xytext=(best_idx + 1, max(va_ll) * 0.9),
                    arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    # Accuracy
    axes[1].plot(iters, 1.0 - np.asarray(tr_er), linewidth=2, label="Train Acc")
    axes[1].plot(iters, 1.0 - np.asarray(va_er), linewidth=2, label="Val Acc")
    axes[1].axvline(best_idx + 1, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Boosting Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_plots_as_png and save_path_png:
        fig.savefig(save_path_png, dpi=120)
    if show_plots:
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)


def set_matplotlib_backend(show_plots: bool):
    """Switch to Agg backend if not displaying plots."""
    if not show_plots:
        matplotlib.use("Agg")
