"""
セル・オートマトン可視化スクリプト。

使い方:
    python3 visualize.py B3/S23
    python3 visualize.py B3/S23 --size 64 --steps 200 --seed 42
    python3 visualize.py B3/S23 --output result.gif   # GIF 保存
    python3 visualize.py B36/S23 --interval 50        # 高速再生
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from PIL import Image

# DISPLAY がない環境では Agg に落として GIF 保存のみにする
_HAS_DISPLAY = bool(os.environ.get("DISPLAY", ""))
if not _HAS_DISPLAY:
    matplotlib.use("Agg")

# output/ はプロジェクトルート直下に置く
_PROJECT_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

sys.path.insert(0, os.path.dirname(__file__))
from full_rule_pareto_search import (
    LifeRule,
    pattern_structure,
    spatial_autocorrelation,
    step,
)


# ======================================
# ルール文字列パーサ
# ======================================

def parse_rule(rule_str: str) -> LifeRule:
    """'B3/S23' 形式の文字列を LifeRule に変換する。"""
    m = re.fullmatch(r"B([0-8]*)/S([0-8]*)", rule_str.strip().upper())
    if not m:
        raise ValueError(
            f"ルール形式が不正です: {rule_str!r}\n"
            "正しい形式: B<数字>/S<数字>  例: B3/S23"
        )
    birth   = frozenset(int(c) for c in m.group(1))
    survive = frozenset(int(c) for c in m.group(2))
    return LifeRule(birth=birth, survive=survive)


# ======================================
# フレーム描画ヘルパー
# ======================================

def _make_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig = plt.figure(figsize=(7, 8), facecolor="#111111")
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[10, 1], hspace=0.08)
    ax_board   = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])
    ax_board.axis("off")
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis("off")
    ax_metrics.set_facecolor("#111111")
    return fig, ax_board, ax_metrics


def _render_frame(
    fig: plt.Figure,
    ax_board: plt.Axes,
    ax_metrics: plt.Axes,
    rule: LifeRule,
    board: np.ndarray,
    t: int,
    steps: int,
    status: str,
) -> Image.Image:
    """現在の盤面を matplotlib で描画して PIL Image として返す。"""
    c          = spatial_autocorrelation(board)
    s          = pattern_structure(board)
    live_ratio = float(board.mean())

    ax_board.cla()
    ax_board.imshow(board, cmap="binary_r", vmin=0, vmax=1, interpolation="nearest")
    ax_board.axis("off")

    step_label = f"step {t:>4d} / {steps}" if not status else f"step {t:>4d}  [{status}]"
    ax_board.text(
        0.02, 0.97, step_label,
        ha="left", va="top", color="#aaffaa", fontsize=11,
        fontfamily="monospace", transform=ax_board.transAxes,
    )

    ax_metrics.cla()
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis("off")
    ax_metrics.set_facecolor("#111111")
    ax_metrics.text(
        0.5, 0.5,
        f"C={c:.3f}   S={s:.3f}   live={live_ratio:.3f}",
        ha="center", va="center", color="#cccccc", fontsize=11,
        fontfamily="monospace", transform=ax_metrics.transAxes,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


# ======================================
# GIF 保存（PIL で直接フレームを積む）
# ======================================

def _save_gif(
    rule: LifeRule,
    board: np.ndarray,
    steps: int,
    interval: int,
    path: str,
) -> None:
    fig, ax_board, ax_metrics = _make_figure()
    fig.suptitle(f"Rule: {rule}", color="white", fontsize=14, fontweight="bold", y=0.98)

    frames: list[Image.Image] = []
    print(f"GIF を保存中: {path}  ({steps} フレーム) ", end="", flush=True)

    for t in range(steps):
        frames.append(_render_frame(fig, ax_board, ax_metrics, rule, board, t, steps, ""))

        nb = step(board, rule)
        if nb.sum() == 0 or np.array_equal(nb, board):
            status = "extinct" if nb.sum() == 0 else "fixed"
            frames.append(_render_frame(fig, ax_board, ax_metrics, rule, nb, t + 1, steps, status))
            break
        board = nb

        if t % 10 == 0:
            print(".", end="", flush=True)

    plt.close(fig)

    fps = max(1, 1000 // interval)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=interval,
        optimize=False,
    )
    print(f"  完了 ({len(frames)} フレーム)")


# ======================================
# インタラクティブ表示（DISPLAY あり時）
# ======================================

def _show_interactive(
    rule: LifeRule,
    board: np.ndarray,
    steps: int,
    interval: int,
) -> None:
    fig, ax_board, ax_metrics = _make_figure()
    fig.suptitle(f"Rule: {rule}", color="white", fontsize=14, fontweight="bold", y=0.98)

    img = ax_board.imshow(board, cmap="binary_r", vmin=0, vmax=1, interpolation="nearest", animated=True)

    metrics_text = ax_metrics.text(
        0.5, 0.5, "",
        ha="center", va="center", color="#cccccc", fontsize=11,
        fontfamily="monospace", transform=ax_metrics.transAxes,
    )
    step_text = ax_board.text(
        0.02, 0.97, "",
        ha="left", va="top", color="#aaffaa", fontsize=11,
        fontfamily="monospace", transform=ax_board.transAxes,
    )
    fig.text(
        0.99, 0.01, "Space: pause  R: reset",
        ha="right", va="bottom", color="#555555", fontsize=8,
    )

    state = {"board": board.copy(), "step": 0, "paused": False, "done": False}
    rng   = np.random.default_rng()

    def update(_frame: int) -> list:
        if state["paused"] or state["done"]:
            return [img, metrics_text, step_text]
        b = state["board"]
        t = state["step"]
        img.set_data(b)
        step_text.set_text(f"step {t:>4d} / {steps}")
        metrics_text.set_text(
            f"C={spatial_autocorrelation(b):.3f}   "
            f"S={pattern_structure(b):.3f}   "
            f"live={float(b.mean()):.3f}"
        )
        nb = step(b, rule)
        if nb.sum() == 0 or np.array_equal(nb, b):
            state["done"] = True
            label = "extinct" if nb.sum() == 0 else "fixed"
            step_text.set_text(f"step {t:>4d}  [{label}]")
        else:
            state["board"] = nb
            state["step"]  = t + 1
        return [img, metrics_text, step_text]

    def on_key(event: matplotlib.backend_bases.KeyEvent) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "r":
            state["board"]  = (rng.random(board.shape) < 0.35).astype(np.uint8)
            state["step"]   = 0
            state["done"]   = False
            state["paused"] = False

    fig.canvas.mpl_connect("key_press_event", on_key)
    animation.FuncAnimation(fig, update, frames=steps, interval=interval, blit=True, repeat=False)
    plt.show()


# ======================================
# エントリポイント
# ======================================

def run_visualization(
    rule: LifeRule,
    size: int,
    steps: int,
    init_prob: float,
    seed: int,
    interval: int,
    output: str | None,
) -> None:
    rng   = np.random.default_rng(seed)
    board = (rng.random((size, size)) < init_prob).astype(np.uint8)

    if output:
        _save_gif(rule, board, steps, interval, output)
    elif _HAS_DISPLAY:
        _show_interactive(rule, board, steps, interval)
    else:
        os.makedirs(_DEFAULT_OUTPUT_DIR, exist_ok=True)
        filename = f"{str(rule).replace('/', '_')}_seed{seed}.gif"
        path = os.path.join(_DEFAULT_OUTPUT_DIR, filename)
        _save_gif(rule, board, steps, interval, path)


# ======================================
# CLI
# ======================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="セル・オートマトンのアニメーション可視化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python3 visualize.py B3/S23
  python3 visualize.py B36/S23 --size 64 --steps 200
  python3 visualize.py B3/S23 --output conway.gif
  python3 visualize.py B/S     --steps 5 --output dead.gif
""",
    )
    parser.add_argument("rule",           type=str,                  help="ルール文字列 (例: B3/S23)")
    parser.add_argument("--size",         type=int,   default=64,    help="盤面サイズ (default: 64)")
    parser.add_argument("--steps",        type=int,   default=100,   help="最大ステップ数 (default: 100)")
    parser.add_argument("--seed",         type=int,   default=42,    help="乱数シード (default: 42)")
    parser.add_argument("--init-prob",    type=float, default=0.35,  help="初期生セル確率 (default: 0.35)")
    parser.add_argument("--interval",     type=int,   default=100,   help="フレーム間隔 ms (default: 100)")
    parser.add_argument("--output",       type=str,   default=None,  help="保存先ファイル (.gif)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        rule = parse_rule(args.rule)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"ルール    : {rule}")
    print(f"盤面サイズ : {args.size}x{args.size}")
    print(f"ステップ数 : {args.steps}")
    print(f"シード    : {args.seed}")
    if _HAS_DISPLAY and not args.output:
        print("Space: 一時停止  R: リセット")
    print()

    run_visualization(
        rule      = rule,
        size      = args.size,
        steps     = args.steps,
        init_prob = args.init_prob,
        seed      = args.seed,
        interval  = args.interval,
        output    = args.output,
    )


if __name__ == "__main__":
    main()
