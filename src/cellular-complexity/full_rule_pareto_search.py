from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ======================================
# 1. データ構造
# ======================================

@dataclass(frozen=True)
class LifeRule:
    birth: frozenset[int]
    survive: frozenset[int]

    def __post_init__(self) -> None:
        # ステップごとの np.isin + list() 変換を避けるため bool[9] テーブルをキャッシュ
        birth_lut = np.zeros(9, dtype=bool)
        survive_lut = np.zeros(9, dtype=bool)
        for b in self.birth:
            birth_lut[b] = True
        for s in self.survive:
            survive_lut[s] = True
        object.__setattr__(self, '_birth_lookup', birth_lut)
        object.__setattr__(self, '_survive_lookup', survive_lut)

    def __str__(self) -> str:
        b = "".join(str(x) for x in sorted(self.birth))
        s = "".join(str(x) for x in sorted(self.survive))
        return f"B{b}/S{s}"


@dataclass
class RuleMetrics:
    rule: LifeRule
    clustering: float      # C: 空間自己相関
    organization: float    # O: 自己組織化指標
    structure: float       # S: パターン構造度
    final_live_ratio: float


# ======================================
# 2. 近傍カウントと1ステップ更新
# ======================================

def count_neighbors(board: np.ndarray) -> np.ndarray:
    """
    各セルの周囲8セルの生存数を数える。トーラス境界。
    np.pad(wrap) で1枚の拡張配列を作りスライス加算するため、
    np.roll を8回呼ぶより中間配列が少なく約2.4倍速い。
    """
    p = np.pad(board, 1, mode="wrap")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
        p[1:-1, :-2] +               p[1:-1, 2:] +
        p[2:,  :-2] + p[2:,  1:-1] + p[2:,  2:]
    )


def step(board: np.ndarray, rule: LifeRule) -> np.ndarray:
    """ライフゲーム系ルールで1ステップ更新する。"""
    neighbors = count_neighbors(board)
    birth_mask   = rule._birth_lookup[neighbors]   & (board == 0)
    survive_mask = rule._survive_lookup[neighbors] & (board == 1)
    next_board = np.zeros_like(board, dtype=np.uint8)
    next_board[birth_mask | survive_mask] = 1
    return next_board


# ======================================
# 3. 指標 C, O, S
# ======================================

def spatial_autocorrelation(board: np.ndarray) -> float:
    """
    C: 空間自己相関。生セルが他の生セルの近くに集積する度合い。

    Pearson corr(board, count_neighbors(board)) を [0,1] に正規化。
    - 高い  → セルが島状に集積（Conway的）
    - ~0    → ランダム分布（ノイズ的）
    """
    p = float(board.mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    b = board.astype(float)
    n = count_neighbors(board).astype(float)
    b_c, n_c = b - p, n - p * 8
    num = float(np.mean(b_c * n_c))
    den = float(np.std(b) * np.std(n))
    if den < 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, (num / den + 1.0) / 2.0)))


# 3x3 パッチをビット整数（0-511）に変換するための重みベクトル（定数）
_PATCH_POWERS = (2 ** np.arange(9, dtype=np.uint16)).reshape(1, 9)


def pattern_structure(board: np.ndarray) -> float:
    """
    S: パターン構造度。少数のパターンが支配的かどうか。

    観測された 3x3 パターンの頻度分布エントロピーを正規化して反転する。
    - 高い  → 特定パターンが支配（整理された状態）
    - ~0    → 全パターンほぼ均等（ランダムノイズ）
    """
    windows = sliding_window_view(board, (3, 3))
    packed = (windows.reshape(-1, 9).astype(np.uint16) * _PATCH_POWERS).sum(axis=1)
    counts = np.bincount(packed, minlength=512)
    counts = counts[counts > 0]
    if len(counts) == 0:
        return 0.0
    probs = counts / counts.sum()
    H = float(-np.sum(probs * np.log2(probs + 1e-12)))
    max_H = np.log2(float(len(counts)))
    if max_H < 1e-9:
        return 1.0  # 1種類のみ
    return float(1.0 - H / max_H)


def _self_organization(deltas: list[float], actual_steps: int, max_steps: int) -> float:
    """
    O: 自己組織化指標。シミュレーション後半で変化率が下がる度合い。

    後半の変化率が前半より低いほど高い。ライフタイムペナルティで
    即死・即固定ルールへの過大評価を防ぐ。
    """
    if len(deltas) < 2:
        return 0.0
    mid = max(1, len(deltas) // 2)
    first  = float(np.mean(deltas[:mid]))
    second = float(np.mean(deltas[mid:]))
    if first < 1e-9:
        return 0.0
    raw = max(0.0, 1.0 - second / first)
    return raw * (actual_steps / max_steps)


# ======================================
# 4. ルール全列挙
# ======================================

def all_subsets_of_0_to_8() -> list[frozenset[int]]:
    """{0..8} の全部分集合 512 通りをビットマスクで生成する。"""
    nums = list(range(9))
    return [
        frozenset(nums[i] for i in range(9) if (mask >> i) & 1)
        for mask in range(512)
    ]


def generate_all_rules() -> Iterator[LifeRule]:
    """全ルール 262,144 通りを順番に返す。"""
    subsets = all_subsets_of_0_to_8()
    for birth in subsets:
        for survive in subsets:
            yield LifeRule(birth=birth, survive=survive)


# ======================================
# 5. 1ルールを評価
# ======================================

def simulate_rule_once(
    rule: LifeRule,
    size: int,
    steps: int,
    init_live_prob: float,
    seed: int,
) -> RuleMetrics:
    rng = np.random.default_rng(seed)
    board = (rng.random((size, size)) < init_live_prob).astype(np.uint8)
    n_cells = size * size

    clustering_sum = 0.0
    structure_sum  = 0.0
    deltas: list[float] = []
    lifetime = steps

    for t in range(steps):
        clustering_sum += spatial_autocorrelation(board)
        structure_sum  += pattern_structure(board)

        next_board = step(board, rule)
        deltas.append(float(np.sum(next_board != board)) / n_cells)

        if next_board.sum() == 0 or np.array_equal(next_board, board):
            lifetime = t + 1
            board = next_board
            break

        board = next_board

    count = len(deltas)
    return RuleMetrics(
        rule=rule,
        clustering=clustering_sum / count if count > 0 else 0.0,
        organization=_self_organization(deltas, lifetime, steps),
        structure=structure_sum / count if count > 0 else 0.0,
        final_live_ratio=float(board.mean()),
    )


def simulate_rule_average(
    rule: LifeRule,
    size: int = 32,
    steps: int = 50,
    trials: int = 1,
    init_live_prob: float = 0.5,
    base_seed: int = 0,
) -> RuleMetrics:
    """同じルールを複数回試して平均を取る。"""
    if trials == 1:
        return simulate_rule_once(rule, size, steps, init_live_prob, base_seed)

    raw = np.array([
        (m.clustering, m.organization, m.structure, m.final_live_ratio)
        for m in (
            simulate_rule_once(rule, size, steps, init_live_prob, base_seed + i)
            for i in range(trials)
        )
    ])
    means = raw.mean(axis=0)
    return RuleMetrics(
        rule=rule,
        clustering=float(means[0]),
        organization=float(means[1]),
        structure=float(means[2]),
        final_live_ratio=float(means[3]),
    )


# ======================================
# 6. CSV 保存
# ======================================

_CSV_HEADER = ["rule", "C_clustering", "O_organization", "S_structure", "final_live_ratio"]


def open_result_csv(path: str) -> tuple[object, object]:
    """CSV ファイルを開いてヘッダを書き込む。(file, writer) を返す。"""
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(_CSV_HEADER)
    return f, writer


def write_metrics_row(writer: object, m: RuleMetrics) -> None:
    writer.writerow([
        str(m.rule),
        f"{m.clustering:.6f}",
        f"{m.organization:.6f}",
        f"{m.structure:.6f}",
        f"{m.final_live_ratio:.6f}",
    ])


def save_metrics_csv(path: str, results: list[RuleMetrics]) -> None:
    f, writer = open_result_csv(path)
    for m in results:
        write_metrics_row(writer, m)
    f.close()


# ======================================
# 7. パレート最適
# ======================================

def dominates(a: RuleMetrics, b: RuleMetrics) -> bool:
    """
    a が b を支配する条件:
    C, O, S の全てで a >= b かつ少なくとも1つで a > b。
    """
    return (
        a.clustering   >= b.clustering   and
        a.organization >= b.organization and
        a.structure    >= b.structure    and
        (a.clustering   > b.clustering   or
         a.organization > b.organization or
         a.structure    > b.structure)
    )


def pareto_front(metrics_list: list[RuleMetrics]) -> list[RuleMetrics]:
    """
    誰にも支配されないルールだけを残す。
    各候補を numpy ベクトル演算で判定する（計算量は O(n²) のまま）。
    """
    if not metrics_list:
        return []

    vals = np.array([
        (m.clustering, m.organization, m.structure)
        for m in metrics_list
    ])
    n = len(vals)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        ge = np.all(vals >= vals[i], axis=1)
        gt = np.any(vals > vals[i], axis=1)
        mask = ge & gt
        mask[i] = False
        if mask.any():
            dominated[i] = True

    return [m for i, m in enumerate(metrics_list) if not dominated[i]]


def reduce_duplicates_by_metrics(
    metrics_list: list[RuleMetrics],
    digits: int = 6,
) -> list[RuleMetrics]:
    """C, O, S が丸め後に一致するものを1件に集約してパレート判定件数を削減する。"""
    if not metrics_list:
        return []

    vals = np.array([
        (m.clustering, m.organization, m.structure)
        for m in metrics_list
    ])
    _, unique_idx = np.unique(np.round(vals, digits), axis=0, return_index=True)
    return [metrics_list[i] for i in sorted(unique_idx)]


# ======================================
# 8. 並列化ワーカー（モジュールレベルで定義しないと pickle 不可）
# ======================================

def _worker(args: tuple) -> tuple[str, float, float, float, float]:
    """1ルールを評価してシリアライズ可能なタプルで返す。"""
    birth_list, survive_list, size, steps, trials, init_live_prob, seed = args
    rule = LifeRule(birth=frozenset(birth_list), survive=frozenset(survive_list))
    m = simulate_rule_average(rule, size, steps, trials, init_live_prob, seed)
    return (str(m.rule), m.clustering, m.organization, m.structure, m.final_live_ratio)


def _make_worker_args(
    rules: list[LifeRule],
    size: int,
    steps: int,
    trials: int,
    init_live_prob: float,
) -> list[tuple]:
    return [
        (list(r.birth), list(r.survive), size, steps, trials, init_live_prob, (i + 1) * 1000)
        for i, r in enumerate(rules)
    ]


# ======================================
# 9. 実行本体
# ======================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="セル・オートマトン全ルールパレート探索")
    parser.add_argument("--size",           type=int,   default=32,    help="盤面サイズ（正方形）")
    parser.add_argument("--steps",          type=int,   default=50,    help="最大ステップ数")
    parser.add_argument("--trials",         type=int,   default=1,     help="1ルールあたりの試行回数")
    parser.add_argument("--init-live-prob", type=float, default=0.5,   help="初期生セル確率")
    parser.add_argument("--workers",        type=int,   default=None,  help="並列プロセス数（省略時は CPU 数）")
    parser.add_argument("--output-dir",     type=str,   default=".",   help="出力先ディレクトリ")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    size           = args.size
    steps          = args.steps
    trials         = args.trials
    init_live_prob = args.init_live_prob
    workers        = args.workers or multiprocessing.cpu_count()
    output_dir     = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    all_csv_path    = os.path.join(output_dir, "all_rules_results.csv")
    pareto_csv_path = os.path.join(output_dir, "pareto_front_results.csv")

    total_rules = 512 * 512

    print(f"全ルール探索を開始します")
    print(f"盤面サイズ : {size}x{size}")
    print(f"ステップ数 : {steps}")
    print(f"試行回数   : {trials}")
    print(f"プロセス数 : {workers}")
    print(f"対象ルール数: {total_rules:,}")
    print()

    all_rules = list(generate_all_rules())
    worker_args = _make_worker_args(all_rules, size, steps, trials, init_live_prob)

    start_time = time.time()
    results: list[RuleMetrics] = []

    csv_file, csv_writer = open_result_csv(all_csv_path)

    try:
        with multiprocessing.Pool(processes=workers) as pool:
            for idx, row in enumerate(
                pool.imap(_worker, worker_args, chunksize=64), start=1
            ):
                rule_str, c, o, s, final = row
                csv_writer.writerow([rule_str, f"{c:.6f}", f"{o:.6f}", f"{s:.6f}", f"{final:.6f}"])

                rule = all_rules[idx - 1]
                results.append(RuleMetrics(
                    rule=rule,
                    clustering=c,
                    organization=o,
                    structure=s,
                    final_live_ratio=final,
                ))

                if idx % 5000 == 0:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed
                    remaining = (total_rules - idx) / rate
                    print(
                        f"{idx:>7,} / {total_rules:,}  "
                        f"経過 {elapsed:6.1f}s  "
                        f"残り ~{remaining:6.1f}s  "
                        f"({rate:.0f} rules/s)"
                    )
    finally:
        csv_file.close()

    elapsed = time.time() - start_time
    print(f"\n全ルールの評価が完了しました。  総時間: {elapsed:.1f}s  ({total_rules/elapsed:.0f} rules/s)")
    print(f"{all_csv_path} を保存しました。")

    reduced = reduce_duplicates_by_metrics(results, digits=6)
    print(f"\nパレート判定前の件数  : {len(results):,}")
    print(f"重複削減後の件数      : {len(reduced):,}")

    pareto = pareto_front(reduced)

    pareto_sorted = sorted(
        pareto,
        key=lambda m: (m.clustering, m.organization, m.structure),
        reverse=True,
    )

    save_metrics_csv(pareto_csv_path, pareto_sorted)
    print(f"{pareto_csv_path} を保存しました。")
    print(f"パレート最適ルール数  : {len(pareto_sorted):,}")

    print("\n上位30件（C, O, S の辞書順降順）")
    for m in pareto_sorted[:30]:
        print(
            f"{str(m.rule):15s} "
            f"C={m.clustering:.4f} "
            f"O={m.organization:.4f} "
            f"S={m.structure:.4f} "
            f"final={m.final_live_ratio:.4f}"
        )

    total_elapsed = time.time() - start_time
    print(f"\n総処理時間: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
