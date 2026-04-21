# cellular-complexity

2次元二値セル・オートマトン（ライフゲーム系）の全ルール **262,144 通り**を探索し、構造的複雑性の高いルールをパレート最適集合として抽出するツール。

## 背景

ライフゲーム（Conway's Game of Life）のようなセル・オートマトンは、単純なルールから複雑な振る舞いを生み出す。Birth/Survival 条件の組み合わせは 512×512 = 262,144 通りある。これらすべてを評価し、「Conway's Life に近い複雑性」を持つルールを探索する。

## 指標

3つの指標を多目的最適化（パレート最適）で同時最大化する。

| 指標 | 意味 | 高いとき |
|------|------|---------|
| **C** (Clustering) | 空間自己相関。生セルが近くにまとまる傾向 | セルが集積してパターンを形成 |
| **O** (Organization) | 自己組織化。後半で変化率が低下する度合い | カオスから秩序へ移行 |
| **S** (Structure) | パターン構造度。少数パターンが支配する度合い | 特定パターンが整理された形で出現 |

これらはランダムノイズ維持ルールでは低くなり、Conway's Life のような構造生成ルールで高くなるよう設計されている。

## インストール

```bash
pip install numpy matplotlib pillow
```

Python 3.10 以上、NumPy 1.24 以上を推奨。

## 使い方

### 全ルール探索

```bash
cd src/cellular-complexity
python3 full_rule_pareto_search.py --output-dir ../../output
```

オプション:

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--size` | 32 | 盤面サイズ（正方形） |
| `--steps` | 50 | 1ルールあたりの最大ステップ数 |
| `--trials` | 1 | 1ルールあたりの試行回数 |
| `--init-live-prob` | 0.5 | 初期生セル確率 |
| `--workers` | CPU数 | 並列プロセス数 |
| `--output-dir` | . | 出力先ディレクトリ |

出力ファイル:
- `all_rules_results.csv` — 全 262,144 ルールの指標値
- `pareto_front_results.csv` — パレート最適なルール群

### 可視化

```bash
python3 visualize.py B3/S23
python3 visualize.py B3/S23 --size 64 --steps 200
python3 visualize.py B3/S23 --output output/conway.gif
```

DISPLAY がある環境ではインタラクティブ表示（Space: 一時停止、R: リセット）。  
ない環境（SSH 接続など）では自動的に `output/` へ GIF を保存する。

## ファイル構成

```
cellular-complexity/
  src/cellular-complexity/
    full_rule_pareto_search.py   # 探索エンジン（指標計算・パレート抽出）
    visualize.py                 # ルール可視化・GIF 生成
  tests/
    conftest.py
    test_full_rule_pareto_search.py
  output/                        # 実行結果（git 管理外）
  report/
    report_240621.md             # 実験レポート
```

## テスト

```bash
pytest tests/
```

69件のテストで以下を検証している:
- ルール定義・近傍カウント・1ステップ更新の正確性
- 各指標（C/O/S）の数理的性質
- Conway's Life がランダムノイズ維持ルールよりも C・S で高スコアであること
- パレート最適の定義の整合性

## 実装の特徴

- **高速化**: `np.pad` + スライス加算による近傍カウント、`sliding_window_view` + ビットパックによる 3x3 パターン集計
- **並列化**: `multiprocessing.Pool` で全 CPU コアを使用
- **省メモリ**: CSV へのストリーミング書き込みで全件をメモリに保持しない
- **再現性**: 乱数シード固定で結果を再現可能

## パレート最適の定義

ルール `a` がルール `b` を支配する条件:
- C, O, S のすべてで `a >= b`
- かつ少なくとも1つで `a > b`

いずれのルールにも支配されないルールの集合をパレート最適集合とする。
