"""
シンプソンのパラドックス解説用の図（図1：全体、図2：層別）を生成するスクリプト
Issue #405 再現用スクリプト

実行方法:
    uv run --with matplotlib --with numpy python scripts/generate_simpsons_charts.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定（Windows環境優先・フォールバック対応）
plt.rcParams["font.sans-serif"] = [
    "Meiryo",
    "Yu Gothic",
    "MS Gothic",
    "Hiragino Sans",
    "IPAexGothic",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

# 出力先ディレクトリ
OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 架空データ（Issue #405 例A）
xa = np.array([1, 2, 3, 4, 5], dtype=float)
ya = np.array([9.0, 10.2, 10.8, 12.2, 12.9], dtype=float)

xb = np.array([6, 7, 8, 9, 10], dtype=float)
yb = np.array([1.0, 2.2, 2.8, 4.2, 4.9], dtype=float)

x_all = np.concatenate([xa, xb])
y_all = np.concatenate([ya, yb])

# 回帰直線のパラメータ（最小二乗法）
# 全体
p_all = np.polyfit(x_all, y_all, 1)
r_all = np.corrcoef(x_all, y_all)[0, 1]

# タイプA
p_a = np.polyfit(xa, ya, 1)
r_a = np.corrcoef(xa, ya)[0, 1]

# タイプB
p_b = np.polyfit(xb, yb, 1)
r_b = np.corrcoef(xb, yb)[0, 1]

# プロット用範囲設定（両図で統一）
X_LIM = (0, 11)
Y_LIM = (-1.0, 15.5)
x_line_all = np.linspace(0.5, 10.5, 100)
x_line_a = np.linspace(0.8, 5.2, 100)
x_line_b = np.linspace(5.8, 10.2, 100)

# カラーパレット（視認性とアクセシビリティに配慮）
COLOR_OVERALL = "#4A5568"  # スレートグレー
COLOR_OVERALL_LINE = "#E53E3E"  # 警告レッド
COLOR_A = "#2B6CB0"  # ブルー
COLOR_B = "#DD6B20"  # アンバー/オレンジ


def create_overall_chart():
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=200)

    # 全体データ点
    ax.scatter(
        x_all,
        y_all,
        color=COLOR_OVERALL,
        s=100,
        alpha=0.85,
        edgecolor="#1A202C",
        linewidth=1.2,
        zorder=3,
        label="店舗データ（全10店舗）",
    )

    # 全体回帰直線
    ax.plot(
        x_line_all,
        np.polyval(p_all, x_line_all),
        color=COLOR_OVERALL_LINE,
        linestyle="--",
        linewidth=2.2,
        zorder=2,
        label=f"全体の回帰直線 (r = {r_all:.2f})",
    )

    # グラフ装飾
    ax.set_title(
        "図1：全体で見ると（タイプ区別なし）\n広告費が多いほど売上が低い？",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("広告費（架空単位）", fontsize=11, fontweight="bold")
    ax.set_ylabel("売上（架空単位）", fontsize=11, fontweight="bold")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E0")
    ax.set_axisbelow(True)

    # 相関係数のコールアウト
    ax.text(
        0.05,
        0.10,
        f"全体相関係数: r = {r_all:.2f}\n（全体では負の相関：右下がり）",
        transform=ax.transAxes,
        fontsize=10.5,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#FFF5F5",
            edgecolor="#FEB2B2",
            alpha=0.9,
        ),
    )

    # 架空データ注記
    ax.text(
        0.98,
        0.03,
        "※説明用の架空データです",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#718096",
        horizontalalignment="right",
    )

    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "simpsons-overall.svg")
    plt.close(fig)
    print("Saved simpsons-overall.svg")


def create_stratified_chart():
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=200)

    # 全体回帰直線（比較用・控えめな赤破線）
    ax.plot(
        x_line_all,
        np.polyval(p_all, x_line_all),
        color=COLOR_OVERALL_LINE,
        linestyle="--",
        linewidth=1.8,
        alpha=0.45,
        zorder=1,
        label=f"全体の回帰直線 (r = {r_all:.2f})",
    )

    # タイプA
    ax.scatter(
        xa,
        ya,
        color=COLOR_A,
        marker="o",
        s=110,
        alpha=0.9,
        edgecolor="#1A365D",
        linewidth=1.2,
        zorder=4,
        label=f"タイプA：丸 (r = +{r_a:.2f})",
    )
    ax.plot(
        x_line_a,
        np.polyval(p_a, x_line_a),
        color=COLOR_A,
        linestyle="-",
        linewidth=2.4,
        zorder=3,
    )

    # タイプB
    ax.scatter(
        xb,
        yb,
        color=COLOR_B,
        marker="s",
        s=100,
        alpha=0.9,
        edgecolor="#7B341E",
        linewidth=1.2,
        zorder=4,
        label=f"タイプB：四角 (r = +{r_b:.2f})",
    )
    ax.plot(
        x_line_b,
        np.polyval(p_b, x_line_b),
        color=COLOR_B,
        linestyle="-",
        linewidth=2.4,
        zorder=3,
    )

    # グループごとの説明注釈（因果表現を避け、相関・水準を記述）
    ax.text(
        3.0,
        13.5,
        "タイプA: 売上高水準・広告費少\n（タイプ内の相関は右上がり）",
        fontsize=9,
        color=COLOR_A,
        fontweight="bold",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#EBF8FF",
            edgecolor="#BEE3F8",
            alpha=0.9,
        ),
    )
    ax.text(
        8.0,
        5.5,
        "タイプB: 売上低水準・広告費多\n（タイプ内の相関は右上がり）",
        fontsize=9,
        color=COLOR_B,
        fontweight="bold",
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#FFFAF0",
            edgecolor="#FEEBC8",
            alpha=0.9,
        ),
    )

    # グラフ装飾
    ax.set_title(
        "図2：店舗タイプ別に色分けすると\nどのタイプでも広告費が多いほど売上が高い！",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("広告費（架空単位）", fontsize=11, fontweight="bold")
    ax.set_ylabel("売上（架空単位）", fontsize=11, fontweight="bold")
    ax.set_xlim(X_LIM)
    ax.set_ylim(Y_LIM)
    ax.grid(True, linestyle=":", alpha=0.5, color="#CBD5E0")
    ax.set_axisbelow(True)

    # 架空データ注記
    ax.text(
        0.98,
        0.03,
        "※説明用の架空データです",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#718096",
        horizontalalignment="right",
    )

    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)
    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "simpsons-stratified.svg")
    plt.close(fig)
    print("Saved simpsons-stratified.svg")


if __name__ == "__main__":
    create_overall_chart()
    create_stratified_chart()
