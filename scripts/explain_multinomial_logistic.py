"""手書き数字を多項ロジスティック回帰で分類し、係数のヒートマップを作る。

実行 (bearworks-docs リポジトリのルートから):
  uv run --no-project --with scikit-learn==1.8.0 python scripts/explain_multinomial_logistic.py

データは scikit-learn 付属の公開 digits (8x8)。
Kaggle Digit Recognizer (28x28) とは異なるデータ。
"""
from pathlib import Path
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _hex_color(weight: float, limit: float) -> str:
    """負=青、ゼロ=白、正=赤。係数の大きさを相対表示する。"""
    t = min(1.0, abs(weight) / limit) if limit else 0.0
    pale = (248, 250, 252)
    end = (37, 99, 235) if weight < 0 else (220, 38, 38)
    rgb = tuple(round(a * (1 - t) + b * t) for a, b in zip(pale, end))
    return "#" + "".join(f"{v:02x}" for v in rgb)


def save_coefficient_map(coefs, output: Path) -> None:
    """学習済み係数の 8x8 ヒートマップを外部ライブラリなしでSVG化する。"""
    digits = (0, 1, 9)
    max_abs = max(abs(float(v)) for d in digits for v in coefs[d])
    cell, step, top = 24, 240, 96
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="353" viewBox="0 0 850 353" role="img" aria-labelledby="title desc">',
        '<title id="title">多項ロジスティック回帰の数字0・1・9の係数ヒートマップ</title>',
        '<desc id="desc">scikit-learn digitsの8×8画像を学習した各クラスの標準化後の係数。赤は正、青は負、白はゼロ付近。特定のピクセルの因果的重要度ではない。</desc>',
        '<rect width="850" height="353" rx="14" fill="#ffffff"/>',
        '<text x="28" y="38" font-size="23" font-weight="bold" fill="#172554">数字の「判断材料」を覗いてみる</text>',
        '<text x="28" y="66" font-size="14" fill="#475569">scikit-learn digits（8×8）で学習した係数。3クラスを抜粋。</text>',
    ]
    for j, digit in enumerate(digits):
        left = 58 + j * step
        parts.append(
            f'<text x="{left + 96}" y="{top - 16}" font-size="21" text-anchor="middle" font-weight="bold" fill="#0f172a">数字 {digit}</text>'
        )
        for k, weight in enumerate(coefs[digit]):
            x = left + (k % 8) * cell
            y = top + (k // 8) * cell
            color = _hex_color(float(weight), max_abs)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff" stroke-width="0.5"/>'
            )
    parts.append(
        '<text x="58" y="313" font-size="14" fill="#2563eb" font-weight="bold">青：負の重み</text>'
        '<text x="350" y="313" font-size="14" fill="#475569">白：ほぼゼロ</text>'
        '<text x="625" y="313" font-size="14" fill="#dc2626" font-weight="bold">赤：正の重み</text>'
        '<text x="28" y="338" font-size="12" fill="#64748b">※ 係数はクラス間で相対的。色の強さだけで予測への寄与や因果関係は断定できません。</text>'
        '</svg>'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"係数の図を保存: {output}")


def main() -> None:
    x, y = load_digits(return_X_y=True)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    model = make_pipeline(
        StandardScaler(),  # 訓練データでだけ平均・標準偏差を学習する
        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_valid)
    accuracy = accuracy_score(y_valid, predicted)
    print(f"scikit-learn {sklearn.__version__} / digits {len(y)}件")
    print(f"検証 {len(y_valid)}件：正解率 {accuracy:.2%} / 誤答 {(predicted != y_valid).sum()}件")
    print("予測最初の1件:", model.predict_proba(x_valid[:1]).round(3).tolist()[0])
    print("混同行列（行=正解、列=予測）:\n", confusion_matrix(y_valid, predicted))

    save_coefficient_map(
        model.named_steps["logisticregression"].coef_,
        Path("docs/images/logistic-regression-pixel-weights.svg"),
    )


if __name__ == "__main__":
    main()
