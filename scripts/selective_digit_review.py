"""手書き数字の「AIが迷ったものだけ人に回す」を再現する。

実行:
  uv run --no-project --with scikit-learn==1.8.0 python scripts/selective_digit_review.py

Kaggleの画像ではなく、scikit-learn付属の8x8 digitsデータを使います。
"""
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


OUT = Path("docs/images")
THRESHOLDS = (0.0, 0.50, 0.60, 0.70)  # 説明用に固定。実運用の最適値ではない。


def summarize(actual, prediction, score, threshold):
    auto = score >= threshold
    wrong = prediction != actual
    return {
        "threshold": threshold,
        "auto": int(auto.sum()),
        "review": int((~auto).sum()),
        "auto_errors": int((auto & wrong).sum()),
        "review_errors": int((~auto & wrong).sum()),
    }


def draw_tradeoff(rows):
    """自動件数と人の確認件数のトレードオフをSVG化。"""
    font = 'font-family="sans-serif"'
    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1030" height="475" viewBox="0 0 1030 475" role="img" aria-labelledby="title desc">',
        '<title id="title">スコアしきい値と確認件数・自動誤答</title>',
        '<desc id="desc">digitsの同じ360枚でしきい値を変えて比較した結果。</desc>',
        '<rect width="1030" height="475" rx="16" fill="white"/>',
        f'<text x="34" y="48" font-size="25" font-weight="bold" fill="#0f172a" {font}>人が確認する件数を増やすと、自動の間違いはどう変わる？</text>',
        f'<text x="34" y="80" font-size="15" fill="#475569" {font}>同じ検証360枚を、ランダムフォレストの予測スコアに応じて振り分けた結果</text>',
        '<rect x="36" y="112" width="16" height="16" fill="#2563eb"/>',
        f'<text x="60" y="126" font-size="14" fill="#334155" {font}>自動判定</text>',
        '<rect x="216" y="112" width="16" height="16" fill="#f59e0b"/>',
        f'<text x="240" y="126" font-size="14" fill="#334155" {font}>人が確認</text>',
        f'<text x="790" y="126" font-size="15" font-weight="bold" fill="#334155" {font}>自動判定の誤答</text>',
    ]
    for i, row in enumerate(rows):
        y = 177 + i * 67
        a, h = row["auto"], row["review"]
        width = a / 360 * 555
        name = "制限なし" if row["threshold"] == 0 else f'{row["threshold"]:.2f} 以上'
        p.extend([
            f'<text x="37" y="{y+21}" font-size="17" fill="#1e293b" {font}>{name}</text>',
            f'<rect x="165" y="{y}" width="{width:.1f}" height="34" rx="4" fill="#2563eb"/>',
            f'<rect x="{165+width:.1f}" y="{y}" width="{h / 360 * 555:.1f}" height="34" fill="#f59e0b"/>',
            f'<text x="798" y="{y+26}" font-size="25" font-weight="bold" fill="{"#dc2626" if row["auto_errors"] else "#15803d"}" {font}>{row["auto_errors"]}枚</text>',
            f'<text x="165" y="{y+52}" font-size="13" fill="#475569" {font}>自動 {a}枚 ／ 人が確認 {h}枚</text>',
        ])
    p.extend([
        f'<text x="34" y="454" font-size="13" fill="#64748b" {font}>※ 誤答0枚はこの検証データでの結果。新しいデータでの正解は保証されません。</text>',
        '</svg>',
    ])
    (OUT / "digit-human-review-tradeoff.svg").write_text("\n".join(p) + "\n", encoding="utf-8")


def draw_examples(digits, valid_ids, actual, prediction, scores):
    """高スコアの誤答と低スコアの正解を、元の8x8画素で表示。"""
    font = 'font-family="sans-serif"'
    p = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="940" height="340" viewBox="0 0 940 340" role="img" aria-labelledby="title desc">',
        '<title id="title">予測スコアが高い誤答と低い正解</title>',
        '<desc id="desc">digits ID1553は正解8・予測1・スコア0.60、ID103は正解3・予測3・スコア0.30。</desc>',
        '<rect width="940" height="340" rx="15" fill="white"/>',
        f'<text x="28" y="42" font-size="24" font-weight="bold" fill="#0f172a" {font}>スコアが高くても間違える。低くても正解する。</text>',
    ]
    id_to_index = {int(original): i for i, original in enumerate(valid_ids)}
    for idx, (original, heading) in enumerate(((1553, "高めのスコアでも誤答"), (103, "低いスコアでも正解"))):
        i = id_to_index[original]
        x = 24 + idx * 458
        truth, guess = int(actual[i]), int(prediction[i])
        score = float(scores[i])
        p.extend([
            f'<rect x="{x}" y="63" width="437" height="241" rx="10" fill="#eff6ff" stroke="#cbd5e1"/>',
            f'<text x="{x+18}" y="95" font-size="18" font-weight="bold" fill="#1e293b" {font}>{heading}</text>',
            f'<text x="{x+18}" y="126" font-size="15" fill="#475569" {font}>digits画像 ID {original}</text>',
            f'<text x="{x+18}" y="165" font-size="19" fill="#0f172a" {font}>正解：{truth}</text>',
            f'<text x="{x+18}" y="198" font-size="19" fill="{"#b91c1c" if truth != guess else "#15803d"}" {font}>AIの予測：{guess}</text>',
            f'<text x="{x+18}" y="232" font-size="19" fill="#334155" {font}>最大スコア：{score:.2f}</text>',
        ])
        for k, pixel in enumerate(digits.images[original].ravel()):
            gray = 255 - round(int(pixel) * 255 / 16)
            p.append(f'<rect x="{x+259+(k % 8)*19}" y="{116+(k // 8)*19}" width="19" height="19" fill="rgb({gray},{gray},{gray})"/>')
    p.extend([
        f'<text x="30" y="327" font-size="12" fill="#64748b" {font}>※ 最大スコアはpredict_probaの出力。正解確率として保証された数値ではありません。</text>',
        '</svg>',
    ])
    (OUT / "digit-human-review-examples.svg").write_text("\n".join(p) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    ids = np.arange(len(digits.target))
    x_train, x_test, y_train, y_test, _, valid_ids = train_test_split(
        digits.data, digits.target, ids,
        test_size=0.2, stratify=digits.target, random_state=42,
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    prediction = model.classes_[probabilities.argmax(axis=1)]
    scores = probabilities.max(axis=1)
    print(f"検証: {len(y_test)}件 / 正解: {int((prediction == y_test).sum())}件")
    rows = [summarize(y_test, prediction, scores, threshold) for threshold in THRESHOLDS]
    for r in rows:
        print(f'しきい値{r["threshold"]:.2f}: 自動{r["auto"]}件、'
              f'人の確認{r["review"]}件、自動誤答{r["auto_errors"]}件、'
              f'確認対象に含まれる誤答{r["review_errors"]}件')
    draw_tradeoff(rows)
    draw_examples(digits, valid_ids, y_test, prediction, scores)
    print("作成: docs/images/digit-human-review-{tradeoff,examples}.svg")
    print("注: しきい値は説明用の固定値で、将来の性能を保証しません。")


if __name__ == "__main__":
    main()
