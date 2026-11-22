"""機械学習はどこで間違える？ 公開手書き数字による最小の誤答分析。

実行:
  uv run --no-project --with scikit-learn==1.8.0 --with matplotlib==3.10.8 python scripts/analyze_digit_errors.py

Kaggleの28×28画像とは別の、scikit-learn付属digits（8×8）を使用。
"""

from pathlib import Path
import numpy as np
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT = Path("docs/images")


def plot_confusions(y_true, predictions):
    """混同行列の対角線を隠し、取り違えだけを可視化。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), constrained_layout=True)
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        matrix = confusion_matrix(y_true, y_pred, labels=np.arange(10))
        np.fill_diagonal(matrix, 0)
        ax.imshow(matrix, cmap="Oranges", vmin=0, vmax=5)
        for r, c in np.argwhere(matrix > 0):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=11)
        ax.set(xticks=range(10), yticks=range(10),
               xlabel="Predicted digit", ylabel="True digit", title=name)
    fig.suptitle("Where did the models make mistakes? (correct predictions hidden)")
    fig.savefig(OUTPUT / "digit-errors-confusion.svg", bbox_inches="tight")
    plt.close(fig)


def plot_examples(digits, valid_ids, y_true, predictions):
    """検証データの実際の誤答6例。IDは公開digitsデータ内の行番号。"""
    examples = [
        ("Both wrong", 1553), ("Both wrong", 492),
        ("Only LR wrong", 1256), ("Only LR wrong", 639),
        ("Only RF wrong", 1593), ("Only RF wrong", 1216),
    ]
    index_lookup = {int(data_id): i for i, data_id in enumerate(valid_ids)}
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 9.2), constrained_layout=True)
    lr, rf = predictions["Logistic regression"], predictions["Random forest"]
    for ax, (category, data_id) in zip(axes.ravel(), examples):
        i = index_lookup[data_id]
        ax.imshow(digits.images[data_id], cmap="gray_r", vmin=0, vmax=16,
                  interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{category} | ID {data_id}\nTrue {y_true[i]} / LR {lr[i]} / RF {rf[i]}",
            fontsize=12, pad=7,
        )
    fig.suptitle("Six real misclassified digit images (8 x 8 pixels)")
    fig.savefig(OUTPUT / "digit-errors-examples.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    all_ids = np.arange(len(digits.target))
    x_train, x_valid, y_train, y_valid, _, valid_ids = train_test_split(
        digits.data, digits.target, all_ids,
        test_size=0.2, stratify=digits.target, random_state=42,
    )
    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
        ),
    }
    predictions = {}
    print(f"scikit-learn {sklearn.__version__}; validation: {len(y_valid)}")
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_valid)
        predictions[name] = prediction
        matrix = confusion_matrix(y_valid, prediction)
        print(f"{name}: accuracy={accuracy_score(y_valid, prediction):.4%}, "
              f"wrong={int(np.sum(prediction != y_valid))}")
        pairs = sorted(
            [(int(matrix[r, c]), r, c)
             for r in range(10) for c in range(10) if r != c and matrix[r, c]],
            reverse=True,
        )
        print("  main errors (count, correct label, predicted label):", pairs[:5])
    lr_wrong = predictions["Logistic regression"] != y_valid
    rf_wrong = predictions["Random forest"] != y_valid
    print("Both correct:", int(np.sum(~lr_wrong & ~rf_wrong)))
    print("Both wrong:", int(np.sum(lr_wrong & rf_wrong)))
    print("Only LR wrong:", int(np.sum(lr_wrong & ~rf_wrong)))
    print("Only RF wrong:", int(np.sum(~lr_wrong & rf_wrong)))
    plot_confusions(y_valid, predictions)
    plot_examples(digits, valid_ids, y_valid, predictions)


if __name__ == "__main__":
    main()
