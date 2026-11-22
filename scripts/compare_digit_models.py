"""Kaggle #402 の学びを、別データで再現する最小比較実験。

実行:
    uv run --no-project --with scikit-learn==1.8.0 python scripts/compare_digit_models.py

注意: このスクリプトは Kaggle の Digit Recognizer データを使いません。
"""
import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    x, y = load_digits(return_X_y=True)  # 8x8、計1,797件
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    models = {
        "ロジスティック回帰": make_pipeline(
            StandardScaler(),  # 検証データを見ずに学習データだけで標準化
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "ランダムフォレスト": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
    }

    print(f"scikit-learn: {sklearn.__version__}")
    print(f"検証用画像: {len(y_valid)}件")
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        accuracy = accuracy_score(y_valid, predictions)
        print(f"{name}: 正解率 {accuracy:.2%} / 誤答 {(predictions != y_valid).sum()}件")
        if name == "ランダムフォレスト":
            print("ランダムフォレストの混同行列（行=正解、列=予測）:")
            print(confusion_matrix(y_valid, predictions))


if __name__ == "__main__":
    main()
