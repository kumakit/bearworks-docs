"""学習済みCNNのフィルターと特徴マップを可視化する公開データ教材。

使い方（リポジトリのルートで実行）:
  python scripts/visualize_learned_cnn.py

PyTorch/NumPy/scikit-learn/Matplotlib が必要。CPUで実行します。
使用データ: scikit-learn digits（8x8）。Kaggleで提出したCNNとは別の小型モデル。
記事掲載のSVGと同じ学習データ・モデルで図を作りますが、描画形式はMatplotlibです。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI不要
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64
OUT = Path("docs/images")


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def inspect(self, batch):
        x1 = torch.relu(self.conv1(batch))
        p1 = self.pool1(x1)
        x2 = torch.relu(self.conv2(p1))
        p2 = self.pool2(x2)
        return {"conv1": x1, "pool1": p1, "conv2": x2,
                "logits": self.head(p2)}

    def forward(self, batch):
        return self.inspect(batch)["logits"]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
    })


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("図を保存:", OUT / name)


def visualize_kernels(before, after):
    """フィルターを共通色軸で比較。数値の違いを誤認しないため。"""
    vmax = max(float(np.abs(before).max()), float(np.abs(after).max()))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
    for i in range(8):
        row = 2 * (i // 4)
        col = i % 4
        for j, (label, kernels) in enumerate((("Before", before), ("After", after))):
            ax = axes[row + j, col]
            matrix = kernels[i, 0]
            ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(f"{label} / filter {i + 1}")
            for y in range(3):
                for x in range(3):
                    ax.text(x, y, f"{float(matrix[y,x]):+.2f}",
                            ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Real CNN kernels: before vs after 30 epochs", fontsize=16)
    save(fig, "cnn-actual-learned-filters.svg")


def visualize_feature_maps(image, original_id, label, model):
    """元画像とconv1,pool1,conv2の各4チャネルを表示。"""
    model.eval()
    with torch.no_grad():
        maps = model.inspect(image.unsqueeze(0))
    fig, axes = plt.subplots(3, 5, figsize=(12, 8.7), constrained_layout=True)
    for row, (layer, name) in enumerate((("conv1", "Conv1 + ReLU"),
                                         ("pool1", "MaxPool1"),
                                         ("conv2", "Conv2 + ReLU"))):
        acts = maps[layer][0].numpy()
        scale = max(float(acts.max()), 1e-6)  # 同じ層で同一スケール
        axes[row, 0].imshow(image[0].numpy(), cmap="gray_r",
                            vmin=0, vmax=1, interpolation="nearest")
        axes[row, 0].set_title(f"digit {label} (ID {original_id})")
        axes[row, 0].axis("off")
        for channel in range(4):
            ax = axes[row, channel + 1]
            ax.imshow(acts[channel], cmap="magma", vmin=0, vmax=scale,
                      interpolation="nearest")
            ax.set_title(f"{name} / ch {channel + 1}")
            ax.axis("off")
    fig.suptitle("Real activations in a trained CNN", fontsize=16)
    save(fig, "cnn-actual-feature-maps.svg")


def visualize_comparison(one, id_one, eight, id_eight, model):
    """数字1と8を同じ3フィルターに通した比較。色は両者で共通スケール。"""
    model.eval()
    with torch.no_grad():
        m1 = model.inspect(one.unsqueeze(0))["conv1"][0].numpy()
        m8 = model.inspect(eight.unsqueeze(0))["conv1"][0].numpy()
    vmax = max(float(m1.max()), float(m8.max()), 1e-6)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5), constrained_layout=True)
    for row, (digit, image, original_id, maps) in enumerate(
        ((1, one, id_one, m1), (8, eight, id_eight, m8))
    ):
        axes[row, 0].imshow(image[0].numpy(), cmap="gray_r",
                            vmin=0, vmax=1, interpolation="nearest")
        axes[row, 0].set_title(f"digit {digit} (ID {original_id})")
        axes[row, 0].axis("off")
        for col, channel in enumerate((0, 1, 3), start=1):
            axes[row, col].imshow(maps[channel], cmap="magma",
                                  vmin=0, vmax=vmax, interpolation="nearest")
            axes[row, col].set_title(f"Conv1 / channel {channel + 1}")
            axes[row, col].axis("off")
    fig.suptitle("Same trained filters, different digit images", fontsize=16)
    save(fig, "cnn-actual-one-vs-eight.svg")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    style()
    digits = load_digits()
    ids = np.arange(len(digits.target))
    train_ids, test_ids = train_test_split(
        ids, test_size=0.2, random_state=SEED, stratify=digits.target
    )
    images = torch.tensor(digits.images.astype(np.float32) / 16).unsqueeze(1)
    labels = torch.tensor(digits.target, dtype=torch.long)
    x_train, y_train = images[train_ids], labels[train_ids]
    x_test, y_test = images[test_ids], labels[test_ids]
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=BATCH_SIZE,
        shuffle=True, generator=torch.Generator().manual_seed(SEED)
    )
    model = SmallCNN()
    initial = model.conv1.weight.detach().numpy().copy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.CrossEntropyLoss()
    print(f"torch {torch.__version__} / sklearn {sklearn.__version__} / seed {SEED}")
    print(f"学習 {len(train_ids)}枚、検証 {len(test_ids)}枚、CPU")
    model.train()
    for epoch in range(EPOCHS):
        for x, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
        acts = model.inspect(x_test)["conv1"]
    print(f"検証正解率 {accuracy:.2%}（{int((preds == y_test).sum())}/{len(y_test)}枚）")
    after = model.conv1.weight.detach().numpy()
    diff = np.abs(after - initial).mean(axis=(1,2,3))
    maxima = acts.amax(dim=(0,2,3)).numpy()
    print("フィルターごとの重みの平均絶対変化:", np.round(diff, 4).tolist())
    print("Conv1のチャネルごとの検証データ最大出力:", np.round(maxima, 4).tolist())
    visualize_kernels(initial, after)
    correct = (preds == y_test).numpy()
    sample1 = next(i for i in range(len(test_ids)) if int(y_test[i]) == 1 and correct[i])
    sample8 = next(i for i in range(len(test_ids)) if int(y_test[i]) == 8 and correct[i])
    visualize_feature_maps(x_test[sample8], int(test_ids[sample8]), 8, model)
    visualize_comparison(x_test[sample1], int(test_ids[sample1]),
                         x_test[sample8], int(test_ids[sample8]), model)
    print(f"図の実画像: digit1 ID {test_ids[sample1]}、digit8 ID {test_ids[sample8]}")
    print("注意：Kaggleに提出したCNNの学習済み重みを使った実験ではありません。")


if __name__ == "__main__":
    main()
