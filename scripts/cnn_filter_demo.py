"""図解記事「CNNはどう数字を読む？」に登場する計算を再現する最小コード。

使い方: python scripts/cnn_filter_demo.py
外部パッケージ不要 / CNNモデルの学習そのものは行いません。
"""

# 各行に同じ形の縦線がある、説明用の5x5画像（0=暗い、1=明るい）
image = [
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
]

# 左右の濃淡差を検出する、説明用の固定フィルター
# 学習したCNNの重みではありません。
kernel = [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
]


def scan(image_5x5, kernel_3x3):
    """パディングなし、1マスずつ進めるCNN式の積和を計算。"""
    output = []
    for row in range(3):
        values = []
        for col in range(3):
            total = sum(
                image_5x5[row + r][col + c] * kernel_3x3[r][c]
                for r in range(3)
                for c in range(3)
            )
            values.append(total)
        output.append(values)
    return output


feature_map = scan(image, kernel)
print("3×3フィルターの出力（ReLUの前）:")
for row in feature_map:
    print(row)

print("\nReLUの後（負の値は0）:")
for row in feature_map:
    print([max(0, value) for value in row])

# 別の独立した例。各2×2ブロックの最大値を抜き出す
pool_input = [
    [1, 3, 0, 2],
    [2, 1, 5, 1],
    [0, 4, 2, 2],
    [1, 0, 3, 6],
]
pool_output = [
    [
        max(
            pool_input[2 * r][2 * c],
            pool_input[2 * r][2 * c + 1],
            pool_input[2 * r + 1][2 * c],
            pool_input[2 * r + 1][2 * c + 1],
        )
        for c in range(2)
    ]
    for r in range(2)
]
print("\n2×2 MaxPoolingの結果:")
for row in pool_output:
    print(row)

assert feature_map == [[3, 3, -3], [3, 3, -3], [3, 3, -3]]
assert pool_output == [[3, 5], [4, 6]]
