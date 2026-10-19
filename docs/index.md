# <span class="icon-with-blob animate-wave"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-database" style="color: #3B82F6;"><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 5A9 3 0 0 0 21 5A9 3 0 0 0 3 5"/><path d="M21 12A9 3 0 0 0 3 12"/><path d="M3 12A9 3 0 0 0 21 12"/></svg><span class="blob blob-blue blob-sm"></span></span> Data Science Docs

データサイエンスや統計学の学習記録、実務の中で得た気づきを書き留めている**個人用のナレッジベース**です。最初から完璧な記事を目指すのではなく、日々の学びや試行錯誤の過程を気軽にストックすることを大切にしています。

なお、一般向けの制作物やまとまったコンテンツは [Bearworks Hub (bearworks.uk)](https://bearworks.uk/) で公開しています。

---

## 統計学

統計検定の試験対策を中心に、基礎理論から分析現場での注意点までを整理しています。

- [**統計検定2級 学習用まとめ**](statistics_l2.md)
    - 2級の全体像、t分布、記述統計の要点、確率分布シミュレーターへのリンク。
- **記述統計**
    - [**1変数記述統計**](univariate-descriptive-statistics.md): 平均値・中央値・分散・四分位数・歪度など、ひとつの変数を要約する基礎指標。
    - [**2変数記述統計**](bivariate-descriptive-statistics.md): 散布図、共分散、相関係数、クロス集計表、単回帰分析など、変数間の関係を捉える手法。
- **分析の落とし穴**
    - [**切断効果（データ選択による相関の変化）**](truncation-effect.md): データの切り出し方によって、本来ある相関が消えたり反転したりする現象の実例と解説。
    - [**シンプソンのパラドックス（集計構成比による相関・割合の逆転）**](simpsons-paradox.md): 全体では負の相関なのにグループ別では正になるなど、交絡要因によって結論が逆転する現象の実例と解説。

## データサイエンス

- [**統計検定 DS発展 勉強メモ**](ds_dev.md)
    - データサイエンス発展（DS発展）のシラバスに沿った学習ノート（AI利活用、数理基礎、Pythonプログラミング、データハンドリングなど）。

## 実務への応用

- [**独立と排反（問い合わせ・障害分析）**](independence-mutual-exclusivity-in-operations.md)
    - 問い合わせ件数の二重計上防止や、システム障害の共通原因リスク評価など、実務に役立つ考え方とチェックリスト。
