サマリー
# [PreTraM: Self-Supervised Pre-training via Connecting Trajectory and Map](https://www.ecva.net//papers/eccv_2022/papers_ECCV/html/3965_ECCV_2022_paper.php)


# 1. どんな論文か？
軌跡予測問題のためにHD-Mapに対する特徴量encoderを特徴空間でのaugumentationでpositive pairをつくり対照学習を行うMCLと、軌跡情報のencoder出力と対応する位置のHD-Mapのencoder出力をpositive pairとするTMCLを組み合わせることで、軌跡推定に最適な特徴量encoderを事前学習する手法PreTraMを提案している。

# 2 . 新規性
軌跡推定の分野では、軌跡情報と周辺HD-mapを用いたsupervisedな手法が主で、self-supervisedな事前学習法は殆ど研究されていなかった。  

# 3. 結果
既存手法であるAgentFomer, Trajectron++にPreTraMを適用した結果以下がわかった  
- AgentFormerにてオリジナルより17.1倍早い事前学習時間を実現
- ADE-5でAgenFormerでは0.097の改善、Trajectronでは0.074の改善
- AgentFormerにて70%のデータ使用で、オリジナルでの100%のデータ使用時の精度を越えた

# 4.その他（なぜ通ったか？等）
軌跡推定の問題では、そもそもの軌跡データのアノテーションコストが使用機材、クレンジング作業自体共に高く、収集困難な状況で有効なself-supervisedな事前学習法が求められていた。  
その環境の中で、軌跡データより比較的多く入手可能なHD-Mapの特徴空間での表現を、軌跡推定に適した形に上手く事前学習されている点が評価されていると感じた。  

## メモ
AgentFormer

Trajectron++