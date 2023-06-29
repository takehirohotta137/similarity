import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# CSVファイルのパス
csv_file = 'streamer_unique_chat_authors3.csv'

# CSVファイルを読み込む
data = pd.read_csv(csv_file)


# ユーザーごとにストリーマーのコメント数を数える
user_streamer_counts = {}

# 各ユーザーに対してストリーマーごとのコメント数を数える
for i, row in data.iterrows():
    user = row['streamer_name']
    unique_chat_authors = eval(row['unique_chat_authors'])
    for author in unique_chat_authors:
        if author not in user_streamer_counts:
            user_streamer_counts[author] = {}
        user_streamer_counts[author][user] = user_streamer_counts[author].get(user, 0) + 1

# ユーザー-アイテム行列を作成
user_item_matrix = pd.DataFrame(user_streamer_counts).fillna(0).astype(int).transpose()
# コサイン類似度を計算
similarity = cosine_similarity(user_item_matrix)
simirarity_df = pd.DataFrame(similarity,columns=user_item_matrix.index,index=user_item_matrix.index)
simirarity_df.to_csv('simirarity.csv')
print(simirarity_df)