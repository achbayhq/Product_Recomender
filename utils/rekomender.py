import pandas as pd
from collections import defaultdict
from typing import List, Tuple

def load_data():
    df = pd.read_csv("data/dummy_interactions.csv")
    user_item_matrix = df.pivot_table(
        index='user_id',
        columns='product_id',
        values='interaction_score',
        fill_value=0
    )
    user_similarity_df = pd.read_csv("data/user_similarity.csv", index_col=0)
    user_similarity_df.columns = user_similarity_df.columns.astype(int)  # Pastikan integer
    return user_item_matrix, user_similarity_df

def get_recommendations(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    n_recommendations: int = 10
) -> List[Tuple[int, float]]:
    if user_id not in user_similarity_df.index:
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:4]

    recommendations = defaultdict(float)
    for similar_user, similarity_score in similar_users.items():
        for product_id, score in user_item_matrix.loc[int(similar_user)].items():
            if user_item_matrix.loc[user_id, product_id] == 0:
                recommendations[product_id] += score * similarity_score

    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return sorted_recommendations[:n_recommendations]
