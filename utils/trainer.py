import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def train_and_save_model(
    input_path="data/dummy_interactions.csv",
    output_path="data/user_similarity.csv"
):
    df = pd.read_csv(input_path)

    user_item_matrix = df.pivot_table(
        index='user_id',
        columns='product_id',
        values='interaction_score',
        fill_value=0
    )

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    user_similarity_df.to_csv(output_path)
    return user_item_matrix, user_similarity_df
