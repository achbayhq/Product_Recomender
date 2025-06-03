import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_save_model(
    input_path: str = "data/dummy_interactions.csv",
    output_path: str = "data/user_similarity.csv"
) -> None:
    """
    Training model CF dan simpan similarity matrix ke CSV.
    """
    try:
        # Load data
        df = pd.read_csv(input_path)
        
        # Bentuk user-item matrix
        user_item_matrix = df.pivot_table(
            index='user_id',
            columns='product_id',
            values='interaction_score',
            fill_value=0
        )
        
        # Hitung similarity
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        
        # Simpan hasil
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        user_similarity_df.to_csv(output_path)
        logger.info(f"Model tersimpan di {output_path}")

    except Exception as e:
        logger.error(f"Gagal training model: {e}")
        raise