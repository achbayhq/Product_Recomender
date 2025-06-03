import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_matrices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Muat user-item matrix dan similarity matrix."""
    try:
        interactions = pd.read_csv("data/dummy_interactions.csv")
        user_item_matrix = interactions.pivot_table(
            index='user_id',
            columns='product_id',
            values='interaction_score',
            fill_value=0
        )
        
        user_similarity_df = pd.read_csv(
            "data/user_similarity.csv", 
            index_col="user_id"
        )
        
        return user_item_matrix, user_similarity_df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def generate_recommendations(
    user_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity_df: pd.DataFrame,
    top_n: int = 10
) -> List[Tuple[int, float]]:
    """Generate rekomendasi dari similarity matrix."""
    try:
        # Validasi user exists
        if user_id not in user_similarity_df.index:
            logger.warning(f"User {user_id} tidak ditemukan")
            return []

        # 1. Cari 3 user paling mirip (exclude diri sendiri)
        similar_users = (
            user_similarity_df[user_id]
            .sort_values(ascending=False)
            .iloc[1:4]
        )

        # 2. Hitung score rekomendasi
        recommendations: Dict[int, float] = {}
        for similar_user, similarity in similar_users.items():
            for product_id in user_item_matrix.columns:
                if user_item_matrix.loc[user_id, product_id] == 0:  # Produk belum diinteraksi
                    recommendations[product_id] = (
                        recommendations.get(product_id, 0) + 
                        user_item_matrix.loc[similar_user, product_id] * similarity
                    )

        # 3. Ambil top_n rekomendasi
        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:top_n]

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []

def get_fallback_recommendations(top_n: int = 10) -> List[Tuple[int, float]]:
    """Fallback ke produk populer jika user baru."""
    try:
        interactions = pd.read_csv("data/dummy_interactions.csv")
        popular_products = (
            interactions.groupby('product_id')['interaction_score']
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        return list(popular_products.items())
    except Exception as e:
        logger.error(f"Error generating fallback: {e}")
        return []