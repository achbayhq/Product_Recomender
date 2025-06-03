from datetime import datetime
from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
from utils.trainer import train_and_save_model
from utils.rekomender import load_matrices, generate_recommendations, get_fallback_recommendations
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Inisialisasi Data ---
try:
    user_item_matrix, user_similarity_df = load_matrices()
except Exception as e:
    logger.error(f"Startup error: {e}")
    user_item_matrix, user_similarity_df = None, None

# --- Skedul Training ---
scheduler = BackgroundScheduler()
scheduler.add_job(
    train_and_save_model,
    'interval',
    weeks=1,
    next_run_time=datetime.now()  # Jalankan segera saat startup
)
scheduler.start()

# --- Endpoint ---
@app.get("/recommend")
async def recommend(user_id: int):
    try:
        if user_item_matrix is None or user_similarity_df is None:
            raise HTTPException(status_code=503, detail="Model belum siap")
        
        recommendations = generate_recommendations(
            user_id, 
            user_item_matrix, 
            user_similarity_df
        )
        
        if not recommendations:
            recommendations = get_fallback_recommendations()
            logger.info(f"Fallback untuk user {user_id}")

        return {
            "user_id": user_id,
            "recommendations": [
                {"product_id": pid, "score": float(score)}
                for pid, score in recommendations
            ]
        }

    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info")