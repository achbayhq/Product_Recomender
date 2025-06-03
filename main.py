from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from utils.trainer import train_and_save_model
from utils.rekomender import load_data, get_recommendations
import uvicorn

app = FastAPI()

# Load model saat startup
try:
    user_item_matrix, user_similarity_df = load_data()
except Exception as e:
    user_item_matrix, user_similarity_df = None, None
    print(f"Load model error: {e}")

@app.get("/recommend")
def recommend(user_id: int):
    if user_item_matrix is None or user_similarity_df is None:
        raise HTTPException(status_code=500, detail="Model belum siap.")

    if user_id not in user_item_matrix.index:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} tidak ditemukan.")

    results = get_recommendations(user_id, user_item_matrix, user_similarity_df)
    return {
        "user_id": user_id,
        "recommendations": [
            {"product_id": pid, "score": round(score, 2)} for pid, score in results
        ]
    }

# Tambahkan uvicorn.run() agar bisa dijalankan langsung
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
