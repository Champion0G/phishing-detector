"""
api/app.py - FastAPI Backend for Phishing Detector
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import predict_url

app = FastAPI(title="Phishing Detector API", version="1.0.0")

# Enable CORS for Next.js frontend (default dev port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

@app.get("/")
def read_root():
    return {"status": "online", "model_version": "V14 Hybrid"}

@app.post("/analyze")
def analyze_url(request: URLRequest):
    if not request.url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    try:
        result = predict_url(request.url)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
