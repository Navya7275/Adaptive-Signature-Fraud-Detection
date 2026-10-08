"""
FastAPI application entry point.

Run with:
  cd backend
  uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.db import init_db
from app.routes import users, verify


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup (lifespan replaces deprecated on_event)."""
    init_db()
    print("=" * 60)
    print("  Adaptive Signature Fraud Detection System")
    print("  API running at http://localhost:8000")
    print("  Docs at http://localhost:8000/docs")
    print("=" * 60)
    yield


# ── Initialize ──
app = FastAPI(
    title="Adaptive Signature Fraud Detection System",
    description="Aging-aware signature verification with temporal drift profiling",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (for React frontend) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
app.include_router(users.router)
app.include_router(verify.router)


@app.get("/")
async def root():
    return {
        "system": "Adaptive Aging-Aware Signature Fraud Detection",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "enroll_user": "POST /api/users/enroll",
            "verify_signature": "POST /api/verify/",
            "list_users": "GET /api/users/",
            "user_history": "GET /api/users/{user_id}/history",
            "alerts": "GET /api/verify/alerts",
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}