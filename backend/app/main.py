from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.api.chat import router

# Create tables on startup (fine for a portfolio project; use Alembic
# migrations if this ever needs to survive schema changes in production).
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Stock Intelligence Platform",
    description="Historical analysis, technical indicators, ML predictions, and risk metrics for stocks.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"status": "ok", "service": "AI Stock Intelligence Platform API"}


@app.get("/health")
def health():
    return {"status": "healthy"}
