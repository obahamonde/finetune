from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.handler import app as handler

app = FastAPI(
    title="Fine-tuning API",
    description="API for fine-tuning models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(handler, prefix="/v1")
