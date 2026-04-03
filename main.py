"""
DocuMind AI — FastAPI Application Entry Point
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from pathlib import Path
import os
import uvicorn

from backend.core import settings, get_logger
from backend.api.routes import router

logger = get_logger("main")

# ── Base Directory (VERY IMPORTANT) ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files (CSS, JS) ────────────────────────────────────────────────────
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "frontend/static")),
    name="static"
)

# ── Templates (HTML) ──────────────────────────────────────────────────────────
templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "frontend/templates")
)

# ── Register API Routes ───────────────────────────────────────────────────────
app.include_router(router)

# ── Serv Frontend ────────────────────────────────────────────────────────────
from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("frontend/templates/index.html")


# ── Startup Event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name} v{settings.app_version}")
    logger.info(f"  LLM Model      : {settings.llm_model}")
    logger.info(f"  Embedding Model: {settings.embedding_model}")
    logger.info(f"  Groq API Key   : {'✅ Set' if settings.GROQ_API_KEY else '❌ NOT SET'}")
    logger.info(f"  Vector Store   : {settings.vectorstore_path}")
    logger.info(f"  API Docs       : http://localhost:{settings.port}/docs")
    logger.info(f"  Frontend       : http://localhost:{settings.port}/")
    logger.info("=" * 60)

    if not settings.GROQ_API_KEY:
        logger.warning(
            "⚠️  GROQ_API_KEY is not set!\n"
            "   Get a FREE key at: https://console.groq.com\n"
            "   Then add it to your .env file"
        )

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )