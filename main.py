"""DocuMind AI application entry point."""
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from collections import defaultdict, deque
from fastapi import FastAPI, Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from backend.core import settings, get_logger
from backend.api.routes import router

logger = get_logger("main")
BASE_DIR = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.api_key:
        logger.warning("API_KEY is empty; API-key authentication is disabled (development mode).")
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not configured.")
    yield

app = FastAPI(title=settings.app_name, description=settings.app_description,
    version=settings.app_version, lifespan=lifespan)

@app.exception_handler(StarletteHTTPException)
async def safe_server_errors(request: Request, exc: StarletteHTTPException):
    if exc.status_code >= 500:
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse({"detail": "Internal server error.", "request_id": request_id},
            status_code=exc.status_code)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
origins = settings.cors_origins
wildcard = "*" in origins
app.add_middleware(CORSMiddleware, allow_origins=origins,
    allow_credentials=not wildcard, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend" / "static"), name="static")
app.include_router(router)

_hits = defaultdict(deque)
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
# NOTE: rate limits are per client IP and kept in memory (single worker). Behind a
# reverse proxy configure trusted proxy headers, or limit at the proxy instead.


def _purge_stale_hits(now: float) -> None:
    """Remove empty/expired buckets so the limiter cannot grow without bound."""
    for key in [k for k, w in _hits.items() if not w or now - w[-1] > 60]:
        _hits.pop(key, None)

@app.middleware("http")
async def request_controls(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    request_id = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
    request.state.request_id = request_id
    limit = 10 if request.url.path == "/api/upload" else 30 if request.url.path in ("/api/query", "/api/query/stream") else None
    if limit:
        key = (request.client.host if request.client else "unknown", request.url.path)
        now = time.monotonic()
        if len(_hits) > 5000: _purge_stale_hits(now)
        window = _hits[key]
        while window and now - window[0] > 60: window.popleft()
        if len(window) >= limit:
            return JSONResponse({"detail": "Rate limit exceeded. Please try again shortly.", "request_id": request_id},
                status_code=429, headers={"X-Request-ID": request_id, "Retry-After": "60"})
        window.append(now)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error | request_id=%s", request_id)
        response = JSONResponse({"detail": "Internal server error.", "request_id": request_id}, status_code=500)
    response.headers["X-Request-ID"] = request_id
    logger.info("request complete | request_id=%s method=%s path=%s status=%s",
        request_id, request.method, request.url.path, response.status_code)
    return response

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(BASE_DIR / "frontend" / "templates" / "index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port,
        reload=settings.debug, log_level="info")
