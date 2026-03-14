from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import tempfile
import os
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kawatsu624.hiho.jp",
        "http://kawatsu624.hiho.jp",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = None
startup_error = None

try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key, timeout=60.0)
    else:
        startup_error = "OPENAI_API_KEY is not set"
except Exception as e:
    startup_error = f"OpenAI client init failed: {str(e)}"


@app.get("/")
def root():
    return {"ok": True, "startup_error": startup_error}


@app.get("/health")
def health():
    return {"ok": True, "startup_error": startup_error}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if startup_error:
        raise HTTPException(status_code=500, detail=startup_error)

    if client is None:
        raise HTTPException(status_code=500, detail="OpenAI client is not initialized")

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty audio")

    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1].lower()

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(data)

        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language="ja",
            )

        return {"ok": True, "text": tr.text or ""}

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"transcribe failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
