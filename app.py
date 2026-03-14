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
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ここでタイムアウトを明示
client = OpenAI(timeout=60.0)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/openai-check")
def openai_check():
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "ok": True,
        "has_openai_api_key": has_key
    }


async def _do_transcribe(audio: UploadFile) -> str:
    print("1. _do_transcribe start")

    data = await audio.read()
    size = len(data) if data else 0
    print(f"2. audio bytes = {size}")

    if not data:
        raise HTTPException(status_code=400, detail="empty audio")

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    suffix = ".webm"
    if audio.filename and "." in audio.filename:
        suffix = "." + audio.filename.rsplit(".", 1)[-1].lower()

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(data)

        print(f"3. temp file saved = {tmp_path}")
        print("4. calling OpenAI transcription...")

        with open(tmp_path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language="ja",
            )

        print("5. OpenAI transcription done")

        text = tr.text or ""
        print(f"6. transcript length = {len(text)}")

        return text

    except Exception as e:
        print("ERROR in _do_transcribe")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"transcribe failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print("7. temp file removed")
            except Exception:
                pass


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    print("POST /api/transcribe called")
    text = await _do_transcribe(audio)
    return {"ok": True, "text": text}