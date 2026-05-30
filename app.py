from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import tempfile
import os
import traceback
import json
from copy import deepcopy
import ffmpeg

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


DEFAULT_REPORT = {
    "実施日": "",
    "前回実施日": "",
    "次回予定日": "",
    "利用者名": "",
    "利用者名カナ": "",
    "出力氏名": "",
    "性別": "",
    "生年月日": "",
    "年齢": "",
    "介護度": "",
    "認定開始日": "",
    "認定終了日": "",
    "住所": "",
    "住所1": "",
    "住所2": "",
    "住所3": "",
    "電話番号": "",
    "電話番号1": "",
    "担当名": "",
    "ケアマネ姓名": "",
    "お話伺った人1": "",
    "お話伺った人2": "",
    "お話伺った人3": "",
    "お話伺った人その他": "",
    "確認方法1": "",
    "確認方法2": "",
    "専門相談員による結果": "",
    "福祉用具利用目標": [],
    "商品一覧": [],
    "身体状況の変化1": "",
    "身体状況の変化2": "",
    "身体状況の変化備考": "",
    "ご家族状況の変化1": "",
    "ご家族状況の変化2": "",
    "ご家族状況の変化備考": "",
    "お気持ちの変化1": "",
    "お気持ちの変化2": "",
    "お気持ちの変化備考": "",
    "生活状況の変化1": "",
    "生活状況の変化2": "",
    "生活状況の変化備考": "",
    "見直しの必要性1": "",
    "見直しの必要性2": ""
}

DEFAULT_GOAL = {
    "目標": "",
    "達成度": "",
    "備考": ""
}

DEFAULT_PRODUCT = {
    "対応理由記号": "",
    "選択制対象区分": "",
    "サービス名": "",
    "利用開始日": "",
    "商品名": "",
    "使用状況の問題1": "",
    "点検結果1": "",
    "今後の方針1": "",
    "使用状況の問題2": "",
    "点検結果2": "",
    "今後の方針2": "",
    "モニタリング備考": ""
}


def convert_audio_to_mp3(input_path: str, output_path: str):
    (
        ffmpeg
        .input(input_path)
        .output(
            output_path,
            acodec="libmp3lame",
            ac=1,
            ar="44100",
            format="mp3"
        )
        .overwrite_output()
        .run(quiet=True)
    )


def build_prompt(transcript: str) -> str:
    return f"""
# 指示
あなたは福祉用具レンタル事業所のモニタリング担当です。
次の「福祉用具レンタル事業所の担当者」と「利用者・家族・関係者」による対話記録から、
モニタリング帳票入力用のJSON形式で出力してください。

必ずJSONのみを返してください。
説明文、Markdown、コードブロックは禁止です。
すべてのキーを必ず出力してください。
値が不明な場合は空文字、配列項目は空配列で返してください。

JSON形式:
{json.dumps(DEFAULT_REPORT, ensure_ascii=False, indent=2)}

配列要素の形式:

福祉用具利用目標 の1件:
{json.dumps(DEFAULT_GOAL, ensure_ascii=False, indent=2)}

商品一覧 の1件:
{json.dumps(DEFAULT_PRODUCT, ensure_ascii=False, indent=2)}

補足:
- 福祉用具利用目標 は最大4件
- 商品一覧 は最大8件
- 達成度 は「達成」「一部達成」「未達成」のいずれかで返してください
- 使用状況の問題1 は「なし」または空文字で返してください
- 使用状況の問題2 は「あり」または空文字で返してください
- 点検結果1 は「問題なし」または空文字で返してください
- 点検結果2 は「問題あり」または空文字で返してください
- 今後の方針1 は「継続」または空文字で返してください
- 今後の方針2 は「再検討」または空文字で返してください
- 身体状況の変化1 は「なし」または空文字で返してください
- 身体状況の変化2 は「あり」または空文字で返してください
- ご家族状況の変化1 は「なし」または空文字で返してください
- ご家族状況の変化2 は「あり」または空文字で返してください
- お気持ちの変化1 は「なし」または空文字で返してください
- お気持ちの変化2 は「あり」または空文字で返してください
- 生活状況の変化1 は「なし」または空文字で返してください
- 生活状況の変化2 は「あり」または空文字で返してください
- 見直しの必要性1 は「なし」または空文字で返してください
- 見直しの必要性2 は「あり」または空文字で返してください
- Excel側で〇変換するため、上記の固定文言以外は使わないでください
- 備考欄や専門相談員による結果は、帳票向けに短く簡潔にしてください
- 会話にない内容は推測しすぎず空文字にしてください

文字起こし:
{transcript}
""".strip()


def extract_text_from_response(res) -> str:
    if hasattr(res, "output_text") and res.output_text:
        return res.output_text

    try:
        parts = []
        for out in getattr(res, "output", []):
            for content in getattr(out, "content", []):
                text_value = getattr(content, "text", None)
                if text_value:
                    parts.append(text_value)
        joined = "\n".join(parts).strip()
        if joined:
            return joined
    except Exception:
        pass

    return str(res)


def merge_dict(defaults: dict, actual: dict) -> dict:
    result = deepcopy(defaults)

    if not isinstance(actual, dict):
        return result

    for key, value in actual.items():
        result[key] = value

    return result


def normalize_report(raw: dict) -> dict:
    report = merge_dict(DEFAULT_REPORT, raw if isinstance(raw, dict) else {})

    goals = report.get("福祉用具利用目標")
    if not isinstance(goals, list):
        goals = []

    normalized_goals = []
    for g in goals[:4]:
        normalized_goals.append(
            merge_dict(DEFAULT_GOAL, g if isinstance(g, dict) else {})
        )
    report["福祉用具利用目標"] = normalized_goals

    products = report.get("商品一覧")
    if not isinstance(products, list):
        products = []

    normalized_products = []
    for p in products[:8]:
        normalized_products.append(
            merge_dict(DEFAULT_PRODUCT, p if isinstance(p, dict) else {})
        )
    report["商品一覧"] = normalized_products

    for key, value in list(report.items()):
        if value is None:
            report[key] = ""

    return report


def safe_json(text: str) -> dict:
    try:
        raw = json.loads(text)
        return normalize_report(raw)
    except Exception:
        fallback = deepcopy(DEFAULT_REPORT)
        fallback["専門相談員による結果"] = text
        return fallback


def create_summary_json(transcript: str) -> str:
    prompt = build_prompt(transcript)

    res = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
    )

    text = extract_text_from_response(res)
    report = safe_json(text)

    return json.dumps(report, ensure_ascii=False)


def transcribe_file(file_path: str):
    with open(file_path, "rb") as f:
        return client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language="ja",
        )


@app.get("/")
def root():
    return {
        "ok": True,
        "startup_error": startup_error,
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "startup_error": startup_error,
        "version": "transcribe_api_side_audio_convert_v3",
    }


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
    converted_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(data)

        print("filename =", audio.filename)
        print("content_type =", audio.content_type)
        print("size =", len(data))
        print("suffix =", suffix)

        try:
            tr = transcribe_file(tmp_path)

        except Exception as first_error:
            print("first transcription failed:", str(first_error))

            converted_path = tmp_path + ".mp3"
            convert_audio_to_mp3(tmp_path, converted_path)

            tr = transcribe_file(converted_path)

        transcript_text = tr.text or ""

        if not transcript_text.strip():
            raise HTTPException(status_code=500, detail="transcription result empty")

        summary_json = create_summary_json(transcript_text)

        return {
            "ok": True,
            "text": summary_json,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"transcribe failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except Exception:
                pass
