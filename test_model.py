from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import os
import re
from dotenv import load_dotenv

# ==============================
# ENV SETUP
# ==============================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

# ==============================
# APP SETUP
# ==============================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==============================
# OPENROUTER CONFIG
# ==============================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

# ==============================
# LANGUAGE DETECTION
# ==============================
def is_tamil(text: str) -> bool:
    return any('\u0B80' <= ch <= '\u0BFF' for ch in text)

# ==============================
# SYSTEM PROMPTS (UNCHANGED)
# ==============================

TAMIL_PROMPT = """You are a Tamil Educational Keyword Extraction Assistant.

Read a Tamil paragraph and generate:
1) One short Tamil title
2) Level-1 keywords (main concepts)
3) Level-2 keywords (sub-concepts) for each Level-1 keyword

TEXT RULES
- Normalize text internally.
- Do NOT select stopwords: இந்த, அந்த, இது, அது, என்கிற, ஆனால், என்று, எனும், மற்றும், மூலம், உள்ள, ஒரு, பற்றி, போன்ற, ஆகிய, இருந்து, மீது
- Select only nouns or noun phrases (1–3 Tamil words).
- No verbs.
- No filler phrases.
- No invented words.
- All keywords must appear exactly as in the paragraph.

LEVEL-1 KEYWORDS
- Extract 8 to 12 main keywords.
- Prefer names, places, events, roles, achievements.
- Keep keywords short and clear.

LEVEL-2 KEYWORDS
- For each Level-1 keyword, extract 2 to 4 related sub-keywords.
- Sub-keywords must be directly related to the parent keyword.
- Sub-keywords must appear exactly as in the paragraph.
- Do NOT repeat Level-1 keywords.
- Do NOT mix sub-keywords between different parents.

YEAR RULE
- A year must never appear alone.
- A year is allowed only when attached to an event or action
  (e.g., டி20 உலகக் கோப்பை 2007, 2024 ஓய்வு).

TITLE RULES
- One Tamil title (2–5 words).
- Reflect the main idea.
- No stopwords.

OUTPUT FORMAT (JSON ONLY)
{
  "title": "<Tamil title>",
  "keywords": [
    {
      "level1": "<main keyword>",
      "level2": ["<sub keyword 1>", "<sub keyword 2>"]
    }
  ]
}

IMPORTANT
- Do not summarize the paragraph.
- Do not explain anything.
- Output only the JSON.
"""

ENGLISH_PROMPT = """You are an English Educational Keyword Extraction Assistant.

Read an English paragraph and generate:
1) One short English title
2) Level-1 keywords (main concepts)
3) Level-2 keywords (sub-concepts) for each Level-1 keyword

TEXT RULES
- Normalize text internally.
- Do NOT select stopwords: the, is, of, and, to, in, for, with, on, by.
- Select only nouns or noun phrases (1–3 words).
- Avoid verbs.
- No filler phrases.
- No invented words.
- Keywords must appear in the paragraph.

LEVEL-1 KEYWORDS
- Extract 8 to 12 main keywords.
- Prefer names, places, events, roles, achievements.
- Keep keywords short and clear.

LEVEL-2 KEYWORDS
- For each Level-1 keyword, extract 2 to 4 related sub-keywords.
- Sub-keywords must be directly related to the parent keyword.
- Sub-keywords must appear in the paragraph.

TITLE RULES
- One English title (2–5 words).
- Reflect the main idea.
- No stopwords.

OUTPUT FORMAT (JSON ONLY)
{
  "title": "<English title>",
  "keywords": [
    {
      "level1": "<main keyword>",
      "level2": ["<sub keyword 1>", "<sub keyword 2>"]
    }
  ]
}

IMPORTANT
- Do not summarize the paragraph.
- Do not explain anything.
- Output only the JSON.
"""

# ==============================
# HEALTH CHECK
# ==============================
@app.route("/health")
def health():
    return {"status": "ok"}

# ==============================
# MAIN API
# ==============================
@app.route("/extract_keywords", methods=["POST"])
def extract_keywords():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON request"}), 400

        text = data.get("text") or data.get("paragraph")
        if not text:
            return jsonify({"error": "Missing text or paragraph field"}), 400

        system_prompt = TAMIL_PROMPT if is_tamil(text) else ENGLISH_PROMPT

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Tamil-English Keyword Extractor"
        }

        r = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if r.status_code != 200:
            return jsonify({
                "error": "OpenRouter API error",
                "details": r.text
            }), 500

        response_data = r.json()
        content = response_data["choices"][0]["message"]["content"].strip()

        # Remove markdown blocks ONCE
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1]).strip()

        # Extract JSON safely
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return jsonify({
                "error": "No valid JSON found",
                "raw_output": content
            }), 500

        result = json.loads(match.group())

        # ✅ NORMALIZATION (THIS FIXES MISSING TAMIL KEYWORDS)
        final_output = {
            "title": result.get("title", "Untitled"),
            "keywords": []
        }

        for item in result.get("keywords", []):
            if isinstance(item, dict):
                level1 = item.get("level1")
                level2 = item.get("level2", [])
                if isinstance(level1, str) and isinstance(level2, list):
                    final_output["keywords"].append({
                        "level1": level1,
                        "level2": [x for x in level2 if isinstance(x, str)]
                    })

        return jsonify(final_output), 200

    except Exception as e:
        return jsonify({
            "error": "Backend error",
            "details": str(e)
        }), 500

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
