"""
Personal Expense Agent (Streamlit version)
Nebius x NVIDIA Global AI Hackathon - Personal AI track

The agent reads natural-language messages and decides what to do:
  add_expenses | query | summary | chat
Model: an NVIDIA Nemotron model on Nebius Token Factory.

Secrets (Streamlit Cloud -> App settings -> Secrets), TOML format:
  NEBIUS_API_KEY = "your_key"
  MODEL_ID = "exact-nemotron-model-id"   # optional
"""

import os
import re
import json
import datetime
from collections import defaultdict

import streamlit as st
from openai import OpenAI

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
CURRENCY = "Tk"


def secret(name, default=""):
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


API_KEY = secret("NEBIUS_API_KEY")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "missing")


@st.cache_resource
def get_model():
    model_id = str(secret("MODEL_ID")).strip()
    if model_id:
        return model_id
    models = [m.id for m in client.models.list().data]
    nemo = [m for m in models if "nemotron" in m.lower()]
    if not nemo:
        raise RuntimeError("No Nemotron model found. Set MODEL_ID in Secrets.")
    return nemo[0]


def llm(messages, temperature=0.2, max_tokens=800):
    resp = client.chat.completions.create(
        model=get_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


ROUTER_PROMPT = f"""You are the brain of a personal expense tracking agent.
Today's date is {{today}}. Currency is {CURRENCY}.
Read the user's message and reply with ONLY a JSON object (no extra text) in this shape:

{{{{
  "action": "add_expenses" | "query" | "summary" | "chat",
  "expenses": [{{{{"amount": number, "category": string, "note": string}}}}],
  "filter": {{{{"category": string or null, "days": number or null}}}},
  "reply": string
}}}}

Rules:
- add_expenses: user reports money spent. Put each item in "expenses".
  Use short lowercase categories: food, transport, bills, shopping, health,
  education, entertainment, other. Amount must be a number.
- query: user asks about spending. Set "filter": category (or null) and days
  (7 for week, 30 for month, null for all time).
- summary: user wants an overview or saving tips. Use filter days (default 7).
- chat: anything else. Put a short helpful answer in "reply".
- Understand English and Bangla. Reply in the same language the user used.
"""


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found")
    return json.loads(text[start : end + 1])


def tool_add(db, items):
    today = datetime.date.today().isoformat()
    added = []
    for it in items:
        try:
            amount = float(it.get("amount"))
        except (TypeError, ValueError):
            continue
        row = {
            "date": today,
            "category": str(it.get("category", "other")).lower().strip() or "other",
            "amount": amount,
            "note": str(it.get("note", "")).strip(),
        }
        db.append(row)
        added.append(row)
    return added


def tool_filter(db, category=None, days=None):
    rows = db
    if days:
        cutoff = datetime.date.today() - datetime.timedelta(days=int(days) - 1)
        rows = [r for r in rows if datetime.date.fromisoformat(r["date"]) >= cutoff]
    if category:
        rows = [r for r in rows if r["category"] == str(category).lower().strip()]
    return rows


def tool_totals(rows):
    by_cat = defaultdict(float)
    for r in rows:
        by_cat[r["category"]] += r["amount"]
    return {
        "total": round(sum(by_cat.values()), 2),
        "by_category": {k: round(v, 2) for k, v in sorted(by_cat.items(), key=lambda x: -x[1])},
        "count": len(rows),
    }


def agent_step(user_msg, db):
    today = datetime.date.today().isoformat()
    raw = llm(
        [
            {"role": "system", "content": ROUTER_PROMPT.format(today=today)},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    try:
        plan = parse_json(raw)
    except Exception:
        return "Sorry, I could not understand that. Try: 'spent 500 on food'."

    action = plan.get("action", "chat")

    if action == "add_expenses":
        added = tool_add(db, plan.get("expenses") or [])
        if not added:
            return "I couldn't find an amount. Try: 'spent 200 on bus'."
        lines = [
            f"- {r['category']}: {r['amount']:g} {CURRENCY}" + (f" ({r['note']})" if r["note"] else "")
            for r in added
        ]
        total = sum(r["amount"] for r in added)
        return "Saved:\n" + "\n".join(lines) + f"\n\nTotal added: {total:g} {CURRENCY}"

    if action in ("query", "summary"):
        f = plan.get("filter") or {}
        days = f.get("days") or (7 if action == "summary" else None)
        rows = tool_filter(db, f.get("category"), days)
        if not rows:
            return "No expenses found for that period yet."
        facts = tool_totals(rows)
        # Numbers come from Python (accurate); the model only explains them.
        return llm(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly personal finance assistant. Use ONLY the numbers given. "
                        f"Currency: {CURRENCY}. Keep the answer short (max 6 lines). "
                        "For a summary, add 2 practical saving tips. "
                        "Reply in the same language as the user's question."
                    ),
                },
                {"role": "user", "content": f"Question: {user_msg}\nData: {json.dumps(facts)}"},
            ],
            temperature=0.3,
        )

    return plan.get("reply") or "Tell me what you spent, e.g. 'spent 500 on food'."


# ---------- UI ----------

st.set_page_config(page_title="Personal Expense Agent", page_icon="💰")
st.title("💰 Personal Expense Agent")
st.caption(
    "Type your spending naturally (English or Bangla). Examples: "
    "'spent 500 on food and 200 on bus' | 'how much on food this week?' | 'weekly summary'"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "expenses" not in st.session_state:
    st.session_state.expenses = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("e.g. spent 300 on lunch")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not API_KEY:
            reply = "NEBIUS_API_KEY is missing. Add it in App settings -> Secrets."
        else:
            try:
                with st.spinner("Thinking..."):
                    reply = agent_step(prompt, st.session_state.expenses)
            except Exception as e:
                reply = f"Error: {e}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

if st.session_state.expenses:
    st.subheader("Your expenses (private to this session)")
    st.dataframe(st.session_state.expenses, use_container_width=True)
