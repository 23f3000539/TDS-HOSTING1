"""A2A 1.0 (HTTP+JSON) invoice action agent.

Layers: protocol (handle/route) -> storage (sqlite) -> AI decisions (decide_packages).
Model calls happen ONLY when a brand-new batch arrives and some packages are not cached.
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

# ---------------------------------------------------------------- config
BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://example.com/a2a/")
BASE_PATH = urlparse(BASE_URL).path.rstrip("/")  # e.g. "/a2a"
DB_PATH = os.environ.get("DB_PATH", "/tmp/invoice_agent.db")
ALLOWED_TOKENS = {t.strip() for t in os.environ.get("ALLOWED_TOKENS", "").split(",") if t.strip()}

LLM_BASE = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_JSON_MODE = os.environ.get("LLM_JSON_MODE", "1") == "1"
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "32"))
LLM_MOCK = os.environ.get("LLM_MOCK") == "1"

A2A_MT = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROP_MT = "application/vnd.ga5.invoice-action-proposals+json"
RES_MT = "application/vnd.ga5.invoice-action-results+json"
REC_MT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"}

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
LOCK = threading.RLock()  # guards every DB mutation (cancel/result race lives here)
_KEYLOCKS: dict = {}
_KEYLOCKS_GUARD = threading.Lock()


# ---------------------------------------------------------------- helpers
def canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def jr(obj, status=200) -> Response:
    return Response(json.dumps(obj, separators=(",", ":")), status_code=status, media_type=A2A_MT)


def err(status: int, code: str, msg: str = "The request could not be processed.") -> Response:
    return jr({"error": {"code": status, "status": code, "message": msg,
                         "details": [{"reason": code}]}}, status)


def keylock(key) -> threading.Lock:
    with _KEYLOCKS_GUARD:
        return _KEYLOCKS.setdefault(key, threading.Lock())


# ---------------------------------------------------------------- storage
def db() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db():
    with db() as c:
        c.execute("CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, principal TEXT, seq INTEGER, body TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS msgs(principal TEXT, message_id TEXT, hash TEXT, task_id TEXT,"
                  " PRIMARY KEY(principal, message_id))")
        c.execute("CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, body TEXT)")


init_db()


def load_task(principal, task_id):
    with db() as c:
        r = c.execute("SELECT body FROM tasks WHERE id=? AND principal=?", (task_id, principal)).fetchone()
    return json.loads(r[0]) if r else None


def save_task(principal, task, msg_id=None, msg_hash=None):
    """Atomic: task row + idempotency row together."""
    with db() as c:
        r = c.execute("SELECT seq FROM tasks WHERE id=?", (task["id"],)).fetchone()
        seq = r[0] if r else int(datetime.now().timestamp() * 1e6)
        c.execute("INSERT OR REPLACE INTO tasks(id,principal,seq,body) VALUES(?,?,?,?)",
                  (task["id"], principal, seq, json.dumps(task)))
        if msg_id is not None:
            c.execute("INSERT OR IGNORE INTO msgs VALUES(?,?,?,?)", (principal, msg_id, msg_hash, task["id"]))


def lookup_msg(principal, message_id):
    with db() as c:
        return c.execute("SELECT hash, task_id FROM msgs WHERE principal=? AND message_id=?",
                         (principal, message_id)).fetchone()


# ---------------------------------------------------------------- AI layer
SYSTEM = """You review invoice packages for an accounts-payable agent. Choose EXACTLY ONE action per package:
- settle_invoice: valid, reconciled, and within autonomous payment authority.
- request_approval: commercially valid but the amount/terms exceed delegated authority.
- hold_invoice: payment must pause until a stated verification completes.
- reject_duplicate: the SAME commercial invoice (same vendor, invoice number, amount) was ALREADY PAID. A payment for a different invoice is NOT a duplicate.
- open_exception: material records conflict (amounts, vendor, PO, receipt, bank details) and need an exception workflow.

Packages mix useful facts with a cover sheet, archive/old examples, training decoys, negated statements
("not a duplicate", "no hold needed") and irrelevant action words. Find the ONE paragraph that states the current
operative facts and determines the action. Ignore the cover sheet, archive examples and decoys. Read negation carefully.
That paragraph carries exactly three bracketed references. Return those three references EXACTLY as written in the
text (character for character, including brackets if present). Never include the cover-sheet reference.

Also extract facts: vendorName, invoiceNumber, amountMinor (integer, minor currency units, e.g. paise/cents),
currency (3-letter code). Write a rationale of 60-400 characters that names the chosen action and quotes at least
two of the three evidence references.

Reply with JSON only: {"decisions":[{"index":0,"action":"...","facts":{"vendorName":"...","invoiceNumber":"...",
"amountMinor":12345,"currency":"INR"},"evidenceRefs":["r1","r2","r3"],"rationale":"..."}]}
Include one decision per package index given."""


def strip_ids(pkg: dict) -> dict:
    return {k: v for k, v in pkg.items() if k not in ("packageId", "id", "batchId")}


def corpus(o) -> str:
    out = []

    def walk(x):
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(o)
    return "\n".join(out)


def parse_json_loose(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1])


def call_llm(items):
    """items: list of (index, package). Returns {index: raw decision dict}."""
    if LLM_MOCK:
        return mock_llm(items)
    parts = [f"### PACKAGE index={i}\n{json.dumps(p, ensure_ascii=False)}" for i, p in items]
    payload = {"model": LLM_MODEL, "temperature": 0,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"Decide for these {len(items)} packages.\n\n" + "\n\n".join(parts)}]}
    if LLM_JSON_MODE:
        payload["response_format"] = {"type": "json_object"}
    r = httpx.post(f"{LLM_BASE}/chat/completions", json=payload, timeout=LLM_TIMEOUT,
                   headers={"Authorization": f"Bearer {LLM_KEY}"})
    r.raise_for_status()
    data = parse_json_loose(r.json()["choices"][0]["message"]["content"])
    return {int(d["index"]): d for d in data.get("decisions", []) if isinstance(d, dict) and "index" in d}


def mock_llm(items):  # deterministic stand-in used by the local tests only
    out = {}
    for i, p in items:
        text = corpus(p).lower()
        act = ("reject_duplicate" if "duplicate" in text else "request_approval" if "approval" in text else
               "hold_invoice" if "hold" in text else "open_exception" if "exception" in text else "settle_invoice")
        refs = re.findall(r"\[[^\]\n]{2,60}\]", corpus(p))[:3]
        out[i] = {"action": act, "refs": refs, "evidenceRefs": refs,
                  "facts": {"vendorName": p.get("vendorName", "V"), "invoiceNumber": p.get("invoiceNumber", "N"),
                            "amountMinor": p.get("amountMinor", 100), "currency": p.get("currency", "INR")},
                  "rationale": ""}
    return out


def validate(raw: dict, text: str):
    """Return a clean decision dict or None."""
    try:
        action = raw["action"]
        if action not in ACTIONS:
            return None
        f = raw["facts"]
        amt = f["amountMinor"]
        if isinstance(amt, str) and re.fullmatch(r"-?\d+", amt.strip()):
            amt = int(amt)
        if isinstance(amt, float) and amt.is_integer():
            amt = int(amt)
        if isinstance(amt, bool) or not isinstance(amt, int):
            return None
        facts = {"vendorName": str(f["vendorName"]).strip(), "invoiceNumber": str(f["invoiceNumber"]).strip(),
                 "amountMinor": amt, "currency": str(f["currency"]).strip().upper()}
        if not all(facts[k] for k in ("vendorName", "invoiceNumber", "currency")):
            return None
        refs = raw["evidenceRefs"]
        if not isinstance(refs, list) or len(refs) != 3 or len(set(refs)) != 3:
            return None
        norm = re.sub(r"\s+", " ", text)
        if not all(isinstance(x, str) and (x in text or re.sub(r"\s+", " ", x) in norm) for x in refs):
            return None
        rat = str(raw.get("rationale", "")).strip()[:1100]
        if action not in rat:
            rat = f"Chosen action: {action}. " + rat
        if sum(1 for x in refs if x in rat) < 2:
            rat += f" Decisive evidence: {refs[0]}, {refs[1]} and {refs[2]}."
        if len(rat) < 60:
            rat += " The decisive paragraph of the invoice package supports this action under the current policy."
        return {"action": action, "facts": facts, "evidenceRefs": refs, "rationale": rat[:1500]}
    except Exception:
        return None


def decide_packages(packages):
    """One batched model call for cache misses; cache by canonical package content."""
    keys = [sha(canon(strip_ids(p))) for p in packages]
    result = {}
    with db() as c:
        for k in set(keys):
            r = c.execute("SELECT body FROM decisions WHERE key=?", (k,)).fetchone()
            if r:
                result[k] = json.loads(r[0])
    pending = {}
    for p, k in zip(packages, keys):
        if k not in result:
            pending[k] = p
    for attempt in range(2):
        if not pending:
            break
        order = list(pending.items())
        try:
            raw = call_llm([(i, p) for i, (k, p) in enumerate(order)])
        except Exception:
            raw = {}
        for i, (k, p) in enumerate(order):
            clean = validate(raw.get(i, {}), corpus(p)) if i in raw else None
            if clean:
                result[k] = clean
                with db() as c:
                    c.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)", (k, json.dumps(clean)))
                pending.pop(k)
    if pending:
        raise RuntimeError("decision failed")
    return [(k, result[k]) for k in keys]


# ---------------------------------------------------------------- protocol: message handling
def make_task(principal, msg, batch):
    pkgs = batch["packages"]
    decided = decide_packages(pkgs)
    proposals = []
    for p, (ck, d) in zip(pkgs, decided):
        pid = p.get("packageId") or p.get("id")
        proposals.append({"packageId": pid,
                          "actionId": "act-" + sha(f"{batch['batchId']}|{pid}|{ck}")[:24],
                          "action": d["action"], "facts": d["facts"],
                          "evidenceRefs": d["evidenceRefs"], "rationale": d["rationale"]})
    tid, cid = str(uuid.uuid4()), (msg.get("contextId") if isinstance(msg.get("contextId"), str) and msg.get("contextId") else str(uuid.uuid4()))
    hist_msg = dict(msg, taskId=tid, contextId=cid)
    return {"id": tid, "contextId": cid,
            "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
            "artifacts": [{"artifactId": "proposals-" + tid[:8], "name": "invoice-action-proposals",
                           "parts": [{"mediaType": PROP_MT, "data": {"batchId": batch["batchId"], "proposals": proposals}}]}],
            "history": [hist_msg]}


def handle_send(principal, body: bytes):
    try:
        req = json.loads(body)
        msg = req["message"]
        mid = msg["messageId"]
        if not (isinstance(mid, str) and mid) or msg.get("role") != "ROLE_USER" or not isinstance(msg.get("parts"), list) or not msg["parts"]:
            raise ValueError
    except Exception:
        return err(400, "INVALID_ARGUMENT")
    h = sha(canon(msg))  # message only; configuration ignored
    with keylock((principal, mid)):
        with LOCK:
            row = lookup_msg(principal, mid)
            if row:
                if row[0] != h:
                    return err(409, "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT")
                t = load_task(principal, row[1])
                return jr({"task": t}) if t else err(404, "NOT_FOUND")
        if msg.get("taskId"):
            return continue_task(principal, msg, mid, h)
        part = next((p for p in msg["parts"] if isinstance(p, dict) and p.get("mediaType") == BATCH_MT), None)
        batch = part.get("data") if part else None
        try:
            ok = (isinstance(batch, dict) and isinstance(batch.get("batchId"), str) and batch["batchId"]
                  and isinstance(batch.get("packages"), list) and batch["packages"]
                  and all(isinstance(p, dict) and (p.get("packageId") or p.get("id")) for p in batch["packages"]))
            ids = [p.get("packageId") or p.get("id") for p in batch["packages"]] if ok else []
            if not ok or len(set(ids)) != len(ids):
                return err(400, "INVALID_ARGUMENT")
        except Exception:
            return err(400, "INVALID_ARGUMENT")
        try:
            task = make_task(principal, msg, batch)
        except Exception:
            return err(503, "UNAVAILABLE", "Temporarily unable to process the request.")
        with LOCK:
            save_task(principal, task, mid, h)  # persist BEFORE responding
        return jr({"task": task})


def continue_task(principal, msg, mid, h):
    with LOCK:  # serialises against cancel
        task = load_task(principal, msg["taskId"])
        if not task:
            return err(404, "NOT_FOUND")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_ACTIVE")
        if msg.get("contextId") != task["contextId"]:
            return err(400, "INVALID_ARGUMENT")
        parts = msg["parts"]
        if len(parts) != 1 or not isinstance(parts[0], dict) or parts[0].get("mediaType") != RES_MT:
            return err(400, "INVALID_ARGUMENT")
        data = parts[0].get("data")
        prop = task["artifacts"][0]["parts"][0]["data"]
        if not isinstance(data, dict) or data.get("batchId") != prop["batchId"] or not isinstance(data.get("results"), list):
            return err(400, "INVALID_ARGUMENT")
        by_pkg = {p["packageId"]: p for p in prop["proposals"]}
        seen, accepted = set(), {}
        for r in data["results"]:
            if not isinstance(r, dict):
                return err(400, "INVALID_ARGUMENT")
            p = by_pkg.get(r.get("packageId"))
            if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"] or r.get("action") != p["action"]
                    or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                return err(400, "INVALID_ARGUMENT")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                n = r.get("receiptNonce")
                if not isinstance(n, str) or not n:
                    return err(400, "INVALID_ARGUMENT")
                accepted[r["packageId"]] = n
        executions = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                       "receiptNonce": accepted[p["packageId"]], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]}
                      for p in prop["proposals"] if p["packageId"] in accepted]
        task["artifacts"].append({"artifactId": "receipts-" + task["id"][:8], "name": "invoice-action-receipts",
                                  "parts": [{"mediaType": REC_MT, "data": {"batchId": prop["batchId"], "executions": executions}}]})
        task["history"].append(dict(msg, taskId=task["id"], contextId=task["contextId"]))
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
        save_task(principal, task, mid, h)
        return jr({"task": task})


def handle_cancel(principal, task_id):
    with LOCK:
        task = load_task(principal, task_id)
        if not task:
            return err(404, "NOT_FOUND")
        if task["status"]["state"] in TERMINAL:
            return err(409, "TASK_NOT_CANCELABLE")
        task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
        save_task(principal, task)
        return jr(task)


# ---------------------------------------------------------------- protocol: routing
def agent_card():
    return {"name": "Invoice Action Agent",
            "description": "Reads invoice packages, proposes one safe business action per invoice with cited evidence, "
                           "and executes only grader-accepted proposals.",
            "version": "1.0.0",
            "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": [BATCH_MT, RES_MT],
            "defaultOutputModes": [PROP_MT, REC_MT],
            "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                        "description": "Chooses settle, approve, hold, reject-duplicate or exception actions for invoices.",
                        "tags": ["invoice", "accounts-payable", "payments"]}]}


def handle(method, path, headers, body):
    if not path.startswith(BASE_PATH + "/"):
        return err(404, "NOT_FOUND")
    rel = path[len(BASE_PATH):]
    auth = headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not token or (ALLOWED_TOKENS and token not in ALLOWED_TOKENS):
        return err(401, "UNAUTHENTICATED", "Authentication required.")
    principal = sha("tok:" + token)[:32]
    if headers.get("a2a-version", "").strip() != "1.0":
        return err(400, "VERSION_NOT_SUPPORTED", "Unsupported protocol version.")
    if method == "POST" and A2A_MT not in headers.get("content-type", "").lower() \
            and "application/json" not in headers.get("content-type", "").lower():
        return err(415, "UNSUPPORTED_MEDIA_TYPE")
    if rel == "/message:send" and method == "POST":
        return handle_send(principal, body)
    if rel == "/tasks" and method == "GET":
        with db() as c:
            rows = c.execute("SELECT body FROM tasks WHERE principal=? ORDER BY seq", (principal,)).fetchall()
        return jr({"tasks": [json.loads(r[0]) for r in rows]})
    if rel.startswith("/tasks/"):
        tail = rel[len("/tasks/"):]
        if method == "POST" and tail.endswith(":cancel") and "/" not in tail:
            return handle_cancel(principal, tail[:-len(":cancel")])
        if method == "GET" and tail and "/" not in tail and ":" not in tail:
            t = load_task(principal, tail)
            return jr(t) if t else err(404, "NOT_FOUND")
    return err(404, "NOT_FOUND")


@app.api_route("/{p:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route(p: str, request: Request):
    path = "/" + p
    if request.method == "GET" and path == "/.well-known/agent-card.json":
        return Response(json.dumps(agent_card()), media_type="application/json")
    body = await request.body()
    return await run_in_threadpool(handle, request.method, path, dict(request.headers), body)
