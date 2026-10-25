import json, os, threading
os.environ.update(LLM_MOCK="1", DB_PATH="/tmp/test_inv.db", PUBLIC_BASE_URL="https://host.test/a2a/")
if os.path.exists("/tmp/test_inv.db"): os.remove("/tmp/test_inv.db")
from fastapi.testclient import TestClient
import app as A
c = TestClient(A.app)
H = lambda tok="alice": {"Authorization": f"Bearer {tok}", "A2A-Version": "1.0", "Content-Type": "application/a2a+json"}
B = "/a2a"
def pkg(i, word="paid"):
    return {"packageId": f"p{i}", "vendorName": "Acme", "invoiceNumber": f"INV-{i}", "amountMinor": 1000+i, "currency": "INR",
            "doc": f"cover [C-{i}] ... [E1-{i}] [E2-{i}] [E3-{i}] {word}"}
def msg(mid, n=12, batch="b1", word="paid", **kw):
    m = {"messageId": mid, "role": "ROLE_USER", "parts": [{"mediaType": A.BATCH_MT, "data": {"batchId": batch, "policyRevision": "r1", "packages": [pkg(i, word) for i in range(n)]}}]}
    m.update(kw); return {"message": m, "configuration": {"returnImmediately": False}}

# card
r = c.get("/.well-known/agent-card.json"); assert r.status_code == 200
card = r.json(); assert card["supportedInterfaces"][0]["url"] == "https://host.test/a2a/"
# auth / version
assert c.post(f"{B}/message:send", json=msg("x"), headers={"A2A-Version":"1.0"}).status_code in (401,403)
h = H(); h["A2A-Version"] = "0.3"; assert c.post(f"{B}/message:send", json=msg("x"), headers=h).status_code == 400
# create
r = c.post(f"{B}/message:send", json=msg("m1"), headers=H()); assert r.status_code == 200, r.text
t = r.json()["task"]; assert t["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
props = t["artifacts"][0]["parts"][0]["data"]["proposals"]; assert len(props) == 12
assert len({p["actionId"] for p in props}) == 12 and all(len(p["actionId"]) >= 12 for p in props)
assert r.headers["content-type"].startswith("application/a2a+json")
# idempotent replay (reordered keys + changed returnImmediately)
m2 = msg("m1"); m2["configuration"]["returnImmediately"] = True
m2["message"]["parts"][0] = json.loads(json.dumps(m2["message"]["parts"][0], sort_keys=True))
r2 = c.post(f"{B}/message:send", json=m2, headers=H()); assert r2.json()["task"]["id"] == t["id"]
# conflict
bad = msg("m1", n=11); r3 = c.post(f"{B}/message:send", json=bad, headers=H()); assert r3.status_code == 409 and "IDEMPOTENCY_CONFLICT" in r3.text
# isolation
assert c.get(f"{B}/tasks/{t['id']}", headers=H("bob")).status_code in (403, 404)
assert c.get(f"{B}/tasks", headers=H("bob")).json() == {"tasks": []}
assert c.post(f"{B}/tasks/{t['id']}:cancel", headers=H("bob")).status_code in (403, 404)
assert t["id"] not in c.post(f"{B}/tasks/{t['id']}:cancel", headers=H("bob")).text
assert len(c.get(f"{B}/tasks", headers=H()).json()["tasks"]) == 1
# continuation: mismatch rejected
def res(task, mid, accept_n=8, **over):
    pr = task["artifacts"][0]["parts"][0]["data"]
    results = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                "outcome": "ACCEPTED" if i < accept_n else "REJECTED", "receiptNonce": f"n{i}x9"} for i, p in enumerate(pr["proposals"])]
    d = {"batchId": pr["batchId"], "results": results}; d.update(over)
    return {"message": {"messageId": mid, "taskId": task["id"], "contextId": task["contextId"], "role": "ROLE_USER",
                        "parts": [{"mediaType": A.RES_MT, "data": d}]}}
badres = res(t, "r0"); badres["message"]["parts"][0]["data"]["results"][0]["actionId"] = "wrong-action-id"
assert c.post(f"{B}/message:send", json=badres, headers=H()).status_code == 400
assert c.post(f"{B}/message:send", json=res(t, "r0"), headers=H("bob")).status_code in (403, 404)
assert c.get(f"{B}/tasks/{t['id']}", headers=H()).json()["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
r = c.post(f"{B}/message:send", json=res(t, "r1"), headers=H()); assert r.status_code == 200
done = r.json()["task"]; assert done["status"]["state"] == "TASK_STATE_COMPLETED"
assert len(done["artifacts"]) == 2 and len(done["artifacts"][1]["parts"]) == 1
ex = done["artifacts"][1]["parts"][0]["data"]["executions"]; assert len(ex) == 8
assert len(done["history"]) == 2
# replay of result + cancel after terminal
assert c.post(f"{B}/message:send", json=res(t, "r1"), headers=H()).json()["task"] == done
assert c.post(f"{B}/tasks/{t['id']}:cancel", headers=H()).status_code == 409
assert c.get(f"{B}/tasks/{t['id']}", headers=H()).json() == done
# cancel path, then result must 409 and no receipts
t2 = c.post(f"{B}/message:send", json=msg("m2", batch="b2"), headers=H()).json()["task"]
assert c.post(f"{B}/tasks/{t2['id']}:cancel", headers=H()).status_code == 200
assert c.post(f"{B}/message:send", json=res(t2, "r2"), headers=H()).status_code == 409
fin = c.get(f"{B}/tasks/{t2['id']}", headers=H()).json()
assert fin["status"]["state"] == "TASK_STATE_CANCELED" and len(fin["artifacts"]) == 1
# race: result vs cancel, exactly one wins
for k in range(5):
    t3 = c.post(f"{B}/message:send", json=msg(f"m3{k}", batch=f"b3{k}"), headers=H()).json()["task"]
    out = []
    def a(): out.append(("res", c.post(f"{B}/message:send", json=res(t3, f"rr{k}"), headers=H()).status_code))
    def b(): out.append(("can", c.post(f"{B}/tasks/{t3['id']}:cancel", headers=H()).status_code))
    ths = [threading.Thread(target=a), threading.Thread(target=b)]; [x.start() for x in ths]; [x.join() for x in ths]
    assert sorted(s for _, s in out) == [200, 409], out
# concurrent identical requests -> one task
calls = {"n": 0}; orig = A.call_llm
def counting(items): calls["n"] += 1; return orig(items)
A.call_llm = counting
rs = []
def go(): rs.append(c.post(f"{B}/message:send", json=msg("m4", batch="b4", word="fresh-content"), headers=H()).json()["task"]["id"])
ths = [threading.Thread(target=go) for _ in range(6)]; [x.start() for x in ths]; [x.join() for x in ths]
assert len(set(rs)) == 1 and calls["n"] == 1, (rs, calls)
# cache: same package content, new batch/message -> no model call
before = calls["n"]
c.post(f"{B}/message:send", json=msg("m5", batch="b5"), headers=H("carol")); assert calls["n"] == before
print("ALL TESTS PASSED")
