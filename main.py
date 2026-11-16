#!/usr/bin/env python3
"""free-model-aggregator-02 (make.fun) -- single-endpoint free-model gateway.

Serves OpenAI wire (POST /v1/chat/completions) and Anthropic wire
(POST /v1/messages) over OpenRouter's live free-model catalog, reached
through the house gateway (MAKEFUN_ENV_URL). Stdlib only, deterministic:
no wall clock, no unseeded randomness in any exported state.
"""
import json
import os
import sys
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "8080"))
GATEWAY = os.environ.get("MAKEFUN_ENV_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Vendored snapshot; refreshed live at startup when the gateway answers.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json"), encoding="utf-8") as _f:
    CATALOG = json.load(_f)

# Canonical catalog document -- --list must print this byte for byte.
CATALOG_DOC = json.dumps({"gateway": GATEWAY, "models": CATALOG}, separators=(",", ":"), sort_keys=True)

# Capability classes: ordered failover chains (best first).
TOOL_MODELS = [
    "baidu/ernie-5.5-flash-preview:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "google/gemma-4-31b-it:free",
    "dots-studio/dots-3-note-preview:free",
]
JSON_MODELS = [
    "baidu/ernie-5.5-flash-preview:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "dots-studio/dots-3-note-preview:free",
]
LONGCTX_MODELS = [
    "inclusionai/ling-3.2-flash:free",
    "thinkingmachines/inkling-2-small:free",
    "particle-ai/particle-1:free",
]
CODE_MODELS = ["cohere/north-mini-code:free", "google/gemma-4-31b-it:free"]
DEFAULT_MODELS = [
    "baidu/ernie-5.5-flash-preview:free",
    "inclusionai/ling-3.2-flash:free",
    "google/gemma-4-31b-it:free",
]

# Failover state keyed by model id: consecutive-refusal counter (no clock).
FAILS = {}
COOLDOWN_AFTER = 1  # after N refusals, park the model until chain exhausted


def pick_chain(body: dict, anthropic: bool) -> list:
    """Order candidate models for one request by declared capabilities."""
    msgs = body.get("messages") or []
    text_len = sum(len(json.dumps(m)) for m in msgs)
    if body.get("tools") or body.get("tool_choice"):
        chain = TOOL_MODELS
    elif body.get("response_format") or body.get("json_mode"):
        chain = JSON_MODELS
    elif text_len > 200_000:
        chain = LONGCTX_MODELS
    else:
        chain = DEFAULT_MODELS
    sys_text = ""
    if anthropic:
        sys_text = body.get("system") or ""
    else:
        sys_text = "".join(m.get("content", "") for m in msgs if m.get("role") == "system")
    if any(w in sys_text.lower() for w in ("code", "function", "sql", "regex")):
        chain = CODE_MODELS + [m for m in chain if m not in CODE_MODELS]
    return [m for m in chain if m in {c["id"] for c in CATALOG}]


def gateway_chat(model: str, openai_body: dict, stream: bool):
    """POST one completion to the gateway on the OpenAI wire."""
    req = urllib.request.Request(
        GATEWAY + "/chat/completions",
        data=json.dumps({**openai_body, "model": model, "stream": stream}).encode(),
        headers={
            "Authorization": "Bearer " + OPENROUTER_KEY,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        return urllib.request.urlopen(req, timeout=110)
    except urllib.error.HTTPError as e:
        FAILS[model] = FAILS.get(model, 0) + 1
        raise


def _err(anthropic: bool, msg: str) -> dict:
    if anthropic:
        return {"type": "error", "error": {"type": "api_error", "message": msg}}
    return {"error": {"message": msg, "type": "api_error", "code": None}}


def to_openai(body: dict, anthropic: bool) -> dict:
    """Normalize an Anthropic-wire body onto the OpenAI chat wire."""
    if not anthropic:
        return body
    out = {"model": body.get("model") or "", "messages": []}
    for k in ("temperature", "top_p", "max_tokens", "stop", "tools",
              "tool_choice", "response_format", "seed"):
        if body.get(k) is not None:
            out[k] = body[k]
    sys_text = body.get("system")
    if isinstance(sys_text, list):
        sys_text = "".join(b.get("text", "") for b in sys_text if isinstance(b, dict))
    if sys_text:
        out["messages"].append({"role": "system", "content": sys_text})
    for m in body.get("messages") or []:
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, str):
            out["messages"].append({"role": role, "content": c})
            continue
        text = []
        tool_calls = []
        tool_results = []
        for b in c or []:
            if not isinstance(b, dict):
                text.append(str(b))
                continue
            t = b.get("type")
            if t == "text":
                text.append(b.get("text", ""))
            elif t == "tool_use":
                tool_calls.append({"id": b.get("id"), "type": "function",
                                   "function": {"name": b.get("name") or "",
                                                "arguments": json.dumps(b.get("input") or {})}})
            elif t == "tool_result":
                rc = b.get("content")
                if isinstance(rc, list):
                    rc = "".join(x.get("text", "") for x in rc if isinstance(x, dict))
                tool_results.append({"role": "tool",
                                     "tool_call_id": b.get("tool_use_id") or "",
                                     "content": rc if isinstance(rc, str) else json.dumps(rc)})
            else:
                text.append("[%s omitted]" % t)
        if tool_calls:
            out["messages"].append({"role": "assistant",
                                    "content": "".join(text) or None,
                                    "tool_calls": tool_calls})
        elif tool_results:
            out["messages"].extend(tool_results)
            if text:
                out["messages"].append({"role": role, "content": "".join(text)})
        else:
            out["messages"].append({"role": role, "content": "".join(text)})
    return out


def to_anthropic(data: dict, model: str) -> dict:
    """Translate an OpenAI chat completion back onto the Anthropic wire."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = []
    txt = msg.get("content")
    if isinstance(txt, str) and txt:
        content.append({"type": "text", "text": txt})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        content.append({"type": "tool_use", "id": tc.get("id") or "toolu_0",
                        "name": fn.get("name") or "", "input": args})
    if not content:
        content = [{"type": "text", "text": ""}]
    out = {
        "id": "msg_" + (data.get("id") or "0").replace("/", ""),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    fr = choice.get("finish_reason")
    if fr == "stop":
        out["stop_reason"] = "end_turn"
    elif fr == "length":
        out["stop_reason"] = "max_tokens"
    elif fr == "tool_calls":
        out["stop_reason"] = "tool_use"
    usage = data.get("usage") or {}
    if usage.get("prompt_tokens") is not None:
        out["usage"] = {"input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0)}
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": c["id"], "object": "model", "owned_by": c.get("id", "").split("/")[0]}
                for c in CATALOG]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n > 0 else b"{}"
            body = json.loads(raw or b"{}")
        except Exception:
            return self._send(400, _err(False, "invalid JSON body"))
        if self.path == "/v1/chat/completions":
            anthropic = False
        elif self.path == "/v1/messages":
            anthropic = True
        else:
            return self._send(404, _err(False, "unknown route: " + self.path))
        try:
            ob = to_openai(body, anthropic)
        except Exception as e:
            return self._send(400, _err(anthropic, "bad request: %s" % e))
        stream = bool(ob.get("stream")) or bool(body.get("stream"))
        ob["stream"] = stream
        chain = pick_chain(ob, False)
        if not chain:
            return self._send(502, _err(anthropic, "no free model available"))
        last = "all chain models failed"
        for model in chain:
            if FAILS.get(model, 0) >= COOLDOWN_AFTER and model != chain[-1]:
                continue
            try:
                resp = gateway_chat(model, ob, stream)
            except urllib.error.HTTPError as e:
                last = "gateway HTTP %s" % e.code
                if e.code < 500 and e.code != 429:
                    FAILS[model] = 99
                continue
            except Exception as e:
                last = "gateway error: %s" % e
                continue
            try:
                if anthropic:
                    return self._anthropic_response(resp, model, stream)
                return self._openai_response(resp, stream)
            except Exception as e:
                last = "relay error: %s" % e
                continue
        return self._send(502, _err(anthropic, last))

    def _openai_response(self, resp, stream):
        if not stream:
            data = json.loads(resp.read().decode("utf-8", "replace"))
            return self._send(200, data)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for chunk in resp:
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception:
            pass

    def _anthropic_response(self, resp, model, stream):
        if not stream:
            data = json.loads(resp.read().decode("utf-8", "replace"))
            return self._send(200, to_anthropic(data, model))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def ev(name, obj):
            line = "event: %s\ndata: %s\n\n" % (name, json.dumps(obj, separators=(",", ":")))
            self.wfile.write(line.encode())
            self.wfile.flush()

        ev("message_start", {"type": "message_start", "message": {
            "id": "msg_0", "type": "message", "role": "assistant",
            "model": model, "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}}})
        ev("content_block_start", {"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "text", "text": ""}})
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                ch = (chunk.get("choices") or [{}])[0]
                txt = (ch.get("delta") or {}).get("content")
                if isinstance(txt, str) and txt:
                    ev("content_block_delta", {"type": "content_block_delta",
                                               "index": 0,
                                               "delta": {"type": "text_delta",
                                                         "text": txt}})
                if ch.get("finish_reason"):
                    break
        except Exception:
            pass
        ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        ev("message_delta", {"type": "message_delta",
                             "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                             "usage": {"output_tokens": 0}})
        ev("message_stop", {"type": "message_stop"})


def main() -> None:
    if "--list" in sys.argv:
        sys.stdout.write(CATALOG_DOC)
        return
    try:
        req = urllib.request.Request(
            GATEWAY + "/models",
            headers={"Authorization": "Bearer " + OPENROUTER_KEY},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            doc = json.loads(r.read().decode("utf-8", "replace"))
        live = sorted(
            (c for c in doc.get("data", []) if str(c.get("id", "")).endswith(":free")),
            key=lambda c: c["id"],
        )
        if live:
            globals()["CATALOG"] = live
            globals()["CATALOG_DOC"] = json.dumps(
                {"gateway": GATEWAY, "models": live},
                separators=(",", ":"), sort_keys=True)
    except Exception:
        pass  # vendored snapshot stays authoritative
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
