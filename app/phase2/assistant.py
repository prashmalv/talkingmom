"""Governed data assistant: question → whitelisted query → chart + answer.

GPT never writes SQL and never supplies numbers. It only routes a question
to one of the fixed query types below and picks validated parameters; Python
computes every figure from the sample dataset; GPT then phrases those figures
in the asker's language. Each answer records its query (auditable) and the
data source.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ..llm_client import llm_chat, llm_json
from .sample_data import (
    DISTRICTS,
    DISTRICT_NAMES,
    DIVISIONS,
    HEADLINE,
    METRICS,
    MONTHS,
    SCHEMES,
    SOURCE_NOTE,
    facts,
    month_label,
)

LATEST, PREV = MONTHS[-1], MONTHS[-2]
_ALIASES = {a.lower(): d[0] for d in DISTRICTS for a in [d[0], d[1], *d[3]]}


# ---------------------------------------------------------------- helpers

def _fmt(v: float, unit: str) -> str:
    if unit == "%":
        return f"{v:.1f}%"
    return f"{int(round(v)):,}"


def _series(metric: str):
    df = facts()
    return df[df.metric == metric]


def _state(metric: str, month) -> float:
    s = _series(metric)
    s = s[s.month == month].value
    return float(s.mean() if METRICS[metric][4] == "mean" else s.sum())


def _value(metric: str, district: str, month=LATEST) -> float:
    s = _series(metric)
    return float(s[(s.district == district) & (s.month == month)].value.iloc[0])


def _rank(metric: str, district: str) -> tuple[int, int]:
    s = _series(metric)
    s = s[s.month == LATEST].set_index("district").value
    order = s.sort_values(ascending=not METRICS[metric][3])
    return list(order.index).index(district) + 1, len(order)


def _base(kind: str, title: str, subtitle: str = "") -> dict:
    return {"kind": kind, "title": title, "subtitle": subtitle,
            "as_of": month_label(LATEST), "source": SOURCE_NOTE}


# ---------------------------------------------------------------- queries

def state_overview(scheme: str | None = None) -> dict:
    metrics = HEADLINE[scheme] if scheme else [HEADLINE[s][0] for s in SCHEMES]
    kpis = []
    for m in metrics:
        sch, label, unit, hib, _ = METRICS[m]
        now, before = _state(m, LATEST), _state(m, PREV)
        delta = now - before
        kpis.append({"label": label, "scheme": SCHEMES[sch], "value": _fmt(now, unit), "raw": now,
                     "delta": (f"{delta:+.1f} pts" if unit == "%" else f"{delta:+,.0f}") + " vs last month",
                     "good": (delta >= 0) == hib})
    title = f"Maharashtra — {SCHEMES[scheme]}" if scheme else "Maharashtra — state overview"
    v = _base("kpis", title, "State level, all 36 districts")
    v["kpis"] = kpis
    v["facts"] = {k["label"]: k["value"] for k in kpis}
    return v


def district_status(district: str, scheme: str | None = None) -> dict:
    metrics = HEADLINE[scheme] if scheme else [HEADLINE[s][0] for s in SCHEMES]
    kpis = []
    for m in metrics:
        sch, label, unit, hib, _ = METRICS[m]
        val = _value(m, district)
        rank, total = _rank(m, district)
        state_avg = _state(m, LATEST) if METRICS[m][4] == "mean" else _state(m, LATEST) / len(DISTRICT_NAMES)
        kpis.append({"label": label, "scheme": SCHEMES[sch], "value": _fmt(val, unit), "raw": val,
                     "delta": f"State avg {_fmt(state_avg, unit)} · rank {rank}/{total}",
                     "good": (val >= state_avg) == hib})
    division = next(d[2] for d in DISTRICTS if d[0] == district)
    v = _base("kpis", f"{district} district" + (f" — {SCHEMES[scheme]}" if scheme else ""),
              f"{division} division")
    v["kpis"] = kpis
    v["facts"] = {k["label"]: f"{k['value']} ({k['delta']})" for k in kpis}
    return v


def compare(districts: list[str], metric: str) -> dict:
    sch, label, unit, hib, _ = METRICS[metric]
    vals = [_value(metric, d) for d in districts]
    v = _base("bar", f"{label} — district comparison", SCHEMES[sch])
    avg = _state(metric, LATEST) if METRICS[metric][4] == "mean" else None
    v["bar"] = {"labels": districts, "values": vals, "unit": unit, "reference": avg,
                "reference_label": "State average" if avg is not None else None}
    best = (max if hib else min)(zip(vals, districts))[1]
    v["facts"] = {d: _fmt(x, unit) for d, x in zip(districts, vals)} | {"best": best}
    if avg is not None:
        v["facts"]["state average"] = _fmt(avg, unit)
    return v


def ranking(metric: str, order: str = "worst", n: int = 5, division: str | None = None,
            by: str = "district") -> dict:
    sch, label, unit, hib, agg = METRICS[metric]
    s = _series(metric)
    s = s[s.month == LATEST]
    if division:
        s = s[s.division == division]
    if by == "division":
        g = s.groupby("division").value
        s = (g.mean() if agg == "mean" else g.sum())
    else:
        s = s.set_index("district").value
    best_first = s.sort_values(ascending=not hib)
    picked = best_first.head(n) if order == "best" else best_first.tail(n)[::-1]
    word = "Top" if order == "best" else "Bottom"
    scope = f" in {division} division" if division else ""
    v = _base("bar", f"{word} {len(picked)} {'divisions' if by == 'division' else 'districts'}{scope} — {label}",
              SCHEMES[sch] + (" · lower is better" if not hib else ""))
    v["bar"] = {"labels": list(picked.index), "values": [float(x) for x in picked.values], "unit": unit,
                "reference": None, "horizontal": True}
    v["facts"] = {k: _fmt(x, unit) for k, x in picked.items()}
    return v


def trend(metric: str, districts: list[str] | None = None) -> dict:
    sch, label, unit, _hib, _ = METRICS[metric]
    labels = [month_label(m) for m in MONTHS]
    series = []
    for d in (districts or []):
        series.append({"name": d, "values": [_value(metric, d, m) for m in MONTHS]})
    if not districts or METRICS[metric][4] == "mean":
        series.append({"name": "State" + (" average" if METRICS[metric][4] == "mean" else " total"),
                       "values": [_state(metric, m) for m in MONTHS]})
    v = _base("line", f"{label} — last 12 months", ", ".join(districts) if districts else "Maharashtra")
    v["line"] = {"labels": labels, "series": series, "unit": unit}
    v["facts"] = {s["name"]: f"{_fmt(s['values'][0], unit)} → {_fmt(s['values'][-1], unit)}" for s in series}
    return v


def anomalies(scheme: str | None = None) -> dict:
    """Districts whose latest month breaks sharply from their own recent pattern."""
    rows = []
    metrics = [m for m in METRICS if not scheme or METRICS[m][0] == scheme]
    for m in metrics:
        sch, label, unit, hib, _ = METRICS[m]
        s = _series(m).pivot(index="district", columns="month", values="value")
        hist, latest = s.iloc[:, -7:-1], s.iloc[:, -1]
        diffs = hist.diff(axis=1).iloc[:, 1:]
        spread = diffs.abs().stack().median() * 1.4826 + 1e-9
        typical = hist.iloc[:, -3:].mean(axis=1)
        z = (latest - typical) / spread
        bad = -z if hib else z  # positive = worse than expected
        for d in z.index:
            if bad[d] > 4:
                rows.append((float(bad[d]), d, label, _fmt(latest[d], unit), _fmt(typical[d], unit), SCHEMES[sch]))
    rows.sort(reverse=True)
    rows = rows[:8]
    v = _base("table", "Anomalies — sharp changes in the latest month",
              (SCHEMES[scheme] if scheme else "All schemes") + f" · {month_label(LATEST)} vs previous 3 months")
    v["table"] = {"columns": ["District", "Indicator", "Latest", "Usual (3-month avg)", "Scheme"],
                  "rows": [list(r[1:]) for r in rows]}
    v["facts"] = {f"{r[1]} — {r[2]}": f"{r[3]} vs usual {r[4]}" for r in rows} or {"result": "no anomalies"}
    return v


QUERIES = {
    "state_overview": state_overview,
    "district_status": district_status,
    "compare": compare,
    "ranking": ranking,
    "trend": trend,
    "anomalies": anomalies,
}


# ---------------------------------------------------------------- routing

_ROUTER_SYS = """You route questions from senior officials of Maharashtra's Water Supply & Sanitation
Department to ONE governed query. You never answer from your own knowledge and never invent numbers.
Questions may be in English, Hindi or Marathi (Devanagari or romanised).

Query types (intent → params):
- state_overview: {"scheme": scheme_id or null}
- district_status: {"district": name, "scheme": scheme_id or null}
- compare: {"districts": [2-8 names], "metric": metric_id}
- ranking: {"metric": metric_id, "order": "best"|"worst", "n": 3-15, "division": division or null, "by": "district"|"division"}
- trend: {"metric": metric_id, "districts": [0-4 names]}
- anomalies: {"scheme": scheme_id or null}
- unsupported: {"reason": short text}   (not about this data, or impossible to map)

Rules: use EXACT district names from the catalogue (map Devanagari names/aliases to the English
name). Pick the metric that best fits; default for "performance"/"progress" questions is
jjm_coverage_pct. "Lagging/worst/poor" → order "worst"; "best/top" → "best".
Set "show_on_screen": true if the user asks to show/display/open something on the screen.
Set "language" to the language of the question: "en", "hi" or "mr". Hindi and Marathi share
Devanagari — tell them apart by grammar: Hindi uses है/हैं, का/की/के, में, जिले, दिखाओ, करो, क्या;
Marathi uses आहे/आहेत, चा/ची/चे, मध्ये, जिल्हा/जिल्ह्या, दाखवा, करा, काय.

Reply with JSON only: {"intent": ..., "params": {...}, "language": ..., "show_on_screen": bool}"""


def _catalog_text() -> str:
    d = "; ".join(f"{n}|{dev}" + (f"|{','.join(al)}" if al else "") for n, dev, _div, al in DISTRICTS)
    m = "; ".join(f"{k}: {v[1]} ({v[2]}, scheme {v[0]}{', lower is better' if not v[3] else ''})"
                  for k, v in METRICS.items())
    return (f"Districts: {d}\nDivisions: {', '.join(DIVISIONS)}\n"
            f"Schemes: {json.dumps(SCHEMES)}\nMetrics: {m}\n"
            f"Data covers {month_label(MONTHS[0])} – {month_label(LATEST)} (latest).")


def _district(x) -> str:
    name = _ALIASES.get(str(x).strip().lower())
    if not name:
        raise ValueError(f"unknown district '{x}'")
    return name


def _validate(intent: str, p: dict) -> dict:
    """Keep only allowed parameters, with allowed values."""
    out: dict = {}
    if "scheme" in p and p["scheme"]:
        if p["scheme"] not in SCHEMES:
            raise ValueError(f"unknown scheme '{p['scheme']}'")
        out["scheme"] = p["scheme"]
    if intent == "district_status":
        out["district"] = _district(p.get("district"))
    if intent in ("compare", "ranking", "trend"):
        metric = p.get("metric") or "jjm_coverage_pct"
        if metric not in METRICS:
            raise ValueError(f"unknown metric '{metric}'")
        out["metric"] = metric
        out.pop("scheme", None)
    if intent == "compare":
        ds = [_district(d) for d in (p.get("districts") or [])]
        if len(ds) < 2:
            raise ValueError("compare needs at least two districts")
        out["districts"] = list(dict.fromkeys(ds))[:8]
    if intent == "trend":
        out["districts"] = list(dict.fromkeys(_district(d) for d in (p.get("districts") or [])))[:4]
    if intent == "ranking":
        out["order"] = "best" if p.get("order") == "best" else "worst"
        out["n"] = max(3, min(15, int(p.get("n") or 5)))
        if p.get("division"):
            if p["division"] not in DIVISIONS:
                raise ValueError(f"unknown division '{p['division']}'")
            out["division"] = p["division"]
        out["by"] = "division" if p.get("by") == "division" else "district"
    return out


_LANG_NAME = {"en": "English", "hi": "Hindi", "mr": "Marathi"}

# Words that only one of the two languages uses — more reliable than the LLM's guess.
_HI_WORDS = {"है", "हैं", "का", "की", "के", "में", "जिले", "जिला", "दिखाओ", "दिखाइए", "करो", "क्या",
             "कौन", "सबसे", "और", "पर", "से", "बताओ", "बताइए", "स्थिति"}
_MR_WORDS = {"आहे", "आहेत", "चा", "ची", "चे", "मध्ये", "जिल्हा", "जिल्ह्याची", "जिल्ह्यांची", "दाखवा",
             "करा", "काय", "कोणते", "आणि", "सांगा", "स्थिती", "सर्वात"}


def _hi_or_mr(text: str) -> str | None:
    if not any("ऀ" <= ch <= "ॿ" for ch in text):
        return None
    words = {w.strip("?।.,!") for w in text.split()}
    hi, mr = len(words & _HI_WORDS), len(words & _MR_WORDS)
    return "hi" if hi > mr else "mr" if mr > hi else None


def _narrate(question: str, view: dict, lang: str) -> str:
    try:
        return llm_chat("summary", [
            {"role": "system", "content":
                f"Answer a senior government official in {_LANG_NAME.get(lang, 'English')} in 1-2 short "
                "sentences, using ONLY the figures in the data below. No figures that are not given. "
                "Do not mention that the data is sample data (the screen shows that)."},
            {"role": "user", "content": f"Question: {question}\nResult: {view['title']} ({view['as_of']})\n"
                                        f"Data: {json.dumps(view['facts'], ensure_ascii=False)}"},
        ], temperature=0.2)
    except Exception:
        return view["title"] + ": " + "; ".join(f"{k} {v}" for k, v in list(view["facts"].items())[:5])


def ask(question: str, lang_hint: str | None = None) -> dict:
    """Route, run and phrase one question. Raises nothing; errors come back as answers.

    lang_hint: language Whisper heard (voice questions) or the user picked.
    """
    t0 = time.time()
    question = question.strip()
    try:
        route = llm_json("summary", [
            {"role": "system", "content": _ROUTER_SYS + "\n\nCATALOGUE\n" + _catalog_text()},
            {"role": "user", "content": question},
        ])
    except Exception as e:
        return {"question": question, "ok": False, "answer": f"Assistant unavailable: {e}"}

    intent = route.get("intent")
    lang = lang_hint if lang_hint in _LANG_NAME else (_hi_or_mr(question) or route.get("language"))
    lang = lang if lang in _LANG_NAME else "en"
    base = {"question": question, "language": lang, "show_on_screen": bool(route.get("show_on_screen"))}
    if intent not in QUERIES:
        reason = (route.get("params") or {}).get("reason") or "That isn't something the data covers."
        return {**base, "ok": False, "intent": "unsupported",
                "answer": f"I can only answer from the scheme data (status, comparisons, rankings, "
                          f"trends, anomalies). {reason}"}
    try:
        params = _validate(intent, route.get("params") or {})
        view = QUERIES[intent](**params)
    except Exception as e:
        return {**base, "ok": False, "intent": intent, "answer": f"Couldn't run that query: {e}"}

    answer = _narrate(question, view, lang)
    view.pop("facts", None)
    took = time.time() - t0
    print(f"  [assistant] {intent}({json.dumps(params, ensure_ascii=False)}) — {took:.1f}s — "
          f"\"{question[:60]}\"", flush=True)
    return {**base, "ok": True, "intent": intent, "params": params, "view": view, "answer": answer,
            "asked_at": time.time()}
