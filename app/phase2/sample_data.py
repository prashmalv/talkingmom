"""SYNTHETIC sample data for the Phase 2 demo.

Stands in for the department's MIS systems (Jal Jeevan Mission, Swachh Bharat
Mission-Grameen, water quality, grievances) until real integrations exist.
District names are real Maharashtra districts; every number is generated from
a fixed seed, so the demo is repeatable. Never present it as real data — every
answer carries SOURCE_NOTE.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

SOURCE_NOTE = "Sample data (synthetic) — for demonstration only"

# (English name, Devanagari name, division, aliases)
DISTRICTS = [
    ("Mumbai City", "मुंबई शहर", "Konkan", []),
    ("Mumbai Suburban", "मुंबई उपनगर", "Konkan", []),
    ("Thane", "ठाणे", "Konkan", []),
    ("Palghar", "पालघर", "Konkan", []),
    ("Raigad", "रायगड", "Konkan", []),
    ("Ratnagiri", "रत्नागिरी", "Konkan", []),
    ("Sindhudurg", "सिंधुदुर्ग", "Konkan", []),
    ("Pune", "पुणे", "Pune", []),
    ("Satara", "सातारा", "Pune", []),
    ("Sangli", "सांगली", "Pune", []),
    ("Kolhapur", "कोल्हापूर", "Pune", []),
    ("Solapur", "सोलापूर", "Pune", []),
    ("Nashik", "नाशिक", "Nashik", []),
    ("Ahilyanagar", "अहिल्यानगर", "Nashik", ["Ahmednagar"]),
    ("Dhule", "धुळे", "Nashik", []),
    ("Jalgaon", "जळगाव", "Nashik", []),
    ("Nandurbar", "नंदुरबार", "Nashik", []),
    ("Chhatrapati Sambhajinagar", "छत्रपती संभाजीनगर", "Chhatrapati Sambhajinagar", ["Aurangabad"]),
    ("Jalna", "जालना", "Chhatrapati Sambhajinagar", []),
    ("Beed", "बीड", "Chhatrapati Sambhajinagar", []),
    ("Dharashiv", "धाराशिव", "Chhatrapati Sambhajinagar", ["Osmanabad"]),
    ("Latur", "लातूर", "Chhatrapati Sambhajinagar", []),
    ("Nanded", "नांदेड", "Chhatrapati Sambhajinagar", []),
    ("Parbhani", "परभणी", "Chhatrapati Sambhajinagar", []),
    ("Hingoli", "हिंगोली", "Chhatrapati Sambhajinagar", []),
    ("Amravati", "अमरावती", "Amravati", []),
    ("Akola", "अकोला", "Amravati", []),
    ("Washim", "वाशिम", "Amravati", []),
    ("Buldhana", "बुलढाणा", "Amravati", []),
    ("Yavatmal", "यवतमाळ", "Amravati", []),
    ("Nagpur", "नागपूर", "Nagpur", []),
    ("Wardha", "वर्धा", "Nagpur", []),
    ("Bhandara", "भंडारा", "Nagpur", []),
    ("Gondia", "गोंदिया", "Nagpur", []),
    ("Chandrapur", "चंद्रपूर", "Nagpur", []),
    ("Gadchiroli", "गडचिरोली", "Nagpur", []),
]
DISTRICT_NAMES = [d[0] for d in DISTRICTS]
DIVISIONS = sorted({d[2] for d in DISTRICTS})

SCHEMES = {
    "jjm": "Jal Jeevan Mission",
    "sbm": "Swachh Bharat Mission (Grameen)",
    "wq": "Water Quality Monitoring",
    "grv": "Citizen Grievances",
}

# id: (scheme, label, unit, higher_is_better, state aggregation)
METRICS = {
    "jjm_coverage_pct":   ("jjm", "Household tap connection coverage", "%", True, "mean"),
    "jjm_tap_connections": ("jjm", "Functional household tap connections", "count", True, "sum"),
    "jjm_hgj_villages_pct": ("jjm", "Har Ghar Jal certified villages", "%", True, "mean"),
    "sbm_odf_plus_pct":   ("sbm", "ODF Plus villages", "%", True, "mean"),
    "sbm_odf_plus_villages": ("sbm", "ODF Plus villages (count)", "count", True, "sum"),
    "wq_samples_tested":  ("wq", "Water samples tested", "count", True, "sum"),
    "wq_failure_pct":     ("wq", "Water samples failing quality norms", "%", False, "mean"),
    "grv_received":       ("grv", "Grievances received", "count", False, "sum"),
    "grv_pending":        ("grv", "Grievances pending", "count", False, "sum"),
    "grv_overdue":        ("grv", "Grievances overdue (>30 days)", "count", False, "sum"),
    "grv_resolution_pct": ("grv", "Grievance resolution rate", "%", True, "mean"),
}
# Headline metrics per scheme (status cards / default comparisons).
HEADLINE = {
    "jjm": ["jjm_coverage_pct", "jjm_hgj_villages_pct", "jjm_tap_connections"],
    "sbm": ["sbm_odf_plus_pct", "sbm_odf_plus_villages"],
    "wq": ["wq_failure_pct", "wq_samples_tested"],
    "grv": ["grv_resolution_pct", "grv_pending", "grv_overdue"],
}

MONTHS = pd.period_range("2025-10", "2026-09", freq="M")  # 12 months; latest = Sep 2026


def month_label(p) -> str:
    return pd.Period(p, freq="M").strftime("%b %Y")


@lru_cache(maxsize=1)
def facts() -> pd.DataFrame:
    """Long table: district, division, month, metric, value."""
    rng = np.random.default_rng(20261026)
    rows = []
    n = len(MONTHS)
    for di, (name, _dev, division, _al) in enumerate(DISTRICTS):
        urban = name.startswith("Mumbai")
        size = rng.uniform(0.6, 1.8) * (0.3 if urban else 1.0)
        households = int(size * 450_000)
        villages = int(size * 1100) if not urban else 40
        # Built-in demo stories: a few laggards and one water-quality spike.
        lag = 0.75 if name in ("Gadchiroli", "Nandurbar", "Hingoli") else 1.0
        base_cov = min(97, rng.uniform(62, 88) * lag)
        base_hgj = base_cov * rng.uniform(0.55, 0.8)
        base_odf = min(96, rng.uniform(55, 85) * (0.85 if lag < 1 else 1.0))
        base_fail = rng.uniform(2.5, 7.5)
        grv_rate = rng.uniform(0.82, 0.95) * (0.9 if lag < 1 else 1.0)
        for mi, m in enumerate(MONTHS):
            prog = mi / (n - 1)
            monsoon = 1.0 + (0.6 if m.month in (7, 8, 9) else 0.0)
            cov = min(99.5, base_cov + prog * rng.uniform(3, 9) + rng.normal(0, 0.4))
            hgj = min(99.0, base_hgj + prog * rng.uniform(4, 12) + rng.normal(0, 0.5))
            odf = min(99.0, base_odf + prog * rng.uniform(4, 10) + rng.normal(0, 0.6))
            tested = int(size * rng.uniform(2200, 2800) * (1.15 if m.month in (7, 8, 9) else 1.0))
            fail = max(0.5, base_fail * monsoon + rng.normal(0, 0.6))
            if name == "Beed" and m == MONTHS[-1]:
                fail *= 2.6  # anomaly for the demo
            received = int(size * rng.uniform(260, 420))
            resolved = int(received * min(0.99, grv_rate + rng.normal(0, 0.02)))
            pending = received - resolved + int(size * rng.uniform(20, 60))
            overdue = int(pending * rng.uniform(0.15, 0.4) * (1.6 if lag < 1 else 1.0))
            vals = {
                "jjm_coverage_pct": round(cov, 1),
                "jjm_tap_connections": int(households * cov / 100 * rng.uniform(0.93, 0.98)),
                "jjm_hgj_villages_pct": round(hgj, 1),
                "sbm_odf_plus_pct": round(odf, 1),
                "sbm_odf_plus_villages": int(villages * odf / 100),
                "wq_samples_tested": tested,
                "wq_failure_pct": round(fail, 1),
                "grv_received": received,
                "grv_pending": pending,
                "grv_overdue": overdue,
                "grv_resolution_pct": round(100 * resolved / max(received, 1), 1),
            }
            for metric, v in vals.items():
                rows.append((name, division, m, metric, v))
    return pd.DataFrame(rows, columns=["district", "division", "month", "metric", "value"])


def catalog() -> dict:
    """What the assistant is allowed to ask about (sent to the LLM router)."""
    return {
        "districts": [{"name": d[0], "devanagari": d[1], "division": d[2], "aliases": d[3]} for d in DISTRICTS],
        "divisions": DIVISIONS,
        "schemes": SCHEMES,
        "metrics": {k: {"scheme": v[0], "label": v[1], "unit": v[2], "higher_is_better": v[3]}
                    for k, v in METRICS.items()},
        "months": [str(m) for m in MONTHS],
        "latest_month": str(MONTHS[-1]),
    }
