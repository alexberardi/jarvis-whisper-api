"""Report speaker-recognition match rate from jarvis-whisper logs.

Parses "Speaker match: user_id=... confidence=... node=... household=..." lines
emitted by `/transcribe` in `app/main.py`. Counts matched vs unmatched and
summarizes confidence so you can sanity-check threshold tuning.

Usage:
    python scripts/speaker_match_rate.py                  # last 1h
    python scripts/speaker_match_rate.py --since 24h
    python scripts/speaker_match_rate.py --since 7d --node 51663b4f-...
    python scripts/speaker_match_rate.py --logs-url http://10.0.0.103:7702

Requires env JARVIS_APP_ID and JARVIS_APP_KEY (defaults to jarvis-command-center
app credentials when run from a host with those exported).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

LINE_RE = re.compile(
    r"Speaker match: user_id=(?P<user_id>\S+) "
    r"confidence=(?P<conf>[\d.]+) "
    r"node=(?P<node>\S+) "
    r"household=(?P<household>\S+)"
)


def parse_since(value: str) -> str:
    """Accept '1h', '24h', '7d', '30m', or a literal ISO timestamp."""
    m = re.fullmatch(r"(\d+)([smhd])", value)
    if not m:
        # Assume caller passed an ISO timestamp through; trust it.
        return value
    n, unit = int(m.group(1)), m.group(2)
    delta = {"s": timedelta(seconds=n), "m": timedelta(minutes=n),
             "h": timedelta(hours=n), "d": timedelta(days=n)}[unit]
    return (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_logs(logs_url: str, service: str, since: str, app_id: str,
               app_key: str, limit: int = 1000) -> list[dict]:
    """Pull up to `limit` log lines matching 'Speaker match' from the logs API."""
    qs = urllib.parse.urlencode({
        "service": service,
        "search": "Speaker match",
        "since": since,
        "limit": limit,
    })
    req = urllib.request.Request(
        f"{logs_url}/api/v0/logs?{qs}",
        headers={"x-jarvis-app-id": app_id, "x-jarvis-app-key": app_key},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default="1h", help="Time window (e.g. 30m, 1h, 24h, 7d) or ISO timestamp. Default: 1h")
    p.add_argument("--logs-url", default=os.getenv("JARVIS_LOGS_URL", "http://localhost:7702"))
    p.add_argument("--service", default="jarvis-whisper")
    p.add_argument("--limit", type=int, default=1000, help="Max log lines to fetch (logs API hard-caps at 1000)")
    p.add_argument("--node", help="Filter to a single node_id")
    p.add_argument("--json", action="store_true", help="Emit a JSON summary instead of text")
    args = p.parse_args()

    app_id = os.getenv("JARVIS_APP_ID", "jarvis-command-center")
    app_key = os.getenv("JARVIS_APP_KEY")
    if not app_key:
        print("error: set JARVIS_APP_KEY in env (read it from the command-center container)", file=sys.stderr)
        return 2

    since = parse_since(args.since)
    entries = fetch_logs(args.logs_url, args.service, since, app_id, app_key, args.limit)

    parsed = []
    for e in entries:
        m = LINE_RE.search(e.get("message") or "")
        if not m:
            continue
        if args.node and m.group("node") != args.node:
            continue
        parsed.append({
            "ts": e.get("timestamp"),
            "user_id": None if m.group("user_id") == "None" else int(m.group("user_id")),
            "confidence": float(m.group("conf")),
            "node": m.group("node"),
            "household": m.group("household"),
        })

    total = len(parsed)
    matched = [r for r in parsed if r["user_id"] is not None]
    unmatched = [r for r in parsed if r["user_id"] is None]
    rate = (len(matched) / total * 100) if total else 0.0

    per_node: dict[str, dict] = {}
    for r in parsed:
        n = per_node.setdefault(r["node"], {"total": 0, "matched": 0, "confs_matched": []})
        n["total"] += 1
        if r["user_id"] is not None:
            n["matched"] += 1
            n["confs_matched"].append(r["confidence"])

    summary = {
        "since": since,
        "service": args.service,
        "node_filter": args.node,
        "total": total,
        "matched": len(matched),
        "unmatched": len(unmatched),
        "match_rate_pct": round(rate, 2),
        "confidence": {
            "matched": _stats([r["confidence"] for r in matched]),
            "unmatched": _stats([r["confidence"] for r in unmatched]),
        },
        "per_node": {
            n: {
                "total": v["total"],
                "matched": v["matched"],
                "match_rate_pct": round(v["matched"] / v["total"] * 100, 2),
                "mean_conf_matched": round(statistics.mean(v["confs_matched"]), 3) if v["confs_matched"] else None,
            }
            for n, v in per_node.items()
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"window:  since {since}  service={args.service}" + (f"  node={args.node}" if args.node else ""))
    if total == 0:
        print("no 'Speaker match' samples in window. Either no traffic, voice-recognition disabled, or whisper not yet restarted with the new log.")
        return 0
    print(f"samples: {total}  matched: {len(matched)} ({rate:.1f}%)  unmatched: {len(unmatched)} ({100 - rate:.1f}%)")
    print(f"matched   confidence: {summary['confidence']['matched']}")
    print(f"unmatched confidence: {summary['confidence']['unmatched']}")
    if len(per_node) > 1:
        print("per-node:")
        for n, v in summary["per_node"].items():
            short = n[:8] if len(n) > 8 else n
            print(f"  {short}  total={v['total']:>4}  matched={v['matched']:>4}  rate={v['match_rate_pct']:5.1f}%  mean_conf={v['mean_conf_matched']}")
    return 0


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "min": round(min(xs), 3),
        "mean": round(statistics.mean(xs), 3),
        "max": round(max(xs), 3),
        "stdev": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
    }


if __name__ == "__main__":
    sys.exit(main())
