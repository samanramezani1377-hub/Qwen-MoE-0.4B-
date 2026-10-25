#!/usr/bin/env python3
"""Collect a small Persian Wikipedia corpus via the MediaWiki API.

Uses bounded, batched API requests rather than scanning the Hugging Face
Wikipedia streaming dataset from its first shard.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests


API_URL = "https://fa.wikipedia.org/w/api.php"
USER_AGENT = "QwenMoEResearch/1.0 (small research corpus; GitHub Actions)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=160)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--max-requests", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", default="data/fa_wikipedia.txt")
    args = parser.parse_args()

    if args.count < 1 or args.max_requests < 1:
        parser.error("--count and --max-requests must be positive")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    seen: set[int] = set()
    samples: list[str] = []
    failures = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for request_no in range(1, args.max_requests + 1):
            params = {
                "action": "query",
                "generator": "random",
                "grnnamespace": 0,
                "grnlimit": 10,
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "formatversion": 2,
            }
            try:
                response = session.get(API_URL, params=params, timeout=args.timeout)
                response.raise_for_status()
                payload = response.json()
                pages = payload.get("query", {}).get("pages", [])
                if not isinstance(pages, list):
                    raise ValueError("Unexpected MediaWiki API response format")
            except (requests.RequestException, ValueError) as exc:
                failures += 1
                print(
                    f"Request {request_no}/{args.max_requests} failed "
                    f"({failures}/3): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if failures >= 3:
                    break
                time.sleep(min(request_no, 3))
                continue

            failures = 0
            added = 0
            for page in pages:
                page_id = page.get("pageid")
                if page_id is None or page_id in seen:
                    continue
                seen.add(page_id)
                body = " ".join(str(page.get("extract", "")).split())
                if len(body) < args.min_chars:
                    continue
                out.write(body[: args.max_chars] + "\n")
                samples.append(body[: args.max_chars])
                added += 1
                if len(samples) >= args.count:
                    break

            out.flush()
            print(
                f"Batch {request_no}/{args.max_requests}: "
                f"+{added} usable, {len(samples)}/{args.count} collected",
                flush=True,
            )
            if len(samples) >= args.count:
                break
            time.sleep(0.25)

    if len(samples) < args.count:
        print(
            f"ERROR: collected {len(samples)}/{args.count} samples after "
            f"{args.max_requests} requests. Output: {args.output}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(f"SUCCESS: collected {len(samples)} Persian Wikipedia samples -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
