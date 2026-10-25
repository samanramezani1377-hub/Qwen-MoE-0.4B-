#!/usr/bin/env python3
"""Collect a small Persian Wikipedia corpus with bounded batched API requests."""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests

API_URL = "https://fa.wikipedia.org/w/api.php"
USER_AGENT = "QwenMoEResearch/1.0 (GitHub Actions research corpus; contact via repository issues)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--min-chars", type=int, default=160)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output", default="data/fa_wikipedia.txt")
    args = parser.parse_args()
    if args.count < 1 or args.max_requests < 1:
        parser.error("--count and --max-requests must be positive")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    })
    seen: set[int] = set()
    count = 0
    consecutive_failures = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for request_no in range(1, args.max_requests + 1):
            params = {
                "action": "query",
                "generator": "random",
                "grnnamespace": 0,
                "grnlimit": 50,
                "grnfilterredir": "nonredirects",
                "prop": "extracts",
                "explaintext": 1,
                "exchars": args.max_chars,
                "format": "json",
                "formatversion": 2,
                "maxlag": 5,
            }
            try:
                response = session.get(API_URL, params=params, timeout=args.timeout)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "5")
                    try:
                        wait_s = min(max(int(retry_after), 3), 30)
                    except ValueError:
                        wait_s = 5
                    raise requests.HTTPError(f"HTTP 429 rate limited (Retry-After {wait_s}s)", response=response)
                response.raise_for_status()
                payload = response.json()
                pages = payload.get("query", {}).get("pages", [])
                if not isinstance(pages, list):
                    raise ValueError("Unexpected MediaWiki API response format")
            except (requests.RequestException, ValueError) as exc:
                consecutive_failures += 1
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After", "5")
                try:
                    wait_s = min(max(int(retry_after), 3), 30)
                except ValueError:
                    wait_s = min(2 ** consecutive_failures, 15)
                print(
                    f"Request {request_no}/{args.max_requests} failed "
                    f"({consecutive_failures}/3): {exc}; waiting {wait_s}s",
                    file=sys.stderr,
                    flush=True,
                )
                if consecutive_failures >= 3:
                    break
                time.sleep(wait_s)
                continue

            consecutive_failures = 0
            added = 0
            for page in pages:
                page_id = page.get("pageid")
                if page_id is None or page_id in seen:
                    continue
                seen.add(page_id)
                body = " ".join(str(page.get("extract", "")).split())
                if len(body) < args.min_chars:
                    continue
                out.write(body[:args.max_chars] + "\n")
                count += 1
                added += 1
                if count >= args.count:
                    break
            out.flush()
            print(
                f"Batch {request_no}/{args.max_requests}: "
                f"+{added} usable, {count}/{args.count} collected",
                flush=True,
            )
            if count >= args.count:
                break
            time.sleep(2)

    if count < args.count:
        print(
            f"ERROR: collected {count}/{args.count} samples after "
            f"{args.max_requests} bounded batch requests. Output: {args.output}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"SUCCESS: collected {count} Persian Wikipedia samples -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
