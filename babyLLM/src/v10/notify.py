#!/usr/bin/env python3
"""
Notification helper for BabyLM V10 training pipeline.
Supports: Slack webhook, wandb alerts, file-based logging (fallback).
Usage: python notify.py --title "Title" --message "Message" [--slack URL] [--email ADDR]
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime


def send_slack(webhook_url, title, message):
    if not webhook_url:
        return False
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = json.dumps({"text": f"*{title}*\n```[{ts}]\n{message}```"}).encode(
        "utf-8"
    )
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=10)
    return True


def send_email(to_addr, title, message):
    if not to_addr:
        return False
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = f"[{ts}]\n{message}"
    try:
        proc = subprocess.run(
            ["mail", "-s", f"[BabyLM-V10] {title}", to_addr],
            input=body,
            text=True,
            timeout=10,
            capture_output=True,
        )
        if proc.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    log_path = "/tmp/babylm_v10_mail.log"
    with open(log_path, "a") as f:
        f.write(f"[{ts}] MAIL FALLBACK | {title}: {body}\n")
    return False


def send_wandb_alert(title, message, level="INFO"):
    try:
        import wandb

        level_map = {
            "INFO": wandb.AlertLevel.INFO,
            "WARN": wandb.AlertLevel.WARN,
            "ERROR": wandb.AlertLevel.ERROR,
        }
        wandb.alert(
            title=title,
            text=message,
            level=level_map.get(level.upper(), wandb.AlertLevel.INFO),
        )
        return True
    except Exception as e:
        return False


def log_to_file(title, message):
    log_path = "/tmp/babylm_v10_notifications.log"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{ts}] {title}: {message}\n")


def main():
    parser = argparse.ArgumentParser(description="BabyLM V10 Notification Helper")
    parser.add_argument("--title", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--slack", default="", help="Slack webhook URL")
    parser.add_argument("--email", default="", help="Email address")
    parser.add_argument("--wandb", action="store_true", help="Send wandb alert")
    parser.add_argument("--level", default="INFO", choices=["INFO", "WARN", "ERROR"])
    args = parser.parse_args()

    results = []

    if args.slack:
        try:
            send_slack(args.slack, args.title, args.message)
            results.append("slack:OK")
        except Exception as e:
            results.append(f"slack:FAIL({e})")

    if args.email:
        try:
            send_email(args.email, args.title, args.message)
            results.append("email:OK")
        except Exception as e:
            results.append(f"email:FAIL({e})")

    if args.wandb:
        try:
            send_wandb_alert(args.title, args.message, args.level)
            results.append("wandb:OK")
        except Exception as e:
            results.append(f"wandb:FAIL({e})")

    log_to_file(args.title, args.message)

    print(
        f"Notification: {args.title} | {', '.join(results) if results else 'no channels'}"
    )


if __name__ == "__main__":
    main()
