import os

import httpx


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_TIMEOUT_READ = 300.0
DEFAULT_DEEPSEEK_TIMEOUT_CONNECT = 30.0


def get_deepseek_timeout_seconds():
    raw = os.environ.get("DEEPSEEK_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_DEEPSEEK_TIMEOUT_READ
    return max(float(raw), 30.0)


def build_openai_timeout(timeout_seconds=None):
    read_timeout = timeout_seconds if timeout_seconds is not None else get_deepseek_timeout_seconds()
    return httpx.Timeout(
        connect=DEFAULT_DEEPSEEK_TIMEOUT_CONNECT,
        read=read_timeout,
        write=60.0,
        pool=30.0,
    )


def format_llm_error(exc):
    message = f"AI risk analysis is temporarily unavailable: {exc}"
    lowered = str(exc).lower()
    if "timed out" in lowered or "timeout" in lowered:
        message += (
            f" (current API read timeout is {get_deepseek_timeout_seconds():.0f}s; "
            "combined analysis with V4-Pro can be slow—try export DEEPSEEK_TIMEOUT=600 and restart)"
        )
    return message
