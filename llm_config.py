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
    message = f"AI 风险分析暂不可用：{exc}"
    lowered = str(exc).lower()
    if "timed out" in lowered or "timeout" in lowered:
        message += (
            f"（当前 API 读取超时为 {get_deepseek_timeout_seconds():.0f} 秒；"
            "综合判断 + V4-Pro 生成较慢，可执行 export DEEPSEEK_TIMEOUT=600 后重启应用）"
        )
    return message
