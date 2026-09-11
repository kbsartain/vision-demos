"""ViTPose+ Large through the VLM Run gateway, the model the original demo used.

One blocking chat-completions call with the whole clip as a base64 video. The
gateway returns every frame in the format described in src/pose/__init__.py.
Needs VLMRUN_API_KEY, from the environment or a .env file at or above the
project directory.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

CONTENT_OBJECT_FRAMES = "vitpose_plus.pose.frames"


def load_api_key(start: Path) -> tuple[str, str]:
    preset = os.getenv("VLMRUN_API_KEY")
    if preset:
        return preset, "environment"
    for directory in [start, *start.parents]:
        env_file = directory / ".env"
        if env_file.is_file():
            from dotenv import dotenv_values
            key = (dotenv_values(env_file) or {}).get("VLMRUN_API_KEY")
            if key:
                os.environ["VLMRUN_API_KEY"] = key
                return key, str(env_file)
            raise RuntimeError(f"{env_file} has no VLMRUN_API_KEY entry.")
    raise RuntimeError("No VLMRUN_API_KEY in the environment and no .env found. "
                       "Copy .env.example to .env and add a key from https://app.vlm.run")


def build_extra_body(cfg, info) -> dict:
    body = {"method": "pose", "video_fps": info.fps, "precision": int(cfg.PRECISION)}
    if cfg.EVERY_FRAME:
        body["video_max_frames"] = info.n_frames
    return body


def run(video: Path, info, *, cfg, console) -> tuple[list[dict], dict]:
    from openai import OpenAI

    key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  API key from [green]{source}[/] (...{key[-4:]})")
    video_b64 = base64.b64encode(video.read_bytes()).decode("ascii")
    extra_body = build_extra_body(cfg, info)
    console.print(f"  model [bold]{cfg.VLMRUN_MODEL}[/], upload {len(video_b64) / 1e6:.1f} MB base64")

    client = OpenAI(base_url=cfg.VLMRUN_BASE_URL, api_key=key, timeout=cfg.VLMRUN_TIMEOUT,
                    max_retries=1)
    with console.status("[cyan]waiting on the gateway[/]", spinner="dots"):
        response = client.chat.completions.create(
            model=cfg.VLMRUN_MODEL,
            messages=[{"role": "user", "content": [{"type": "video_url", "video_url": {
                "url": f"data:video/mp4;base64,{video_b64}"}}]}],
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    try:
        content = payload["data"][0]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response envelope: {json.dumps(payload)[:400]}") from exc
    if content.get("object") != CONTENT_OBJECT_FRAMES:
        raise RuntimeError(f"Expected {CONTENT_OBJECT_FRAMES}, got {content.get('object')!r}")
    return content["items"], {"backend": "vlmrun", "model": cfg.VLMRUN_MODEL,
                              "extra_body": extra_body, "usage": usage}


def cache_key(cfg) -> dict:
    return {"backend": "vlmrun", "model": cfg.VLMRUN_MODEL, "every_frame": cfg.EVERY_FRAME,
            "precision": cfg.PRECISION}
