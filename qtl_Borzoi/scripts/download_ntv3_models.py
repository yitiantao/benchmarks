#!/usr/bin/env python3
"""Download pinned NTv3 PyTorch checkpoints without the duplicate JAX weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


MODELS = {
    "InstaDeepAI/NTv3_8M_pre": "c57a813117f0f90142098f81cc912b3357c9ecd1",
    "InstaDeepAI/NTv3_100M_pre": "5c685dca15891f5c5b80e0c930e23b87a217e441",
    "InstaDeepAI/NTv3_650M_pre": "5e6050bed864a5a8fb32481096bf555495316b31",
    "InstaDeepAI/NTv3_100M_post": "b7292f0bd1b5b004d28561783a1890e1ec6f94fd",
    "InstaDeepAI/NTv3_650M_post": "ad622051abbe9376bdea3f2727863fa559957e9f",
}
BASE_MODEL = (
    "InstaDeepAI/ntv3_base_model",
    "0ecff3637f0d3ba5b686d1095083218157c2ca34",
)
SUITES = {
    # Covers three parameter scales and both scoring routes.
    "qtl": (
        "InstaDeepAI/NTv3_8M_pre",
        "InstaDeepAI/NTv3_100M_post",
        "InstaDeepAI/NTv3_650M_post",
    ),
    "pre": (
        "InstaDeepAI/NTv3_8M_pre",
        "InstaDeepAI/NTv3_100M_pre",
        "InstaDeepAI/NTv3_650M_pre",
    ),
    "post": (
        "InstaDeepAI/NTv3_100M_post",
        "InstaDeepAI/NTv3_650M_post",
    ),
    "all": tuple(MODELS),
}
ALLOW_PATTERNS = (
    "*.json",
    "*.py",
    "*.md",
    "model.safetensors",
    ".gitattributes",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="qtl")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Exact supported repo ID; repeat to override --suite.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(".benchmark_deps/ntv3_hf")
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import get_token, snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as error:
        raise SystemExit(
            "缺少 huggingface_hub；先运行 scripts/setup_ntv3_benchmark.sh"
        ) from error

    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit(
            "未找到 Hugging Face token。NTv3 权重是 gated 模型：先在各模型页面接受条款，"
            "然后执行 `conda run -n ntv3 hf auth login`，或临时设置 HF_TOKEN。"
            "不要把 token 写入配置文件或聊天消息。"
        )
    selected = tuple(args.models or SUITES[args.suite])
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        parser.error(f"unsupported model IDs: {unknown}")
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloads = []
    requested = (BASE_MODEL,) + tuple((repo_id, MODELS[repo_id]) for repo_id in selected)
    for repo_id, revision in requested:
        print(f"[download] {repo_id}@{revision}", flush=True)
        try:
            snapshot = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                allow_patterns=list(ALLOW_PATTERNS),
            )
        except GatedRepoError as error:
            raise SystemExit(
                f"没有 {repo_id} 的 gated access。请用同一个 Hugging Face 账号打开模型页、"
                "接受条款后重试。"
            ) from error
        except HfHubHTTPError as error:
            raise SystemExit(f"下载 {repo_id} 失败: {error}") from error
        downloads.append(
            {"repo_id": repo_id, "revision": revision, "snapshot": snapshot}
        )
        print(f"[downloaded] {snapshot}", flush=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(cache_dir),
        "models": downloads,
    }
    manifest_path = cache_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
