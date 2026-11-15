#!/usr/bin/env python3
"""Install the optional model Rust accelerator; --required forbids a CI fallback success."""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

RUST_DIR = Path(__file__).resolve().parent.parent / "rust"
MATURIN_REQUIREMENT = "maturin==1.16.0"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--required", action="store_true", help="fail if the extension cannot actually be built"
    )
    args = parser.parse_args(argv)
    if not RUST_DIR.is_dir() or shutil.which("cargo") is None:
        print("Rust工程或cargo不可用；未构建Rust扩展")
        return 1 if args.required else 0
    if shutil.which("uv") is not None:
        # Pinned by the project lock; do not accidentally install into system Python.
        maturin_cmd = [
            "uv",
            "run",
            "--project",
            str(RUST_DIR.parent),
            "--frozen",
            "--group",
            "accel",
            "maturin",
        ]
    else:
        if importlib.util.find_spec("maturin") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", MATURIN_REQUIREMENT])
        maturin_cmd = [sys.executable, "-m", "maturin"]
    print("编译模型Rust扩展 ...")
    result = subprocess.run(
        [
            *maturin_cmd,
            "develop",
            "--manifest-path",
            str(RUST_DIR / "Cargo.toml"),
            "--locked",
            "--release",
        ],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode:
        print(
            "Rust扩展未构建成功"
            if args.required
            else "可选扩展构建失败；回退Python，不能计作Rust验证"
        )
        return max(1, result.returncode) if args.required else 0
    print("Rust扩展构建安装完成；运行时还须核验实际加载的后端")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
