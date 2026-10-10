from __future__ import annotations

import hashlib
from pathlib import Path
import sys


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SHA256 = {
    "lnln_deploy.onnx": "695129e911ba2f116b8587415bd4c9249cdec76608e56d39c3cb82778dad4c58",
    "MSA_FET/exts/OpenFace/FeatureExtraction": "54d8e591bbeacdcf09bf76c36ede3b27d80afa9b991463171e7c05dfbe018435",
    "MSA_FET/exts/pretrained/sfd_face.pth": "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491",
    "MSA_FET/exts/pretrained/TalkSet.pth": "d985ddd07dca08864a28337726f5b7d91b6426bebc7327f0e9b21cf9ff1cc937",
    "MSA_FET/exts/OpenFace/model/patch_experts/cen_patches_0.25_of.dat": "99d3df9888115428075de7b8f1bc86176881de640c2fdf47e2ab0cee556661bf",
    "MSA_FET/exts/OpenFace/model/patch_experts/cen_patches_0.35_of.dat": "1070bff51b077ee6a18ce8e1ebbc6f568e1a7a469f6911a8263413878b3df469",
    "MSA_FET/exts/OpenFace/model/patch_experts/cen_patches_0.50_of.dat": "3f08124067e326e83e3261a55c79493e745b01138c9c1b59713dd9b17424168e",
    "MSA_FET/exts/OpenFace/model/patch_experts/cen_patches_1.00_of.dat": "bed961bbfa2cc41a709d44c1d1d8a92b2537bc46f2fd6ab430c137fedd0e21b0",
}

REQUIRED_LIB_PATTERNS = (
    "libXcomposite.so.1.0.0",
    "libXdamage.so.1.1.0",
    "libatk-1.0.so.0.23510.1",
    "libboost_filesystem.so.1.71.0",
    "libdc1394.so.22.2.1",
    "libgdk-x11-2.0.so.0.2400.32",
    "libgtk-x11-2.0.so.0.2400.32",
    "libopencv_calib3d.so.4.1.0",
    "libopencv_core.so.4.1.0",
    "libopencv_features2d.so.4.1.0",
    "libopencv_flann.so.4.1.0",
    "libopencv_highgui.so.4.1.0",
    "libopencv_imgcodecs.so.4.1.0",
    "libopencv_imgproc.so.4.1.0",
    "libopencv_objdetect.so.4.1.0",
    "libopencv_videoio.so.4.1.0",
    "libtbb.so.2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []

    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"缺少文件: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(f"SHA-256 不匹配: {relative}\n  expected={expected}\n  actual={actual}")

    libs_dir = ROOT / "libs"
    for name in REQUIRED_LIB_PATTERNS:
        if not (libs_dir / name).is_file():
            errors.append(f"缺少运行库: libs/{name}")

    if errors:
        print("资源检查失败：")
        print("\n".join(f"- {item}" for item in errors))
        return 1

    print("资源检查通过，可以开始构建镜像。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
