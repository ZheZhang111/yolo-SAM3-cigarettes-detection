"""Chinese visualization labels for detection results."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_FONT = ROOT / "assets" / "fonts" / "NotoSansCJK-Regular.ttc"

CLASS_NAMES_ZH: dict[int, str] = {
    0: "烟头",
    1: "烟盒",
    2: "香烟",
    3: "打火机",
}

# English name -> Chinese (for VLM log display)
EN_TO_ZH: dict[str, str] = {
    "cigarettes_butts": "烟头",
    "cig-pack": "烟盒",
    "cigarette": "香烟",
    "Lighter": "打火机",
    "lighter": "打火机",
    "other": "其他",
}


def class_name_zh(cls_id: int, names: dict | None = None) -> str:
    if cls_id in CLASS_NAMES_ZH:
        return CLASS_NAMES_ZH[cls_id]
    if names is not None:
        return str(names.get(cls_id, names[cls_id]))
    return str(cls_id)


def en_name_zh(name: str) -> str:
    return EN_TO_ZH.get(name, name)


@lru_cache(maxsize=4)
def _load_font(font_path: str, size: int, index: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size, index=index)


def resolve_font_path(font_path: Path | None = None) -> Path:
    if font_path is not None and font_path.is_file():
        return font_path
    candidates = [
        DEFAULT_FONT,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Chinese font not found. Place NotoSansCJK-Regular.ttc under assets/fonts/."
    )


def draw_chinese_label(
    img_bgr: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    font_size: int,
    color_bgr: tuple[int, int, int],
    font_path: Path | None = None,
    padding: int = 8,
    bold: bool = False,
) -> np.ndarray:
    """Draw text with background on BGR image; returns BGR image."""
    font_file = resolve_font_path(font_path)
    # NotoSansCJK.ttc: index 0 = SC in most Ubuntu packages
    font = _load_font(str(font_file), font_size, 0)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    x, y = xy
    stroke_width = max(2, font_size // 14) if bold else 0
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bg = (color_bgr[2], color_bgr[1], color_bgr[0])
    fg = (255, 255, 255)
    draw.rectangle((x, y, x + tw + padding * 2, y + th + padding * 2), fill=bg)
    draw.text(
        (x + padding, y + padding),
        text,
        font=font,
        fill=fg,
        stroke_width=stroke_width,
        stroke_fill=fg,
    )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
