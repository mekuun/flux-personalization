from PIL import Image, ImageOps
from pathlib import Path
import os

FOLDER   = Path("mask_folder")
OUT_DIR  = Path("grids_out")
BASE_OUT_NAME = "all_files_grid.jpg"
BACKING = "flux_batch_1.png"
PREFIXES  = ["SINGLE", "DOUBLE", "VAL"]

ROWS     = 20
THUMB_W, THUMB_H = 512, 512
MARGIN   = 10
BG_COLOR = (255, 255, 255)
EXTS     = {".png", ".jpg", ".jpeg", ".webp"}

def load_and_pad(path: Path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (THUMB_W, THUMB_H), method=Image.Resampling.LANCZOS)
    pad_w, pad_h = THUMB_W - img.width, THUMB_H - img.height
    return ImageOps.expand(
        img,
        border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
        fill=BG_COLOR
    )

def load_and_pad_overlay(path: Path):
    img = Image.open(path).convert("RGBA")
    img = ImageOps.contain(img, (THUMB_W, THUMB_H), method=Image.Resampling.LANCZOS)
    pad_w, pad_h = THUMB_W - img.width, THUMB_H - img.height
    return ImageOps.expand(
        img,
        border=(pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
        fill=(0, 0, 0, 0)  
    )

def make_grid(images, rows, cols):
    assert len(images) == rows * cols
    grid_w = MARGIN + cols * THUMB_W + (cols - 1) * MARGIN + MARGIN
    grid_h = MARGIN + rows * THUMB_H + (rows - 1) * MARGIN + MARGIN
    canvas = Image.new("RGB", (grid_w, grid_h), BG_COLOR)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = MARGIN + c * (THUMB_W + MARGIN)
        y = MARGIN + r * (THUMB_H + MARGIN)
        canvas.paste(img, (x, y))
    return canvas

def make_grid_overlay(images, rows, cols):
    assert len(images) == rows * cols
    grid_w = MARGIN + cols * THUMB_W + (cols - 1) * MARGIN + MARGIN
    grid_h = MARGIN + rows * THUMB_H + (rows - 1) * MARGIN + MARGIN
    canvas = Image.new("RGBA", (grid_w, grid_h), (0, 0, 0, 0))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = MARGIN + c * (THUMB_W + MARGIN)
        y = MARGIN + r * (THUMB_H + MARGIN)
        img = img.convert("RGBA")
        canvas.paste(img, (x, y), img) 
    return canvas

def main():
    if not FOLDER.exists():
        raise FileNotFoundError(f"Нет папки: {FOLDER}")
    for prefix in PREFIXES:
        files = sorted(
            [p for p in FOLDER.iterdir()
             if p.is_file() and p.suffix.lower() in EXTS and p.name.startswith(prefix)],
            key=lambda p: p.name
        )

        if not files:
            print("[ПРОПУСК] Нет изображений в папке.")
            continue

        total = len(files)
        if total % ROWS != 0:
            raise ValueError(f"Число файлов ({total}) не делится на {ROWS}. "
                             f"Добавь/удали файлы или поменяй ROWS.")

        cols = total // ROWS
        print(f"[ИНФО] Найдено {total} файлов → сетка {ROWS}×{cols}")

        backing = [load_and_pad(BACKING)] * len(files)

        images = [load_and_pad_overlay(p) for p in files]

        grid1 = make_grid(backing, rows=ROWS, cols=cols)          
        grid  = make_grid_overlay(images, rows=ROWS, cols=cols)      

        grid1 = grid1.resize((grid1.size[0]//4, grid1.size[1]//4), Image.Resampling.LANCZOS)
        grid  = grid.resize(grid1.size, Image.Resampling.LANCZOS)

        out = Image.alpha_composite(grid1.convert("RGBA"), grid).convert("RGB")

        OUT_DIR.mkdir(exist_ok=True)
        OUT_NAME = f"{prefix}_" + BASE_OUT_NAME
        out_path = OUT_DIR / OUT_NAME
        out.save(out_path, quality=100)  
        print(f"[OK] Сохранено: {out_path}")

if __name__ == "__main__":
    main()