from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


VENDORS = [
    "Apex Supplies Co.", "Blue River Logistics", "Cedar Tech Solutions",
    "Delta Office Mart", "Eagle Industrial", "Frontier Services LLC",
    "Global Parts Depot", "Harbor Trading Co.", "Inland Equipment",
    "Jasper Consulting", "Keystone Materials", "Lighthouse Media",
    "Metro Digital Print", "Nova Cleaning Services", "Orbit Software Inc.",
    "Pacific Auto Parts", "Quantum Analytics", "Riverside Catering",
    "Summit Construction", "Titan Tools & Hardware",
]

ITEMS = [
    ("Printer Paper A4 (500 sheets)", 2.5, 8.99),
    ("Ballpoint Pens (box of 12)", 1.0, 4.49),
    ("Stapler Heavy Duty", 1.0, 12.99),
    ("Filing Folders (pack of 25)", 3.0, 6.99),
    ("Laptop Stand Adjustable", 1.0, 34.99),
    ("USB-C Hub 7-Port", 2.0, 24.99),
    ("Whiteboard Markers (4-pack)", 5.0, 3.49),
    ("Ergonomic Mouse Pad", 1.0, 9.99),
    ("Sticky Notes 3x3 (12 pads)", 2.0, 7.99),
    ("HDMI Cable 2m", 3.0, 11.49),
    ("Safety Gloves Medium (pair)", 10.0, 2.99),
    ("Heavy Duty Tape 48mm", 6.0, 3.79),
    ("Cleaning Spray 500ml", 4.0, 5.49),
    ("AA Batteries (pack of 20)", 2.0, 8.49),
    ("Cardboard Boxes 40x40x40", 5.0, 4.19),
    ("Screwdriver Set 8-piece", 1.0, 18.99),
    ("Extension Lead 4-socket", 2.0, 14.99),
    ("Hand Sanitizer 500ml", 3.0, 6.79),
    ("Bubble Wrap Roll 50m", 1.0, 19.99),
    ("Packing Tape Dispenser", 2.0, 7.29),
]

TAX_RATES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]


def _random_date() -> str:
    year = random.randint(2022, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def _random_invoice_number() -> str:
    prefix = random.choice(["INV", "RCP", "BILL", "REF", "TXN"])
    number = random.randint(1000, 99999)
    return f"{prefix}-{number}"


def _generate_invoice_data() -> dict:
    vendor = random.choice(VENDORS)
    date = _random_date()
    invoice_number = _random_invoice_number()
    num_items = random.randint(1, 6)
    selected = random.sample(ITEMS, num_items)

    line_items = []
    for desc, default_qty, unit_price in selected:
        quantity = round(random.uniform(1, default_qty * 3), 0)
        quantity = max(1.0, quantity)
        price = round(unit_price * random.uniform(0.85, 1.15), 2)
        item_total = round(quantity * price, 2)
        line_items.append({
            "description": desc,
            "quantity": quantity,
            "unit_price": price,
            "total": item_total,
        })

    subtotal = round(sum(item["total"] for item in line_items), 2)
    tax_rate = random.choice(TAX_RATES)
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)

    return {
        "vendor": vendor,
        "date": date,
        "invoice_number": invoice_number,
        "line_items": line_items,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }


def _add_noise(image: Image.Image, level: float = 0.3) -> Image.Image:
    arr = np.array(image).astype(np.float32)

    if random.random() < level:
        angle = random.uniform(-3, 3)
        from PIL import Image as PILImage
        image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)
        arr = np.array(image).astype(np.float32)

    if random.random() < level:
        noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255)

    if random.random() < level * 0.5:
        from PIL import ImageFilter
        image = Image.fromarray(arr.astype(np.uint8))
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        arr = np.array(image).astype(np.float32)

    if random.random() < level * 0.3:
        brightness = random.uniform(0.75, 1.1)
        arr = np.clip(arr * brightness, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def _render_invoice(data: dict, width: int = 794, height: int = 1123) -> Image.Image:
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    try:
        font_large = ImageFont.truetype("arial.ttf", 22)
        font_medium = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
        font_bold = ImageFont.truetype("arialbd.ttf", 18)
    except OSError:
        font_large = ImageFont.load_default()
        font_medium = font_large
        font_small = font_large
        font_bold = font_large

    margin = 60
    y = margin

    draw.text((margin, y), "INVOICE", font=font_large, fill=(20, 20, 20))
    draw.text((width - margin - 200, y), data["vendor"], font=font_medium, fill=(40, 40, 40))
    y += 40

    draw.line([(margin, y), (width - margin, y)], fill=(180, 180, 180), width=1)
    y += 15

    draw.text((margin, y), f"Invoice No: {data['invoice_number']}", font=font_medium, fill=(60, 60, 60))
    draw.text((margin, y + 22), f"Date: {data['date']}", font=font_medium, fill=(60, 60, 60))
    y += 60

    draw.line([(margin, y), (width - margin, y)], fill=(200, 200, 200), width=1)
    y += 10

    headers = ["Description", "Qty", "Unit Price", "Total"]
    col_x = [margin, margin + 300, margin + 400, margin + 520]
    for i, header in enumerate(headers):
        draw.text((col_x[i], y), header, font=font_bold, fill=(30, 30, 30))
    y += 25

    draw.line([(margin, y), (width - margin, y)], fill=(200, 200, 200), width=1)
    y += 10

    for item in data["line_items"]:
        draw.text((col_x[0], y), item["description"][:40], font=font_small, fill=(50, 50, 50))
        draw.text((col_x[1], y), str(int(item["quantity"])), font=font_small, fill=(50, 50, 50))
        draw.text((col_x[2], y), f"${item['unit_price']:.2f}", font=font_small, fill=(50, 50, 50))
        draw.text((col_x[3], y), f"${item['total']:.2f}", font=font_small, fill=(50, 50, 50))
        y += 22

    y += 10
    draw.line([(margin, y), (width - margin, y)], fill=(200, 200, 200), width=1)
    y += 15

    right_x = width - margin - 200
    draw.text((right_x, y), f"Subtotal: ${data['subtotal']:.2f}", font=font_medium, fill=(50, 50, 50))
    y += 22
    draw.text((right_x, y), f"Tax:      ${data['tax']:.2f}", font=font_medium, fill=(50, 50, 50))
    y += 22
    draw.text((right_x, y), f"Total:    ${data['total']:.2f}", font=font_bold, fill=(10, 10, 10))

    return image


def generate_synthetic_dataset(
    output_dir: Path,
    n_train: int = 1500,
    n_val: int = 250,
    n_test: int = 250,
    noise_level: float = 0.3,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    splits = {"train": n_train, "val": n_val, "test": n_test}

    for split, count in splits.items():
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            doc_id = str(uuid.uuid4())[:8]
            data = _generate_invoice_data()

            image = _render_invoice(data)
            if split in ("train", "val"):
                image = _add_noise(image, level=noise_level)

            image.save(img_dir / f"{doc_id}.png")

            with open(lbl_dir / f"{doc_id}.json", "w") as f:
                json.dump(data, f, indent=2)

        print(f"Generated {count} examples for split: {split}")


def load_sroie_split(sroie_root: Path, split: str = "train") -> list[dict]:
    sroie_root = Path(sroie_root)
    split_dir = sroie_root / split
    img_dir = split_dir / "img"
    entities_dir = split_dir / "entities"

    if not img_dir.exists():
        return []

    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        entity_path = entities_dir / img_path.with_suffix(".txt").name
        if not entity_path.exists():
            continue

        raw = {}
        with open(entity_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, _, value = line.partition(":")
                    raw[key.strip().lower()] = value.strip()

        label = {
            "vendor": raw.get("company"),
            "date": raw.get("date"),
            "invoice_number": None,
            "line_items": [],
            "subtotal": None,
            "tax": None,
            "total": _parse_amount(raw.get("total")),
        }

        samples.append({"image_path": str(img_path), "label": label})

    return samples


def _parse_amount(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = "".join(c for c in value if c.isdigit() or c == ".")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None
