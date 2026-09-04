import os, sys, random, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFont, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "synthetic_hsrp"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = DATASET_DIR / "samples"
SAMPLE_DIR.mkdir(exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\bahnschrift.ttf"

INDIAN_STATES = [
    "DL", "UP", "HR", "MH", "KA", "RJ", "PB", "GJ", "CH", "UK", "BR", "TS", "TN", "KL", "WB", "BH", "OD",
    "AP", "MP", "JH", "AS", "CG", "GA", "HP", "JK", "LA", "ML", "MN", "MZ", "NL", "PY", "SK", "TR"
]

# HEAVY TARGETED CONFUSION SETS (User specifically requested: O, Q, D, 2, 4, V, Y, W, U, and Commercial)
# Extreme focus on O vs Q vs D pairs:
OQD_SERIES = [
    "DQ", "QD", "OD", "DO", "OQ", "QO", "QQ", "DD", "OO",
    "DU", "UD", "QU", "UQ", "OU", "UO",
    "DA", "QA", "OA", "AD", "AQ", "AO",
    "DB", "QB", "OB", "BD", "BQ", "BO",
    "DC", "QC", "OC", "CD", "CQ", "CO",
    "DE", "QE", "OE", "ED", "EQ", "EO",
    "DP", "QP", "OP", "DT", "QT", "OT",
    "D", "Q", "O", "U"
]

# V, Y, W, U series pairs
VYWU_SERIES = [
    "VY", "YV", "VW", "WV", "VU", "UV", "WY", "YW", "UY", "YU", "WW", "VV", "YY", "UU",
    "DV", "DW", "DY", "DU", "EV", "EW", "EY", "EU", "BV", "BW", "BY", "BU",
    "QV", "QW", "QY", "OV", "OW", "OY",
    "VA", "WA", "YA", "UA", "AV", "AW", "AY", "AU",
    "V", "Y", "W", "U"
]

# Commercial series (Taxis, Buses, Auto-rickshaws, Commercial Trucks)
COMMERCIAL_SERIES = [
    "T", "TA", "TB", "TC", "TD", "TQ", "TV", "TW", "TY", "TX", "TR",
    "P", "PA", "PB", "PC", "PD", "PQ", "PT", "PV",
    "F", "FA", "FB", "FC", "FD", "FQ",
    "G", "GA", "GB", "GC", "GD", "GQ",
    "AR", "AT", "AU", "AV", "AQ", "AD",
    "E", "EA", "EB", "ED", "EQ", "ET", "EV"
]

B8SZ_SERIES = [
    "BB", "BS", "BZ", "SB", "SS", "SZ", "ZB", "ZS", "ZZ", "BE", "EB", "SE", "ES",
    "B", "S", "Z"
]

# Targeted numbers: 2 & 4 heavy, triplets & quadruplets (like 5551, 2224, 4442)
TARGETED_NUMBERS = [
    # Consecutive triplets & quadruplets (preventing CTC collapse)
    "5551", "5555", "5552", "5550", "2555", "1555", "4555", "3555", "7555",
    "2224", "4442", "2222", "4444", "2221", "4441", "2228", "4448",
    "0001", "0007", "0008", "0002", "0004", "0009", "1111", "1118", "1112", "1114",
    "8888", "8882", "8884", "8881", "9999", "7777", "3333", "6666",
    # 2 & 4 heavy combinations
    "2424", "4242", "2244", "4422", "2442", "4224", "2444", "4222",
    "2418", "4201", "1428", "2468", "4289", "2024", "2026", "2451", "4255", "2541", "4244", "2422"
]


def generate_plate_string():
    """Generates realistic Indian license plate strings with targeted confusion distributions."""
    # 20% Odisha (OD), 12% DL, 12% UP, 8% HR, 8% MH, 8% KA, 8% GJ, rest others
    r_state = random.random()
    if r_state < 0.20:
        st = "OD"
    elif r_state < 0.32:
        st = "DL"
    elif r_state < 0.44:
        st = "UP"
    elif r_state < 0.52:
        st = "HR"
    elif r_state < 0.60:
        st = "MH"
    elif r_state < 0.68:
        st = "KA"
    elif r_state < 0.76:
        st = "GJ"
    else:
        st = random.choice(INDIAN_STATES)

    # Bharat Series (BH)
    if st == "BH":
        yr = f"{random.randint(21, 26):02d}"
        num = f"{random.randint(1000, 9999):04d}"
        ser = random.choice(["AA", "AB", "DQ", "VW", "UY", "OD", "QD", "OQ", "TV"])
        return f"{yr}BH{num}{ser}"

    # District code: 2 & 4 over-sampled
    r_dist = random.random()
    if st == "OD":
        if r_dist < 0.35:
            dist = "26"  # Specific user district (Bargarh)
        elif r_dist < 0.60:
            dist = random.choice(["02", "04", "14", "24", "12", "22", "34", "05"])
        else:
            dist = f"{random.randint(1, 35):02d}"
    else:
        if r_dist < 0.35:
            dist = random.choice(["24", "42", "02", "04", "12", "14", "22", "44", "26", "46", "28", "48"])
        else:
            dist = f"{random.randint(1, 99):02d}"

    # Series code: heavily sample OQD (35%), Commercial (25%), VYWU (20%), B8SZ (10%), general (10%)
    r_ser = random.random()
    if r_ser < 0.35:
        ser = random.choice(OQD_SERIES)
    elif r_ser < 0.60:
        ser = random.choice(COMMERCIAL_SERIES)
    elif r_ser < 0.80:
        ser = random.choice(VYWU_SERIES)
    elif r_ser < 0.90:
        ser = random.choice(B8SZ_SERIES)
    else:
        ser_len = random.choice([1, 2])
        ser = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=ser_len))

    # Number: heavily sample targeted patterns (50%), 2/4/0/1/5/8 numbers (30%), general (20%)
    r_num = random.random()
    if r_num < 0.50:
        num = random.choice(TARGETED_NUMBERS)
    elif r_num < 0.80:
        num = "".join(random.choices(["2", "4", "5", "0", "1", "8", "6", "9"], k=4))
    else:
        num = f"{random.randint(1, 9999):04d}"

    return f"{st}{dist}{ser}{num}"


def render_plate_to_gray(text_raw, width=160, height=32):
    """
    Renders Indian HSRP plate directly to (32, 160) grayscale array.
    Supports all Indian plate categories:
    - 48% Private White Plate (Black chars)
    - 35% Commercial Yellow Plate (Black chars - Taxis, Buses, Autos, Trucks)
    - 10% Commercial EV Green Plate (Yellow/White chars - Electric buses, cabs, autos)
    - 7% Commercial Rental Black Plate (Yellow chars - Self-drive rentals)
    """
    W, H = width * 2, height * 2
    r_cat = random.random()

    if r_cat < 0.35:
        # Commercial Yellow Plate (Taxis, Autos, Buses, Commercial Trucks)
        bg = (random.randint(240, 255), random.randint(195, 225), random.randint(15, 45))
        char_col = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
        border_col = (40, 40, 40)
    elif r_cat < 0.45:
        # Commercial EV Green Plate (Electric buses, BluSmart cabs, electric autos)
        bg = (random.randint(15, 35), random.randint(120, 160), random.randint(30, 65))
        # Commercial EVs have yellow lettering; private EVs have white
        if random.random() < 0.70:
            char_col = (random.randint(240, 255), random.randint(210, 240), random.randint(20, 50))
        else:
            char_col = (random.randint(235, 255), random.randint(235, 255), random.randint(235, 255))
        border_col = (200, 220, 200)
    elif r_cat < 0.52:
        # Commercial Rental / Self-Drive Black Plate (Zoomcar, luxury rentals)
        bg = (random.randint(20, 40), random.randint(20, 40), random.randint(20, 40))
        char_col = (random.randint(240, 255), random.randint(205, 235), random.randint(20, 50))
        border_col = (180, 180, 180)
    else:
        # Standard Private White Plate
        v = random.randint(235, 255)
        bg = (v, v, v)
        char_col = (random.randint(10, 35), random.randint(10, 35), random.randint(10, 35))
        border_col = (40, 40, 40)

    img = Image.new("RGB", (W, H), color=bg)
    draw = ImageDraw.Draw(img)

    # Outer border
    draw.rounded_rectangle([2, 2, W - 3, H - 3], radius=4, outline=border_col, width=2)

    # Blue IND strip on left
    strip_w = int(W * 0.08)
    draw.rectangle([3, 3, strip_w, H - 4], fill=(0, 55, 150))
    # IND text & hologram circle
    ind_font = ImageFont.truetype(FONT_PATH, 11)
    draw.ellipse([strip_w // 2 - 5, 8, strip_w // 2 + 5, 18], fill=(100, 160, 220), outline=(200, 220, 255))
    draw.text((4, H - 18), "IND", font=ind_font, fill=(255, 255, 255))

    # Realistic spacing formats
    st = text_raw[:2]
    dist = text_raw[2:4]
    if len(text_raw) >= 6 and text_raw[4:6].isalpha():
        ser = text_raw[4:6]
        num = text_raw[6:]
    else:
        ser = text_raw[4:5]
        num = text_raw[5:]

    sp_choice = random.random()
    if sp_choice < 0.50:
        display_str = f"{st} {dist} {ser} {num}"
    elif sp_choice < 0.70:
        display_str = f"{st}{dist} {ser} {num}"
    elif sp_choice < 0.85:
        display_str = f"{st} {dist}{ser}{num}"
    else:
        display_str = text_raw

    # Font sizing & measurement
    font_size = 40
    font = ImageFont.truetype(FONT_PATH, font_size)
    bbox = draw.textbbox((0, 0), display_str, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    avail_w = W - strip_w - 14
    if tw > avail_w:
        font_size = int(40 * (avail_w / tw))
        font = ImageFont.truetype(FONT_PATH, font_size)
        bbox = draw.textbbox((0, 0), display_str, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

    sx = strip_w + 6 + max(0, int((avail_w - tw) / 2))
    sy = int((H - th) / 2) - 2

    draw.text((sx, sy), display_str, font=font, fill=char_col)

    # Screws / rivets (like real plates)
    if random.random() < 0.70:
        for x_screw in [int(W * random.uniform(0.24, 0.32)), int(W * random.uniform(0.68, 0.76))]:
            y_screw = random.randint(6, 12)
            draw.ellipse([x_screw - 4, y_screw - 4, x_screw + 4, y_screw + 4], fill=(160, 160, 160), outline=(70, 70, 70))

    # Downscale smoothly to 160x32
    img = img.resize((width, height), Image.Resampling.BICUBIC)

    # Augmentations: slight blur
    if random.random() < 0.40:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.20, 0.65)))

    # Convert to grayscale
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # Random contrast / brightness jitter
    if random.random() < 0.35:
        factor = random.uniform(0.85, 1.15)
        arr = np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return arr


def build_synthetic_dataset(num_train=60000, num_val=2500):
    print("=" * 70, flush=True)
    print("GENERATING COMPREHENSIVE INDIAN HSRP & COMMERCIAL DATASET", flush=True)
    print("Categories: 48% Private White | 35% Commercial Yellow | 10% EV Green | 7% Rental Black", flush=True)
    print("Targeted Confusions: O vs Q vs D | 2 vs 4 | V, Y, W, U | Triplets (5551) | OD State", flush=True)
    print(f"Target: {num_train:,} Train Plates + {num_val:,} Validation Plates", flush=True)
    print(f"Destination: {DATASET_DIR}", flush=True)
    print("=" * 70, flush=True)

    # 1. Validation Set
    print(f"\n[1/2] Generating validation set ({num_val:,} plates)...", flush=True)
    val_images = np.zeros((num_val, 1, 32, 160), dtype=np.uint8)
    val_labels = []
    t0 = time.time()

    for i in range(num_val):
        t = generate_plate_string()
        arr = render_plate_to_gray(t)
        val_images[i, 0] = arr
        val_labels.append(t)
        if i < 50:
            Image.fromarray(arr).save(SAMPLE_DIR / f"val_{i:03d}_{t}.png")

    torch.save(
        {"images": torch.from_numpy(val_images), "labels": val_labels},
        DATASET_DIR / "hsrp_val_2k.pt"
    )
    print(f"   Saved hsrp_val_2k.pt ({num_val} plates) in {time.time() - t0:.2f}s", flush=True)

    # 2. Training Set
    print(f"\n[2/2] Generating training set ({num_train:,} plates)...", flush=True)
    train_images = np.zeros((num_train, 1, 32, 160), dtype=np.uint8)
    train_labels = []
    t0 = time.time()

    for i in range(num_train):
        t = generate_plate_string()
        arr = render_plate_to_gray(t)
        train_images[i, 0] = arr
        train_labels.append(t)
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"   Generated {i + 1:,}/{num_train:,} plates ({rate:.0f} plates/s)...", flush=True)

    torch.save(
        {"images": torch.from_numpy(train_images), "labels": train_labels},
        DATASET_DIR / "hsrp_train_60k.pt"
    )
    total_time = time.time() - t0
    print(f"   Saved hsrp_train_60k.pt in {total_time:.2f}s ({num_train / total_time:.0f} plates/s)", flush=True)
    print("\nDataset generation complete!", flush=True)
    print(f"Inspect sample visual plates at: {SAMPLE_DIR}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    build_synthetic_dataset()
