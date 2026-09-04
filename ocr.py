import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import config


CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-"
]


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    """Deep Neural Network trained specifically on Indian License Plate characters."""
    def __init__(self, lpr_max_len=18, phase=False, class_num=len(CHARS), dropout_rate=0):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            small_basic_block(ch_in=256, ch_out=256),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1),
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=448 + self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)
        return logits


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size(1), recurrent.size(0), recurrent.size(2)
        t_rec = recurrent.contiguous().view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(b, T, -1)
        return output


class CRNN_BiLSTM(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=37, nh=256):
        super(CRNN_BiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)
        output = self.rnn(conv)
        return output.permute(1, 0, 2)


CRNN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_cached_crnn_model = None

def get_crnn_model():
    """Returns the cached CRNN-BiLSTM model (loaded once, reused across frames)."""
    global _cached_crnn_model
    if _cached_crnn_model is not None:
        return _cached_crnn_model
    crnn_path = getattr(config, "CRNN_MODEL_PATH", config.BASE_DIR / "best_crnn_bilstm.pth")
    if not crnn_path.exists():
        return None
    try:
        device = torch.device(config.DEVICE if isinstance(config.DEVICE, str) else f"cuda:{config.DEVICE}")
        model = CRNN_BiLSTM(imgH=32, nc=1, nclass=len(CRNN_CHARS) + 1, nh=256)
        model.load_state_dict(torch.load(str(crnn_path), map_location=device))
        model.to(device).eval()
        _cached_crnn_model = model
        print(f"[OCR] Loaded CRNN-BiLSTM singleton from {crnn_path.name}")
        return model
    except Exception as e:
        print(f"[WARNING] Could not load CRNN-BiLSTM singleton: {e}")
        return None


def create_lprnet_model():
    """Initializes deep plate recognition neural network (CRNN-BiLSTM or LPRNet)."""
    device = torch.device(config.DEVICE if isinstance(config.DEVICE, str) else f"cuda:{config.DEVICE}")

    # Check for modern CRNN-BiLSTM weights first
    crnn_path = getattr(config, "CRNN_MODEL_PATH", config.BASE_DIR / "best_crnn_bilstm.pth")
    if crnn_path.exists():
        try:
            model = CRNN_BiLSTM(imgH=32, nc=1, nclass=len(CRNN_CHARS) + 1, nh=256)
            model.load_state_dict(torch.load(str(crnn_path), map_location=device))
            model.to(device)
            model.eval()
            print(f"[OCR] Loaded CRNN-BiLSTM model from {crnn_path.name}")
            return model
        except Exception as e:
            print(f"[WARNING] Could not load CRNN-BiLSTM: {e}")

    # Fallback to legacy LPRNet if available
    weights_path = config.LPRNET_MODEL_PATH
    if not weights_path.exists():
        return None

    model = LPRNet(lpr_max_len=18, phase=False, class_num=len(CHARS), dropout_rate=0)
    try:
        model.load_state_dict(torch.load(str(weights_path), map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[WARNING] Could not load LPRNet: {e}")
        return None


_PARSEQ_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


def create_parseq_model():
    """Loads SOTA PARSeq Vision Transformer model on CUDA GPU if available."""
    try:
        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True, trust_repo=True).eval().to(config.DEVICE)
        return model
    except Exception as e:
        print(f"[WARNING] Could not load PARSeq ViT: {e}")
        return None


def predict_parseq(parseq_model, crop):
    """Runs PARSeq Vision Transformer inference on plate crop."""
    if parseq_model is None or crop is None or crop.size == 0:
        return "", 0.0
    try:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(crop_rgb)
        device = next(parseq_model.parameters()).device
        t = _PARSEQ_TRANSFORM(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = parseq_model(t)
            probs = logits.softmax(-1)
            preds, probs_val = parseq_model.tokenizer.decode(probs)
        raw = preds[0] if preds else ""
        conf = float(probs_val[0].cumprod(-1)[-1]) if (probs_val and len(probs_val[0]) > 0) else 0.85
        return raw, conf
    except Exception:
        return "", 0.0


def predict_lprnet(lpr_model, crop):
    """Runs deep learning inference on a tight plate crop (CRNN-BiLSTM or legacy LPRNet)."""
    if lpr_model is None or crop is None or crop.size == 0:
        return "", 0.0

    device = next(lpr_model.parameters()).device

    # Modern CRNN-BiLSTM inference with CTC decoding
    if isinstance(lpr_model, CRNN_BiLSTM):
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        resized = cv2.resize(gray, (160, 32))
        t_img = (torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0).to(device)

        with torch.no_grad():
            logits = lpr_model(t_img)
            T, B, C = logits.size()
            probs = logits.softmax(dim=-1)
            max_probs, max_indices = probs.max(dim=-1)

            max_indices = max_indices.squeeze(1).cpu().numpy()
            max_probs = max_probs.squeeze(1).cpu().numpy()

            output_chars = []
            confs = []
            prev = 0
            for t in range(T):
                idx = max_indices[t]
                if idx != 0 and idx != prev:
                    output_chars.append(CRNN_CHARS[idx - 1])
                    confs.append(float(max_probs[t]))
                prev = idx

            pred_str = "".join(output_chars)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return pred_str, avg_conf

    # Legacy LPRNet fallback
    img = cv2.resize(crop, (94, 24)).astype("float32")
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    t_img = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = lpr_model(t_img).squeeze(0).detach().cpu().numpy()

    exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    output_chars = []
    confs = []
    prev_c = -1
    for i in range(logits.shape[1]):
        c = np.argmax(logits[:, i])
        p = float(probs[c, i])
        if c != prev_c and c < len(CHARS) and CHARS[c] != "-":
            output_chars.append(CHARS[c])
            confs.append(p)
        prev_c = c

    pred_str = "".join(output_chars)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return pred_str, avg_conf


def create_ocr_reader():
    """Initializes the OCR engine: RapidOCR (PP-OCRv4) if available, falling back to EasyOCR."""
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except ImportError:
        pass

    try:
        import easyocr
    except ImportError as error:
        raise RuntimeError(
            "Neither RapidOCR nor EasyOCR is installed. Run pip install rapidocr-onnxruntime."
        ) from error

    config.OCR_MODEL_DIR.mkdir(exist_ok=True)
    return easyocr.Reader(
        ["en"],
        gpu=config.GPU_AVAILABLE,
        model_storage_directory=str(config.OCR_MODEL_DIR),
        verbose=False,
    )


def normalize_state_code(code: str) -> str:
    """Standardizes state prefixes using exact mappings and minimum visual substitution distance."""
    code = code.upper()
    if code in config.INDIAN_STATE_CODES:
        return code
    if code in config.STATE_PREFIX_CORRECTIONS:
        return config.STATE_PREFIX_CORRECTIONS[code]

    # Minimum visual substitution distance against all 36 valid Indian state codes
    best_code = code
    min_cost = 999.0
    for valid in config.INDIAN_STATE_CODES:
        cost = 0.0
        for c1, c2 in zip(code, valid):
            if c1 == c2:
                continue
            cost += config.VISUAL_CONFUSION_COSTS.get((c1, c2), 1.0)
        if cost < min_cost:
            min_cost = cost
            best_code = valid

    # Only snap if the substitution distance is convincingly small (<= 0.25)
    if min_cost <= 0.25:
        return best_code

    return code


def correct_chars(text: str, target_types: str) -> str:
    """Positional character disambiguation."""
    if len(text) != len(target_types):
        return text
    result = []
    for i, (ch, t) in enumerate(zip(text, target_types)):
        if t == "D":
            if i == 3 and ch in ("L", "A", "H"):
                result.append("4")
            else:
                result.append(config.CHAR_TO_DIGIT.get(ch, ch))
        elif t == "S":
            result.append(config.DIGIT_TO_SERIES_CHAR.get(ch, ch))
        elif t == "L":
            result.append(config.DIGIT_TO_CHAR.get(ch, ch))
        else:
            result.append(ch)
    return "".join(result)


def parse_and_score_plate(raw_text: str):
    """
    Evaluates raw OCR text against standard Indian license plate syntactic models
    and returns (best_formatted_text, confidence_bonus, pattern_name).
    Rejects any text that does not match an authentic Indian state code and format.
    """
    if not raw_text:
        return "", 0.0, "empty"

    text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    # Strip country / security badge prefixes
    for pfx in ["INDIA", "HSRP", "HRSP", "IND"]:
        if text.startswith(pfx) and len(text) >= len(pfx) + 6:
            text = text[len(pfx):]
            break

    # Strip single leading or trailing noise character ONLY when text is longer than 10 characters
    if len(text) >= 11:
        if text[:2] not in config.INDIAN_STATE_CODES and (text[1:3] in config.INDIAN_STATE_CODES or text[1:3] in config.STATE_PREFIX_CORRECTIONS):
            text = text[1:]
        elif (text[:2] in config.INDIAN_STATE_CODES or text[:2] in config.STATE_PREFIX_CORRECTIONS) and len(text) == 11:
            # Trailing noise character (e.g. UP14DU3952H -> UP14DU3952)
            text = text[:10]

    if len(text) > 12:
        text = text[:12]

    if len(text) < 4:
        return "", 0.0, "too_short"

    candidates = []

    # 1. Delhi 10-char: DL + 1-digit RTO + 1-letter Category + 2-letter Series + 4 digits (e.g. DL 3C CM 2418, DL 3C CP 6535)
    if len(text) == 10:
        st = normalize_state_code(text[:2])
        if st == "DL":
            c_dl = correct_chars(text, "LLDSSSDDDD")
            cand = f"DL{c_dl[2]}{c_dl[3]}{c_dl[4:6]}{c_dl[6:]}"
            if cand[2].isdigit() and cand[3].isalpha() and cand[4:6].isalpha() and cand[6:].isdigit():
                candidates.append((cand, 0.55, "Delhi_10_1RTO"))

    # 2. Standard 10-char: LL DD SS DDDD (e.g. UP16BZ4237, UP25CS9120, UP32LM1218, HR51BE1188, UP14FJ4529, UP12AA7855)
    if len(text) == 10:
        c_std = correct_chars(text, "LLDDSSDDDD")
        state = normalize_state_code(c_std[:2])
        rto = c_std[2:4]
        series = c_std[4:6]
        digits = c_std[6:]
        cand = f"{state}{rto}{series}{digits}"
        if state in config.INDIAN_STATE_CODES and rto.isdigit() and series.isalpha() and digits.isdigit():
            candidates.append((cand, 0.45, "Standard_10"))

    # 3. Bharat Series 10-char: DD BH DDDD SS (e.g. 22BH1234AA)
    if len(text) == 10 and ("BH" in text[1:4]):
        c_bh = correct_chars(text, "DDLLDDDDSS")
        if c_bh[:2].isdigit() and c_bh[2:4] == "BH" and c_bh[4:8].isdigit() and c_bh[8:].isalpha():
            candidates.append((c_bh, 0.45, "Bharat_10"))

    # 4. Standard 9-char formats:
    if len(text) == 9:
        st = normalize_state_code(text[:2])
        if st == "DL":
            c_dl9 = correct_chars(text, "LLDSSDDDD")
            cand = f"DL{c_dl9[2]}{c_dl9[3]}{c_dl9[4]}{c_dl9[5:]}"
            if cand[2].isdigit() and cand[3:5].isalpha() and cand[5:].isdigit():
                candidates.append((cand, 0.45, "Delhi_9"))

        # Case B: LL DD S DDDD (e.g. UP 78 H 3674)
        c1 = correct_chars(text, "LLDDSDDDD")
        state1 = normalize_state_code(c1[:2])
        cand1 = f"{state1}{c1[2:]}"
        if state1 in config.INDIAN_STATE_CODES and cand1[2:4].isdigit() and cand1[4].isalpha() and cand1[5:].isdigit():
            candidates.append((cand1, 0.40, "Std_9_1Series"))

        # Case C: LL DD SS DDD
        c3 = correct_chars(text, "LLDDSSDDD")
        state3 = normalize_state_code(c3[:2])
        cand3 = f"{state3}{c3[2:]}"
        if state3 in config.INDIAN_STATE_CODES and cand3[2:4].isdigit() and cand3[4:6].isalpha() and cand3[6:].isdigit():
            candidates.append((cand3, 0.35, "Std_9_3Num"))

    # 5. Standard 8-char: LL DD DDDD
    if len(text) == 8:
        c = correct_chars(text, "LLDDDDDD")
        state = normalize_state_code(c[:2])
        cand = f"{state}{c[2:]}"
        if state in config.INDIAN_STATE_CODES and cand[2:4].isdigit() and cand[4:].isdigit():
            candidates.append((cand, 0.20, "Std_8"))

    valid_candidates = []
    for cand, bonus, pat in candidates:
        if pat == "Bharat_10" and cand[2:4] == "BH":
            valid_candidates.append((cand, bonus, pat))
        elif cand[:2] in config.INDIAN_STATE_CODES:
            valid_candidates.append((cand, bonus, pat))

    if not valid_candidates:
        return "", 0.0, "invalid"

    return max(valid_candidates, key=lambda c: c[1])


def assemble_ocr_detections(detections):
    """
    Groups OCR bounding boxes into lines and deduplicates overlapping characters.
    """
    if not detections:
        return []

    parsed = []
    for item in detections:
        if len(item) == 3:
            box, text, conf = item
        elif len(item) == 2:
            box, text = item
            conf = 0.5
        else:
            continue

        clean = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not clean:
            continue

        box = np.array(box)
        min_x = float(np.min(box[:, 0]))
        max_x = float(np.max(box[:, 0]))
        min_y = float(np.min(box[:, 1]))
        max_y = float(np.max(box[:, 1]))
        center_y = (min_y + max_y) / 2.0
        center_x = (min_x + max_x) / 2.0
        box_h = max(max_y - min_y, 1.0)
        box_w = max(max_x - min_x, 1.0)
        char_w = box_w / max(1, len(clean))
        parsed.append({
            "text": clean,
            "conf": float(conf),
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "center_x": center_x,
            "center_y": center_y,
            "box_h": box_h,
            "box_w": box_w,
            "char_w": char_w,
        })

    if not parsed:
        return []

    parsed.sort(key=lambda d: d["center_y"])
    avg_h = sum(d["box_h"] for d in parsed) / len(parsed)

    lines = []
    current_line = [parsed[0]]
    for item in parsed[1:]:
        if abs(item["center_y"] - current_line[0]["center_y"]) < (avg_h * 0.45):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    if current_line:
        lines.append(current_line)

    full_text_parts = []
    conf_scores = []
    for line in lines:
        line.sort(key=lambda d: d["min_x"])
        line_tokens = []
        for i, token in enumerate(line):
            text_tok = token["text"]
            if i > 0:
                prev_token = line[i - 1]
                overlap = prev_token["max_x"] - token["min_x"]
                avg_char_w = (prev_token["char_w"] + token["char_w"]) / 2.0
                if overlap >= 0.35 * avg_char_w:
                    num_overlap = max(1, int(round(overlap / max(1.0, avg_char_w))))
                    if prev_token["conf"] < token["conf"]:
                        if line_tokens and len(line_tokens[-1]) > num_overlap:
                            line_tokens[-1] = line_tokens[-1][:-num_overlap]
                    else:
                        if len(text_tok) > num_overlap:
                            text_tok = text_tok[num_overlap:]
            line_tokens.append(text_tok)
            conf_scores.append(token["conf"])

        full_text_parts.append("".join(line_tokens))

    assembled_text = "".join(full_text_parts)
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    candidates = [(assembled_text, avg_conf)]
    for item in parsed:
        if len(item["text"]) >= 6:
            candidates.append((item["text"], item["conf"]))

    return candidates


def deskew_plate_crop(img):
    """
    Detects plate skew/rotation angle and rectifies it upright using Hough line and contour analysis.
    """
    if img is None or img.size == 0:
        return img, 0.0
    h, w = img.shape[:2]
    if w < 35 or h < 14:
        return img, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(15, int(w * 0.28)),
        minLineLength=max(18, int(w * 0.22)), maxLineGap=10
    )

    angles = []
    if lines is not None:
        for line in lines:
            pts = line.ravel()
            if len(pts) >= 4:
                x1, y1, x2, y2 = pts[:4]
                dx = x2 - x1
                dy = y2 - y1
                if dx != 0:
                    ang = np.degrees(np.arctan2(dy, dx))
                    if -35.0 <= ang <= 35.0 and abs(ang) > 2.0:
                        angles.append(ang)

    if not angles:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (w * h * 0.20):
                rect = cv2.minAreaRect(cnt)
                ang = rect[-1]
                if ang < -45:
                    ang = 90 + ang
                if -35.0 <= ang <= 35.0 and abs(ang) > 2.0:
                    angles.append(ang)

    if not angles:
        return img, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) > 25.0 or abs(median_angle) < 2.0:
        return img, 0.0

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle


def enhance_plate_variants(plate_crop):
    """
    Generates fast, high-clarity image variants for OCR:
    Variant 1: 2.5x Lanczos Upscale + LAB CLAHE (luminance contrast, shadow removal)
    Variant 2: Denoised Grayscale + CLAHE + Unsharp Mask (sharp stroke edges)
    """
    if plate_crop is None or plate_crop.size == 0:
        return []

    h, w = plate_crop.shape[:2]
    target_h = max(110, min(160, int(h * 2.8)))
    scale = target_h / max(h, 1)
    target_w = max(240, int(w * scale))

    resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Variant 1: LAB Color CLAHE (natural character separation without binary artifacting)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    color_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Variant 2: Bilateral Denoising + Gray CLAHE + Unsharp Masking
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
    clahe_gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(denoised)
    sharp = cv2.addWeighted(clahe_gray, 1.5, cv2.GaussianBlur(clahe_gray, (0, 0), 1.2), -0.5, 0)

    return [color_clahe, sharp]


def compute_image_sharpness(img):
    """Computes image sharpness using Laplacian variance."""
    if img is None or img.size == 0:
        return 0.0
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_plate_text(ocr_reader, frame, plate_box, lpr_model=None):
    """
    Ensemble ANPR Recognition Engine:
    1. Automatic Deskew & Normalization
    2. Aspect-Ratio Branching:
       - Multi-line plates (AR < 2.8): 2-line clustering (EasyOCR top+bottom lines, split LPRNet fallback)
       - Standard 1-line plates (AR >= 2.8): LPRNet deep learning + Fast 2-pass EasyOCR
    3. Syntactic parsing and confidence scoring
    """
    x1, y1, x2, y2 = plate_box[:4]
    h_frame, w_frame = frame.shape[:2]

    # Reject tiny distant crops that cannot be read accurately
    bw = x2 - x1
    bh = y2 - y1
    min_w = getattr(config, "MIN_PLATE_OCR_WIDTH", 65)
    min_h = getattr(config, "MIN_PLATE_OCR_HEIGHT", 18)
    if bw < min_w or bh < min_h:
        return "", 0.0, 0.0, None, 0.0

    # Tight crop with safe edge padding to prevent stroke clipping (e.g. legs of 'M')
    pad_h = max(2, int(bh * 0.08))
    pad_w = max(2, int(bw * 0.025))
    ty1 = max(0, y1 - pad_h)
    ty2 = min(h_frame, y2 + pad_h)
    tx1 = max(0, x1 - pad_w)
    tx2 = min(w_frame, x2 + pad_w)
    tight_crop = frame[ty1:ty2, tx1:tx2]
    if tight_crop.size == 0:
        return "", 0.0, 0.0, tight_crop, 0.0

    # Deskew tight crop if tilted
    tight_crop, skew_angle = deskew_plate_crop(tight_crop)
    sharpness = compute_image_sharpness(tight_crop)

    # Context padded crop for EasyOCR
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.18)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w_frame, x2 + pad_x)
    cy2 = min(h_frame, y2 + pad_y)
    padded_crop = frame[cy1:cy2, cx1:cx2]
    if padded_crop.size > 0 and abs(skew_angle) > 2.0:
        padded_crop, _ = deskew_plate_crop(padded_crop)

    # High-Performance SOTA Ensemble: PARSeq Vision Transformer + RapidOCR (PP-OCRv4)
    p_fmt = ""
    p_conf = 0.0
    aspect = bw / float(max(1, bh))
    if lpr_model is not None and hasattr(lpr_model, "tokenizer"):
        # 1. Primary Attempt: Standard 1-line reading on full tight crop
        p_raw, p_conf = predict_parseq(lpr_model, tight_crop)
        p_fmt, p_bonus, _ = parse_and_score_plate(p_raw)

        # 2. Fallback: If 1-line failed and aspect is square (aspect < 2.45), attempt 2-line split
        if (not p_fmt or p_bonus <= 0.0) and aspect < 2.45 and tight_crop.shape[0] >= 25:
            h_c = tight_crop.shape[0]
            top_h = tight_crop[:int(h_c * 0.55), :]
            bot_h = tight_crop[int(h_c * 0.45):, :]
            t_raw, t_conf = predict_parseq(lpr_model, top_h)
            b_raw, b_conf = predict_parseq(lpr_model, bot_h)
            split_raw = t_raw + b_raw
            split_fmt, split_bonus, _ = parse_and_score_plate(split_raw)
            if split_fmt and split_bonus > 0.0:
                p_raw = split_raw
                p_conf = (t_conf + b_conf) / 2.0
                p_fmt = split_fmt

    is_rapid = hasattr(ocr_reader, "text_sys") or type(ocr_reader).__name__ == "RapidOCR"
    if is_rapid:
        r_fmt = ""
        r_conf = 0.0
        crops_to_check = [tight_crop]
        if padded_crop is not None and padded_crop.size > 0:
            crops_to_check.append(padded_crop)

        for c in crops_to_check:
            if c is None or c.size == 0:
                continue
            rapid_results, _ = ocr_reader(c)
            if rapid_results:
                sorted_boxes = sorted(rapid_results, key=lambda b: (b[0][0][1] // 15, b[0][0][0]))
                combined_raw = "".join(b[1] for b in sorted_boxes)
                curr_conf = float(np.mean([float(b[2]) for b in sorted_boxes]))
                fmt, bonus, _ = parse_and_score_plate(combined_raw)
                if fmt and bonus > 0.0:
                    r_fmt = fmt
                    r_conf = curr_conf
                    break
                for b in sorted_boxes:
                    sub_fmt, sub_bonus, _ = parse_and_score_plate(b[1])
                    if sub_fmt and sub_bonus > 0.0:
                        r_fmt = sub_fmt
                        r_conf = float(b[2])
                        break

        # Engine 3: Fine-Tuned CRNN-BiLSTM (Indian HSRP DIN 1451)
        c_fmt = ""
        c_conf = 0.0
        crnn = get_crnn_model()
        if crnn is not None:
            c_raw, c_conf = predict_lprnet(crnn, tight_crop)
            c_fmt, _, _ = parse_and_score_plate(c_raw)

        # Multi-Engine Ensemble with PARSeq Vision Transformer Authority
        votes = {}
        # PARSeq receives 3 votes because ViT autoregressive attention is superior on DIN 1451 digits
        if p_fmt:
            votes[p_fmt] = {"count": 3, "max_conf": p_conf}

        if r_fmt:
            if r_fmt not in votes:
                votes[r_fmt] = {"count": 0, "max_conf": 0.0}
            votes[r_fmt]["count"] += 1
            votes[r_fmt]["max_conf"] = max(votes[r_fmt]["max_conf"], r_conf)

        if c_fmt:
            if c_fmt not in votes:
                votes[c_fmt] = {"count": 0, "max_conf": 0.0}
            votes[c_fmt]["count"] += 1
            votes[c_fmt]["max_conf"] = max(votes[c_fmt]["max_conf"], c_conf)

        if votes:
            # Pick the candidate with most votes, then highest confidence
            best_plate = max(votes.keys(), key=lambda k: (votes[k]["count"], votes[k]["max_conf"]))
            vote_count = votes[best_plate]["count"]
            best_conf = votes[best_plate]["max_conf"]

            if vote_count >= 3:
                return best_plate, best_conf, best_conf + 0.65, tight_crop, sharpness
            elif vote_count >= 2:
                return best_plate, best_conf, best_conf + 0.50, tight_crop, sharpness
            else:
                # Single engine — use the highest confidence result
                if len(best_plate) == 10:
                    return best_plate, best_conf, best_conf + 0.45, tight_crop, sharpness
                return best_plate, best_conf, best_conf + 0.40, tight_crop, sharpness

        return "", 0.0, 0.0, tight_crop, sharpness

    aspect = bw / float(max(1, bh))
    candidates = []

    # Case A: Multi-Line / 2-Line Plate (Aspect Ratio < 2.8)
    if aspect < 2.8:
        # Strategy 1: EasyOCR multi-line box detection and line grouping
        variants = enhance_plate_variants(padded_crop if padded_crop.size > 0 else tight_crop)
        for variant in variants:
            dets = ocr_reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                batch_size=1,
            )
            assembled = assemble_ocr_detections(dets)
            for raw_text, conf in assembled:
                fmt_text, bonus, pat = parse_and_score_plate(raw_text)
                if fmt_text and bonus > 0.0 and conf >= 0.15:
                    candidates.append({
                        "text": fmt_text,
                        "conf": conf,
                        "score": conf + bonus + 0.10,
                        "source": "EasyOCR_2Line",
                    })

        # Strategy 2: LPRNet top/bottom horizontal split
        if lpr_model is not None:
            th = int(tight_crop.shape[0] * 0.52)
            by = int(tight_crop.shape[0] * 0.45)
            top_half = tight_crop[:th, :]
            bot_half = tight_crop[by:, :]
            raw_top, conf_top = predict_lprnet(lpr_model, top_half)
            raw_bot, conf_bot = predict_lprnet(lpr_model, bot_half)
            combo_split = raw_top + raw_bot
            fmt_split, bonus_split, pat_split = parse_and_score_plate(combo_split)
            if fmt_split and bonus_split > 0.0:
                avg_split_conf = (conf_top + conf_bot) / 2.0
                candidates.append({
                    "text": fmt_split,
                    "conf": avg_split_conf,
                    "score": avg_split_conf + bonus_split,
                    "source": "LPRNet_Split",
                })

    # Case B: Standard Single-Line Plate (Aspect Ratio >= 2.8)
    else:
        # Strategy 1: LPRNet direct inference
        if lpr_model is not None:
            lpr_raw, lpr_conf = predict_lprnet(lpr_model, tight_crop)
            lpr_fmt, lpr_bonus, pat = parse_and_score_plate(lpr_raw)
            if lpr_fmt and lpr_bonus > 0.0 and lpr_conf >= 0.30:
                # Suppress isolated Bharat series hallucinations on noisy crops
                if lpr_fmt.startswith("BH") or ("BH" in lpr_fmt and len(lpr_fmt) == 10 and lpr_fmt[:2].isdigit()):
                    if lpr_conf < 0.92:
                        lpr_fmt = ""
                if lpr_fmt:
                    candidates.append({
                        "text": lpr_fmt,
                        "conf": lpr_conf,
                        "score": lpr_conf + lpr_bonus,
                        "source": "LPRNet",
                    })

        # Strategy 2: EasyOCR fast 2-pass
        variants = enhance_plate_variants(padded_crop if padded_crop.size > 0 else tight_crop)
        for variant in variants:
            dets = ocr_reader.readtext(
                variant,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                contrast_ths=0.05,
                adjust_contrast=0.7,
                batch_size=1,
            )
            assembled = assemble_ocr_detections(dets)
            for raw_text, conf in assembled:
                fmt_text, bonus, pat = parse_and_score_plate(raw_text)
                if fmt_text and bonus > 0.0 and conf >= 0.15:
                    candidates.append({
                        "text": fmt_text,
                        "conf": conf,
                        "score": conf + bonus + 0.10,
                        "source": "EasyOCR",
                    })

    if not candidates:
        return "", 0.0, 0.0, tight_crop, sharpness

    # Return top scoring authentic candidate directly (never splice characters across different strings)
    best_candidate = max(candidates, key=lambda c: c["score"])
    return best_candidate["text"], best_candidate["conf"], best_candidate["score"], tight_crop, sharpness

