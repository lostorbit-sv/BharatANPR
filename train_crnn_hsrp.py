import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "synthetic_hsrp"
MODEL_SAVE_PATH = BASE_DIR / "best_crnn_bilstm.pth"

import ocr

CRNN_CHARS = ocr.CRNN_CHARS  # "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" (length 36)
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CRNN_CHARS)}  # 1-based, 0 is blank


def decode_ctc(logits):
    """Greedy CTC decoder."""
    probs = logits.softmax(dim=-1)
    max_indices = probs.argmax(dim=-1).permute(1, 0).cpu().numpy()  # (B, T)

    pred_strings = []
    for b in range(max_indices.shape[0]):
        chars = []
        prev = 0
        for t in range(max_indices.shape[1]):
            idx = max_indices[b, t]
            if idx != 0 and idx != prev:
                chars.append(CRNN_CHARS[idx - 1])
            prev = idx
        pred_strings.append("".join(chars))
    return pred_strings


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("VECTORIZED GPU TRAINING: CRNN-BiLSTM FOR INDIAN HSRP & COMMERCIAL PLATES", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Save Path: {MODEL_SAVE_PATH}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Datasets directly into contiguous tensors
    train_path = DATASET_DIR / "hsrp_train_60k.pt"
    if not train_path.exists():
        train_path = DATASET_DIR / "hsrp_train_48k.pt"

    print(f"Loading {train_path.name} into GPU/RAM...", flush=True)
    train_data = torch.load(str(train_path), weights_only=False)
    train_images = train_data["images"]  # uint8 (60000, 1, 32, 160)
    train_labels = train_data["labels"]
    num_train = len(train_labels)

    # Pre-encode labels into target tensors
    train_targets = [torch.tensor([CHAR_TO_IDX[c] for c in s if c in CHAR_TO_IDX], dtype=torch.long) for s in train_labels]
    train_target_lens = torch.tensor([len(t) for t in train_targets], dtype=torch.long)

    print("Loading hsrp_val_2k.pt into RAM...", flush=True)
    val_data = torch.load(str(DATASET_DIR / "hsrp_val_2k.pt"), weights_only=False)
    val_images = val_data["images"]
    val_labels = val_data["labels"]
    val_targets = [torch.tensor([CHAR_TO_IDX[c] for c in s if c in CHAR_TO_IDX], dtype=torch.long) for s in val_labels]
    val_target_lens = torch.tensor([len(t) for t in val_targets], dtype=torch.long)
    num_val = len(val_labels)

    # 2. Build Model & Load Checkpoint for Fine-Tuning
    num_classes = len(CRNN_CHARS) + 1  # 36 chars + 1 blank
    model = ocr.CRNN_BiLSTM(imgH=32, nc=1, nclass=num_classes, nh=256).to(device)

    if MODEL_SAVE_PATH.exists():
        try:
            ckpt = torch.load(str(MODEL_SAVE_PATH), map_location=device, weights_only=True)
            model.load_state_dict(ckpt)
            print("Successfully loaded pre-trained checkpoint for fine-tuning!", flush=True)
        except Exception as e:
            print(f"Starting from scratch (checkpoint load failed: {e})", flush=True)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    batch_size = 128

    best_exact_acc = 0.0

    print(f"\nStarting training for {epochs} epochs (Batch size: {batch_size}, {num_train // batch_size} batches/epoch)...\n", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        # Vectorized batch shuffle
        perm = torch.randperm(num_train)
        num_batches = 0

        for i in range(0, num_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            B = len(batch_idx)

            # Vectorized GPU transfer & normalization in one shot
            images = (train_images[batch_idx].float().to(device) / 127.5) - 1.0
            targets = torch.cat([train_targets[idx] for idx in batch_idx.tolist()]).to(device)
            target_lengths = train_target_lens[batch_idx].to(device)

            optimizer.zero_grad()
            logits = model(images)  # (T=41, B, C)
            T = logits.size(0)
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long, device=device)

            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        train_loss /= num_batches
        epoch_time = time.time() - t0

        # Fast Vectorized Evaluation
        model.eval()
        correct_plates = 0
        total_plates = 0
        correct_chars = 0
        total_chars = 0

        with torch.no_grad():
            for i in range(0, num_val, batch_size):
                batch_images = (val_images[i : i + batch_size].float().to(device) / 127.5) - 1.0
                logits = model(batch_images)
                preds = decode_ctc(logits)
                targets = val_labels[i : i + batch_size]

                for pred, target in zip(preds, targets):
                    total_plates += 1
                    if pred == target:
                        correct_plates += 1
                    total_chars += len(target)
                    for pc, tc in zip(pred, target):
                        if pc == tc:
                            correct_chars += 1

        exact_acc = (correct_plates / total_plates) * 100.0
        char_acc = (correct_chars / total_chars) * 100.0

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss: {train_loss:.4f} | "
            f"Char Acc: {char_acc:.2f}% | "
            f"Exact Plate Acc: {exact_acc:.2f}% | "
            f"Time: {epoch_time:.1f}s",
            flush=True
        )

        if exact_acc > best_exact_acc:
            best_exact_acc = exact_acc
            torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
            print(f"   --> New Best Model Saved! Exact Accuracy: {best_exact_acc:.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE! Best Exact Plate Accuracy: {best_exact_acc:.2f}%", flush=True)
    print(f"Model saved to: {MODEL_SAVE_PATH}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    train()
