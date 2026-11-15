"""Pengaturan kamera dan deteksi: default, muat/simpan JSON, dan penerapan ke frame.

`camera_settings` adalah dict bersama. Ubah isinya di tempat (update/item assignment),
jangan ditugaskan ulang, agar semua modul yang mengimpornya melihat nilai yang sama.
"""
import json
import os

import cv2

from config import BASE_DIR


CAMERA_SETTINGS_PATH = os.path.join(BASE_DIR, 'camera_settings.json')
CAMERA_DEFAULTS = {"brightness": 0, "zoom": 1.0, "rotation": 0, "flip_h": False, "flip_v": False}


def _default_stable_frames():
    try:
        return max(2, min(15, int(os.getenv('ROTATION_STABLE_FRAMES', '6'))))
    except (ValueError, TypeError):
        return 6


# Setelan deteksi: conf = ambang confidence YOLO, stable_frames = frame stabil agar perubahan sisi dihitung,
# lift_ratio = perubahan luas kotak balok yang dianggap "terangkat"
DETECT_DEFAULTS = {"conf": 0.7, "stable_frames": _default_stable_frames(), "lift_ratio": 0.12}
camera_settings = {**CAMERA_DEFAULTS, **DETECT_DEFAULTS}


def _sanitize_camera_settings(raw):
    """Batasi nilai ke rentang valid; nilai rusak diganti default."""
    out = {**CAMERA_DEFAULTS, **DETECT_DEFAULTS}
    try:
        out["brightness"] = int(max(-100, min(100, int(raw.get("brightness", 0)))))
        out["zoom"] = float(max(1.0, min(3.0, float(raw.get("zoom", 1.0)))))
        rot = int(raw.get("rotation", 0)) % 360
        out["rotation"] = rot if rot in (0, 90, 180, 270) else 0
        out["flip_h"] = bool(raw.get("flip_h", False))
        out["flip_v"] = bool(raw.get("flip_v", False))
        out["conf"] = round(float(max(0.3, min(0.95, float(raw.get("conf", DETECT_DEFAULTS["conf"]))))), 2)
        out["stable_frames"] = int(max(2, min(15, int(raw.get("stable_frames", DETECT_DEFAULTS["stable_frames"])))))
        out["lift_ratio"] = round(float(max(0.05, min(0.40, float(raw.get("lift_ratio", DETECT_DEFAULTS["lift_ratio"]))))), 2)
    except (ValueError, TypeError, AttributeError):
        return {**CAMERA_DEFAULTS, **DETECT_DEFAULTS}
    return out


def load_camera_settings():
    try:
        with open(CAMERA_SETTINGS_PATH, "r", encoding="utf-8") as f:
            camera_settings.update(_sanitize_camera_settings(json.load(f)))
        print(f">>> Camera settings loaded: {camera_settings}")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f">>> Could not load camera settings, using defaults: {e}")


def save_camera_settings():
    try:
        with open(CAMERA_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(camera_settings, f, indent=2)
    except Exception as e:
        print(f">>> Could not save camera settings: {e}")


def apply_camera_settings(frame):
    """Terapkan orientasi -> zoom (digital, tengah frame) -> brightness pada frame BGR dari kamera.
    Dipanggil sebelum seluruh pemrosesan sehingga deteksi memakai gambar yang sama dengan preview."""
    s = camera_settings

    rot = s["rotation"]
    if rot == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rot == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if s["flip_h"] and s["flip_v"]:
        frame = cv2.flip(frame, -1)
    elif s["flip_h"]:
        frame = cv2.flip(frame, 1)
    elif s["flip_v"]:
        frame = cv2.flip(frame, 0)

    zoom = s["zoom"]
    if zoom > 1.001:
        h, w = frame.shape[:2]
        cw, ch = max(1, int(w / zoom)), max(1, int(h / zoom))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        frame = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    if s["brightness"]:
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=s["brightness"])

    return frame


load_camera_settings()
