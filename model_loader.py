"""Pemuatan model deteksi balok (YOLO). Tidak ada yang berjalan saat impor: panggil load_models().

Jangan beri nama file ini models.py: bentrok dengan paket models milik MODEL/yolov5."""
import os
import subprocess
import sys

import torch

from config import BASE_DIR


def load_models():
    """Muat model YOLO. Mengembalikan (model_yolo, device, use_bantal_model).

    use_bantal_model bernilai False bila MODEL_BANTAL tidak aktif atau pemuatan
    bantal.pt gagal sehingga jatuh ke best.pt.
    """
    # Device configuration
    print("GPU CUDA is Available : " + str(torch.cuda.is_available()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_bantal_model = os.getenv('MODEL_BANTAL', 'false').lower() == 'true'
    model_yolo = None
    if use_bantal_model:
        path_model = os.path.join(BASE_DIR, 'MODEL', 'bantal', 'bantal.pt')

        try:
            print(">>> Loading bantal.pt with Ultralytics format...")

            # Cek apakah ini model Ultralytics (v8+ atau v7.0+)
            checkpoint = torch.load(path_model, map_location='cpu', weights_only=False)

            if 'version' in checkpoint:
                print(f">>> Model version: {checkpoint['version']}")

            # Jika model dari Ultralytics (punya key 'version')
            if 'version' in checkpoint and hasattr(checkpoint['model'], '__class__') and                'ultralytics' in str(checkpoint['model'].__class__):

                print(">>> Detected Ultralytics format, loading with ultralytics library...")

                # Coba load dengan ultralytics library
                try:
                    from ultralytics import YOLO
                    model_yolo = YOLO(path_model)
                    model_yolo.to(device)
                    print(">>> Model bantal.pt loaded successfully with Ultralytics YOLO")
                except ImportError:
                    print(">>> Ultralytics library not found, installing...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
                    from ultralytics import YOLO
                    model_yolo = YOLO(path_model)
                    model_yolo.to(device)
                    print(">>> Model bantal.pt loaded successfully with Ultralytics YOLO")

            else:
                # Fallback: gunakan attempt_load biasa
                print(">>> Loading with attempt_load...")
                sys.path.insert(0, os.path.join(BASE_DIR, 'MODEL', 'yolov5'))
                from models.experimental import attempt_load

                model_yolo = attempt_load(path_model, device=device)

                # Inisialisasi grid untuk model lama
                for m in model_yolo.modules():
                    if type(m).__name__ == 'Detect':
                        if not hasattr(m, 'grid'):
                            m.grid = [torch.zeros(1)] * m.nl
                        if not hasattr(m, 'anchor_grid'):
                            m.anchor_grid = [torch.zeros(1)] * m.nl

                model_yolo.to(device)
                print(">>> Model bantal.pt loaded successfully with attempt_load")

        except Exception as e:
            print(f">>> Failed to load bantal.pt: {e}")
            print(">>> Falling back to best.pt...")
            use_bantal_model = False

    if not use_bantal_model:
        path_model = os.path.join(BASE_DIR, 'MODEL', 'exp7', 'weights', 'best.pt')
        model_yolo = torch.hub.load(
            os.path.join(BASE_DIR, 'MODEL', 'yolov5'),
            'custom',
            path=path_model,
            force_reload=True,
            source='local'
        )
        model_yolo.to(device)

    return model_yolo, device, use_bantal_model
