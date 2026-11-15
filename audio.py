import os
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

# Suara dimatikan secara default: play_audio() memakai sd.wait() sehingga seluruh GUI dan kamera
# berhenti sampai file suara selesai diputar. Set ENABLE_SOUND=true di .env untuk menyalakan lagi.
ENABLE_SOUND = os.getenv('ENABLE_SOUND', 'false').lower() == 'true'


def play_audio(wav):
    """
    wav: either a filepath (str / Path) or a numpy array of samples.
    If filepath -> read file with soundfile so we get data and samplerate.
    """
    if not ENABLE_SOUND:
        return

    # jika path diberikan, baca file
    if isinstance(wav, (str, Path)):
        path = str(wav)
        # cek ada file
        if not Path(path).exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        data, samplerate = sf.read(path, dtype='float32')  # data float32 in [-1,1]
    else:
        # asumsi sudah array-like
        data = np.asarray(wav)
        samplerate = 22050  # default jika pemanggil memang mengirim data langsung

    # pastikan bentuk (frames, channels)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    sd.play(data, samplerate)
    sd.wait()
    print(">>> audio.ENABLE_SOUND =", ENABLE_SOUND)