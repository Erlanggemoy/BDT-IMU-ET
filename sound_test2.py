from gtts import gTTS
from io import BytesIO
import pygame

# Teks dalam bahasa Jepang
text = "こんにちは、元気ですか？"

# Konversi teks ke suara
tts = gTTS(text=text, lang="ja")
audio_fp = BytesIO()
tts.write_to_fp(audio_fp)

# Inisialisasi pygame untuk memutar audio
pygame.mixer.init()
audio_fp.seek(0)
pygame.mixer.music.load(audio_fp)
pygame.mixer.music.play()

# Tunggu hingga audio selesai
while pygame.mixer.music.get_busy():
    continue
