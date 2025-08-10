from gtts import gTTS
from playsound import playsound

def SpeakTextG(command):

    text = command
    tts = gTTS(text=text, lang="ja")
    filename = "output.mp3"
    tts.save(filename)
    playsound(filename)

SpeakTextG("こんにちは、お元気ですか？")