import pyttsx3

def Speak(Text):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty("voices")
    engine.setProperty("voices", voices[0].id)
    engine.setProperty("rate", 180)

    print(f"Jarvis: {Text}")
    engine.say(Text)
    engine.runAndWait()
