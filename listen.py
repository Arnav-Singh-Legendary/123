import speech_recognition

def SR():
    recognition = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as source:
        print(f"Jarvis: Listening...")
        recognition.pause_threshold = 1
        Audio = recognition.listen(source, 0, 0)

    try:
        print(f"Jarvis: Recognizing...")
        query = recognition.recognize_google(Audio, language="en-in")
        print(f"Your Command: {query}\n")

    except:
        return ""

    return query

SR()