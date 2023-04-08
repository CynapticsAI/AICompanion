import speech_recognition as sr
from gtts import gTTS
import os
import pyglet
from transformers import pipeline,Conversation
import transformers
import time
transformers.logging.set_verbosity_error()
def speak(text):
    """
    Speaks the text via gTTS
    
    Parameters:
    text: Text to Speak
    """

    # Replace '' with "" and '' with "" and have fun
    command = f'gtts-cli "{text}" --output audio.mp3'
    os.system(command)
    music = pyglet.media.load("audio.mp3", streaming=False)
    music.play()
    os.remove("audio.mp3")

# obtain audio from the microphone
r = sr.Recognizer()

class AI_Companion:
    def __init__(self, asr = "openai/whisper-tiny", chatbot = "microsoft/DialoGPT-small", device = 0):
        """
        Create an Instance of the Companion.

        Parameters:
        asr: Huggingface ASR Model Card. Default: openai/whisper-tiny
        chatbot: Huggingface Conversational Model Card. Default: microsoft/DialoGPT-small
        device: Device to Run the model on. Default: 0 (GPU). Set to 1 to run on CPU.
        """
        self.asr = pipeline("automatic-speech-recognition",model = asr,device=device)
        self.chatbot = pipeline("conversational", model = chatbot,device=device)
        self.chat = Conversation()

    def listen(self, audio, history):
        """
        Convert Speech to Text.

        Parameters:
        audio: Audio Filepath
        history: Chat History

        Returns:
        history : history with recognized text appended
        Audio : empty gradio component to clear gradio voice input
        """
        text = self.asr(audio)["text"]
        history = history + [[text,None]]
        return history , None
    
    def respond(self, history):
        """
        Generates Response to User Input.

        Parameters:
        history: Chat History
        
        Returns:
        history: history with response appended
        """
        self.chat.add_user_input(history[-1][0])
        response = self.chatbot(self.chat)
        history[-1][1] = response.generated_responses[-1]
        return history

bot = AI_Companion()
history = []
speak("Say something!")
for i in range(5):
    time.sleep(1)
    print("Speak")
    with sr.Microphone() as source:
        audio = r.listen(source)
    with open("audio_file.wav", "wb") as file:
        file.write(audio.get_wav_data())
    history , _ = bot.listen("audio_file.wav",history)
    print("You:", history[-1][0])
    history = bot.respond(history)
    speak(history[-1][1])
    print("Bot:", history[-1][1])
    time.sleep(0.25 + len(history[-1][1].split())*5/16)
