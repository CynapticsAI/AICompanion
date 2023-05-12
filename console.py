import speech_recognition as sr
import os
import pyglet
from transformers import pipeline,Conversation
import transformers
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pyttsx3
from Person_bot import *

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
    def __init__(self, asr = "openai/whisper-tiny", chatbot = "af1tang/personaGPT", device = -1,**kwargs):
        """
        Create an Instance of the Companion.
        Parameters:
        asr: Huggingface ASR Model Card. Default: openai/whisper-tiny
        chatbot: Huggingface Conversational Model Card. Default: microsoft/DialoGPT-small
        device: Device to Run the model on. Default: 0 (GPU). Set to 1 to run on CPU.
        """
        self.asr = pipeline("automatic-speech-recognition",model = asr,device=device)
        self.model = GPT2LMHeadModel.from_pretrained(chatbot).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(chatbot)
        # self.chatbot = pipeline("conversational", model=model, tokenizer=tokenizer, device=device)
        self.personas=[]
        self.dialog_hx=[]
        self.sett={
            "do_sample":True,
            "top_k":10,
            "top_p":0.92,
            "max_length":1000,
        }
        self.chat = Conversation()
        self.configureTTS()

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
    def add_fact(self,audio):
        text=self.asr(audio)
        print(text)
        self.personas.append(text['text']+self.tokenizer.eos_token)
        return None
    def respond(self, history,**kwargs):
        """
        Generates Response to User Input.

        Parameters:
        history: Chat History
        
        Returns:
        history: history with response appended
        """
        print(self.personas)
        personas = self.tokenizer.encode(''.join(['<|p2|>'] + self.personas + ['<|sep|>'] + ['<|start|>']))
        self.chat.add_user_input(history[-1][0])
        user_inp= self.tokenizer.encode(history[-1][0]+self.tokenizer.eos_token)
        self.dialog_hx.append(user_inp)
        bot_input_ids = to_var([personas + flatten(self.dialog_hx)]).long()
        full_msg =self.model.generate(bot_input_ids,do_sample=True,top_k=10,top_p=0.92,max_length=1000,pad_token_id=self.tokenizer.eos_token_id)
        response = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        self.dialog_hx.append(response)
        history[-1][1] = self.tokenizer.decode(response, skip_special_tokens=True)
        bot.speak(history[-1][1])
        return history
    def speak(self, text):
        """
        Speaks.

        Parameters:
        text: text to be spoken
        """
        self.engine.say(text)
        self.engine.runAndWait()
        print("Speak")
    
    def configureTTS(self):
        self.engine = pyttsx3.init()

        """ RATE"""
        self.engine.setProperty('rate', 135)     # setting up new voice rate

        """VOLUME"""
        self.engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

        """VOICE"""
        voices = self.engine.getProperty('voices')       #getting details of current voice
        self.engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 0 for male, 1 for female  

bot = AI_Companion(device = 0)
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
    print("Bot:", history[-1][1])
    time.sleep(0.25 + len(history[-1][1].split())*5/16)
