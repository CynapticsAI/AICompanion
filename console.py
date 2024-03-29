import os
import torch
import transformers
from gtts import gTTS
import speech_recognition as sr
from playsound import playsound
from transformers import pipeline,Conversation
from transformers import GPT2LMHeadModel, GPT2Tokenizer

flatten = lambda l: [item for sublist in l for item in sublist]

# Set Logging Level to Error
transformers.logging.set_verbosity_error()

# obtain audio from the microphone
r = sr.Recognizer()

class AI_Companion:

    def __init__(self, asr = "openai/whisper-tiny", chatbot = "af1tang/personaGPT",**kwargs):
        """
        Create an Instance of the Companion.
        Parameters:
        asr: Huggingface ASR Model Card. Default: openai/whisper-tiny
        chatbot: Huggingface Conversational Model Card. Default: af1tang/personaGPT
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize Speech Recognition Pipeline
        self.asr = pipeline("automatic-speech-recognition",model = asr, device = -1 if self.device == "cpu" else 0)

        # Load Language Model and Tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(chatbot).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(chatbot,padding_side='left')
        # Variables for PersonaGPT
        self.personas=[]
        self.dialog_hx=[]
        self.sett={
            "do_sample":True,
            "top_k":10,
            "top_p":0.92,
            "max_length":1000,
        }
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
        return history, None
    
    def add_fact(self, fact):
        """
        Add Fact to Persona.

        Parameters:
        fact
        """
        self.personas.append(fact + self.tokenizer.eos_token)
    
    def respond(self, history):
        """
        Generates Response to User Input.

        Parameters:
        history: Chat History
        
        Returns:
        history: history with response appended
        """
        # Add Personas
        personas = self.tokenizer.encode(''.join(['<|p2|>'] + self.personas + ['<|sep|>'] + ['<|start|>']))

        # Add User Input
        self.chat.add_user_input(history[-1][0])
        user_inp= self.tokenizer.encode(history[-1][0] + self.tokenizer.eos_token)
        self.dialog_hx.append(user_inp)
        bot_input_ids = self.to_var([personas + flatten(self.dialog_hx)]).long()

        # Generate Response
        full_msg =self.model.generate(bot_input_ids,do_sample = True,
                                      top_k = 10,
                                      top_p = 0.92,
                                      max_new_tokens = 2000,
                                      pad_token_id = self.tokenizer.eos_token_id)
        response = self.to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        self.dialog_hx.append(response)

        #Add Response to History
        history[-1][1] = self.tokenizer.decode(response, skip_special_tokens=True)

        # Speak Response
        bot.speak(history[-1][1])
        return history
    
    def speak(self, text):
        """
        Speaks the given text using gTTS,
        Parameters:
        text: text to be spoken
        """
        tts = gTTS(text, lang='en')
        tts.save('out.mp3')
        playsound("out.mp3")
        os.remove("out.mp3")


    
    def to_data(self, x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def to_var(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
        return x

if __name__ == "__main__":
    bot = AI_Companion(device = 0)
    history = []
    bot.speak("Hi, I am your AI Companion. Do you want to add any specific traits?")
    persona = input("Y/n")
    while persona.lower() == 'y':
        bot.add_fact(input("Enter Fact:"))
        persona = input("Add More? (Y/n)")

    bot.speak("Configured. What you want to talk about?")
    
    for i in range(5):

        # Save Audio from mic
        with sr.Microphone() as source:
            audio = r.listen(source)
        with open("audio_file.wav", "wb") as file:
            file.write(audio.get_wav_data())
        
        # Bot Listens and Understands Audio(ASR)
        history , _ = bot.listen("audio_file.wav",history)

        # Print your Conversation
        print("You:", history[-1][0])
        history = bot.respond(history)
        print("Bot:", history[-1][1])
