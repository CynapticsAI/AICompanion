import torch
import os
import gradio as gr
from gtts import gTTS
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
css = """
#input {background-color: #FFCCCB} 
"""
# Utility Functions
flatten = lambda l: [item for sublist in l for item in sublist]

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def clear():
    return None,[]

def append(text, history):
    history.append([text,None])
    history , audio = bot.respond(history)
    return history, audio, None

class AI_Companion:
    """
    Class that Implements AI Companion.
    """

    def __init__(self, asr = "openai/whisper-tiny", chatbot = "facebook/blenderbot-3B"):
        """
        Create an Instance of the Companion.
        Parameters:
        asr: Huggingface ASR Model Card. Default: openai/whisper-tiny
        chatbot: Huggingface Conversational Model Card. Default: af1tang/personaGPT
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.asr = pipeline("automatic-speech-recognition",model = asr,device= -1 if self.device == "cpu" else 0)
        # self.model = GPT2LMHeadModel.from_pretrained(chatbot).to(self.device)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(chatbot)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(chatbot).to(self.device)
        self.tokenizer=AutoTokenizer.from_pretrained(chatbot)
        self.personas=[]
        self.dialog_hx=[]
        self.sett={
            "do_sample":True,
            "top_k":10,
            "top_p":0.92,
            "max_length":1000,
        }

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
        history.append([text,None])
        return history , None
    
    def add_fact(self,audio):
        '''
        Add fact to Persona.
        Takes in Audio, converts it into text and adds it to the facts list.

        Parameters:
        audio : audio of the spoken fact
        '''
        text=self.asr(audio)
        self.personas.append(text['text']+self.tokenizer.eos_token)
        return None
    
    def respond(self, history,**kwargs):
        """
        Generates Response to User Input.

        Parameters:
        history: Chat History
        
        Returns:
        history: history with response appended
        audio: audio of the spoken response
        """

        personas = self.tokenizer.encode(''.join(['<|p2|>'] + self.personas + ['<|sep|>'] + ['<|start|>']))
        print(history)
        user_inp= self.tokenizer.encode(history[-1][0])
        self.dialog_hx.append(user_inp)
        bot_input_ids = to_var([personas + flatten(self.dialog_hx)]).long()
        if (len(bot_input_ids[0])>128):
            bot_input_ids=torch.narrow(bot_input_ids,1,-128,128)
        with torch.no_grad():
            full_msg = self.model.generate(bot_input_ids,do_sample = True,
                                      top_k = 10,
                                      top_p = 0.92,
                                      max_new_tokens= 512)

        response = to_data(full_msg[0])
        self.dialog_hx.append(response)
        history[-1][1] = self.tokenizer.decode(response, skip_special_tokens=True)
        self.speak(history[-1][1])

        return history, "out.mp3"
    
    def talk(self, audio, history):
        history, _ = self.listen(audio, history)
        history, audio = self.respond(history)
        return history, None, audio

    def speak(self, text):
        """
        Speaks the given text using gTTS,
        Parameters:
        text: text to be spoken
        """
        tts = gTTS(text, lang='en')
        tts.save('out.mp3')

# Initialize AI Companion
bot = AI_Companion()

# Create the Interface
with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id = "chatbot").style(height = 300)
    audio = gr.Audio(source = "microphone", type = "filepath", label = "Input")
    msg = gr.Textbox()
    audio1 = gr.Audio(type = "filepath", label = "Output",elem_id="input")
    with gr.Row():
        b1 = gr.Button("Submit")
        b2 = gr.Button("Clear")
        b3=  gr.Button("Add Fact")
    b1.click(bot.talk, [audio, chatbot], [chatbot, audio, audio1])
    msg.submit(append, [msg, chatbot], [chatbot, audio1, msg])
    b2.click(clear, [] , [audio,chatbot])
    b3.click(bot.add_fact, [audio], [audio])
demo.launch(share=True)