import gradio as gr
import pyttsx3
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from Person_bot import *



class AI_Companion:
    def __init__(self, asr = "openai/whisper-tiny", chatbot = "af1tang/personaGPT", device = -1,**kwargs):
        """
        Create an Instance of the Companion.
        Parameters:
        asr: Huggingface ASR Model Card. Default: openai/whisper-tiny
        chatbot: Huggingface Conversational Model Card. Default: af1tang/personaGPT
        device: Device to Run the model on. Default: -1 (GPU). Set to 1 to run on CPU.
        """
        self.asr = pipeline("automatic-speech-recognition",model = asr,device=device)
        self.model = GPT2LMHeadModel.from_pretrained(chatbot)
        self.tokenizer = GPT2Tokenizer.from_pretrained(chatbot)
        self.personas=[]
        self.dialog_hx=[]
        self.sett={
            "do_sample":True,
            "top_k":10,
            "top_p":0.92,
            "max_length":1000,
        }
        # self.chat = Conversation()
        self.configureTTS()

    def configureTTS(self):
        self.engine = pyttsx3.init()

        """ RATE """
        self.engine.setProperty('rate', 135)     # setting up new voice rate

        """ VOLUME """
        self.engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

        """ VOICE """
        voices = self.engine.getProperty('voices')       #getting details of current voice
        self.engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 0 for male, 1 for female 

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
        history = history + [(text,None)]
        return history , None
    def add_fact(self,audio):
        '''
        Add fact to Persona.

        Parameters:
        fact
        '''
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
        personas = self.tokenizer.encode(''.join(['<|p2|>'] + self.personas + ['<|sep|>'] + ['<|start|>']))
        # self.chat.add_user_input(history[-1][0])
        user_inp= self.tokenizer.encode(history[-1][0]+self.tokenizer.eos_token)
        self.dialog_hx.append(user_inp)
        bot_input_ids = to_var([personas + flatten(self.dialog_hx)]).long()
        full_msg =self.model.generate(bot_input_ids,do_sample=True,top_k=10,top_p=0.92,max_length=1000,pad_token_id=self.tokenizer.eos_token_id)
        response = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        self.dialog_hx.append(response)
        history[-1][1] = self.tokenizer.decode(response, skip_special_tokens=True)
        self.engine.save_to_file(history[-1][1] , 'voice.mp3')
        self.engine.runAndWait()
        return history, 'voice.mp3'
bot = AI_Companion()

def clear():
    return None,[]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=600)
    audio = gr.Audio(source="microphone", type="filepath")
    with gr.Row():
        b1 = gr.Button("Submit")
        b2 = gr.Button("Clear")
        b3=  gr.Button("Add Fact")
    b1.click(bot.listen, [audio, chatbot], [chatbot, audio]).then(bot.respond, chatbot, [chatbot,audio])
    b2.click(clear, [] , [audio,chatbot])
demo.launch()