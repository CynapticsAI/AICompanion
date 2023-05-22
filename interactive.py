import torch
import gradio as gr
from gtts import gTTS
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
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
class AI_Companion:
    """
    Class that Implements AI Companion.
    """

    def __init__(self, asr = "openai/whisper-tiny", chatbot = "af1tang/personaGPT",**kwargs):
        """
        Create an Instance of the Companion.
        Parameters:
        asr: Huggingface ASR Model Card. Default: openai/whisper-tiny
        chatbot: Huggingface Conversational Model Card. Default: af1tang/personaGPT
        device: Device to Run the model on. Default: -1 (GPU). Set to 1 to run on CPU.
        """

        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.asr = pipeline("automatic-speech-recognition",model = asr,device=self.device)
        self.model = GPT2LMHeadModel.from_pretrained(chatbot).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(chatbot)
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
        history = history + [(text,None)]
        return history , None
    
    def add_fact(self,audio):
        '''
        Add fact to Persona.
        Takes in Audio, converts it into text and adds it to the facts list.

        Parameters:
        audio : audio of the spoken fact
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
        audio: audio of the spoken response
        """

        personas = self.tokenizer.encode(''.join(['<|p2|>'] + self.personas + ['<|sep|>'] + ['<|start|>']))
        user_inp= self.tokenizer.encode(history[-1][0]+self.tokenizer.eos_token)
        self.dialog_hx.append(user_inp)
        bot_input_ids = to_var([personas + flatten(self.dialog_hx)]).long()

        full_msg =self.model.generate(bot_input_ids, 
                                      do_sample = True,
                                      top_k = 10,
                                      top_p = 0.92,
                                      max_length = 1000,
                                      pad_token_id = self.tokenizer.eos_token_id)
        

        response = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        self.dialog_hx.append(response)
        history[-1][1] = self.tokenizer.decode(response, skip_special_tokens=True)
        print(history[-1][1])
        self.speak(history[-1][1])

        return history, "out.mp3"

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
    chatbot = gr.Chatbot([], elem_id = "chatbot").style(height = 450)
    audio = gr.Audio(source = "microphone", type = "filepath", label = "Input")
    audio1 = gr.Audio(type = "filepath", label = "Output",elem_id="input")
    with gr.Row():
        b1 = gr.Button("Submit")
        b2 = gr.Button("Clear")
        b3=  gr.Button("Add Fact")
    b1.click(bot.listen, [audio, chatbot], [chatbot, audio]).then(bot.respond, chatbot, [chatbot, audio1])
    b2.click(clear, [] , [audio,chatbot])
    b3.click(bot.add_fact, [audio], [audio])
demo.launch()