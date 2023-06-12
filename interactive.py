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
    return None,[],[]

def append(text, history,dialog_hx,personas):
    history.append([text,None])
    history , audio,dialog_hx= bot.respond(history,dialog_hx,personas)
    return history, audio, None,dialog_hx

class AI_Companion:
    """
    Class that Implements AI Companion.
    """

    def __init__(self, asr = "openai/whisper-tiny", chatbot = "af1tang/personaGPT"):
        """
        Create an Instance of the Companion.
        Parameters:
        asr: Huggingface ASR Model Card. Default: openai/whisper-tiny
        chatbot: Huggingface Conversational Model Card. Default: af1tang/personaGPT
        """

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.asr = pipeline("automatic-speech-recognition",model = asr,device= -1 if self.device == "cpu" else 0)
        self.model = GPT2LMHeadModel.from_pretrained(chatbot).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(chatbot)
        self.personas=[]
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
    
    def add_fact(self,audio,personas,msg):
        '''
        Add fact to Persona.
        Takes in Audio, converts it into text and adds it to the facts list.

        Parameters:
        audio : audio of the spoken fact
        '''
        if audio is not None:
            text=self.asr(audio)
            personas.append(text['text']+self.tokenizer.eos_token)
        else:
            personas.append(msg+self.tokenizer.eos_token)
        return None,personas,None
    
    def respond(self, history,dialog_hx,personas,**kwargs):
        """
        Generates Response to User Input.

        Parameters:
        history: Chat History
        
        Returns:
        history: history with response appended
        audio: audio of the spoken response
        """

        person = self.tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))
        user_inp= self.tokenizer.encode(history[-1][0]+self.tokenizer.eos_token)
        dialog_hx.append(user_inp)
        bot_input_ids = to_var([person + flatten(dialog_hx)]).long()
        with torch.no_grad():

            full_msg = self.model.generate(bot_input_ids,
                                        repetition_penalty=1.4,
                                        top_k = 10,
                                        top_p = 0.92,
                                        max_new_tokens = 256,
                                        num_beams=2,
                                        pad_token_id = self.tokenizer.eos_token_id)
        

        response = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
        dialog_hx.append(response)
        history[-1][1] = self.tokenizer.decode(response, skip_special_tokens=True)
        self.speak(history[-1][1])
        return history, "out.mp3",dialog_hx
    
    def talk(self, audio, history,dialog_hx,personas,text):
        if audio is not None:
            history, _ = self.listen(audio, history)
        else:
            history.append([text,None])
        history, audio,dialog_hx = self.respond(history,dialog_hx,personas)
        return history, None, audio,dialog_hx,None

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
personas=[]
for i in ['I\'m a 19 year old girl','I study at IIT Indore','I am an easy-going and fun loving person','I love to swim','I am friendly, nice ,fun and kind','I am studious and get good grades']:
    response = i+ bot.tokenizer.eos_token
    personas.append(response)


# Create the Interface
with gr.Blocks() as demo:
    dialog_hx=gr.State([])
    personas=gr.State(personas)
    chatbot = gr.Chatbot([], elem_id = "chatbot").style(height = 300)
    audio = gr.Audio(source = "microphone", type = "filepath", label = "Input")
    msg = gr.Textbox()
    audio1 = gr.Audio(type = "filepath", label = "Output",elem_id="input")
    with gr.Row():
        b1 = gr.Button("Submit")
        b2 = gr.Button("Clear")
        b3=  gr.Button("Add Fact")
    b1.click(bot.talk, [audio, chatbot,dialog_hx,personas,msg], [chatbot, audio, audio1,dialog_hx,msg])
    msg.submit(append, [msg, chatbot,dialog_hx,personas], [chatbot, audio1, msg,dialog_hx])
    b2.click(clear, [] , [audio,chatbot,dialog_hx])
    b3.click(bot.add_fact, [audio,personas,msg], [audio,personas,msg])
demo.launch(share=True)