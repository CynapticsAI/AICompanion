import gradio as gr
from transformers import pipeline,Conversation
#ffmpeg

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
        history = history + [(text,None)]
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

def clear():
    return None

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=600)
    audio = gr.Audio(source="microphone", type="filepath")
    with gr.Row():
        b1 = gr.Button("Submit")
        b2 = gr.Button("Clear")

    b1.click(bot.listen, [audio, chatbot], [chatbot, audio]).then(bot.respond, chatbot, chatbot)
    b2.click(clear, [] , audio)

demo.launch()