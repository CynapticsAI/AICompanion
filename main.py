from fastapi import FastAPI
import blenderbot
import gradio as gr
app = FastAPI()
app = gr.mount_gradio_app(app, blenderbot.demo, path='/')