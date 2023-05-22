from fastapi import FastAPI
import interactive
import gradio as gr
app = FastAPI()
app = gr.mount_gradio_app(app, interactive.demo, path='/')