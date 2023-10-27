import gradio as gr
import sqlite3
from decouple import config
import pandas as pd
import webbrowser
import os
import asyncio
import subprocess
from Inference_UI import StableDiffusionUIGenerator
import threading
import webbrowser
import requests
from pyngrok import ngrok
import re
from selenium import webdriver
import execjs
import psutil
import subprocess
import multiprocessing
from psutil import process_iter
from signal import SIGTERM, SIGKILL
import string
import random


def extract_url_from_string(input_string):
    """
    Extract a URL from a string using regex.
    """
    url_pattern = r'"(https?://[^"]+)"'
    matches = re.search(url_pattern, input_string)
    if matches:
        extracted_url = matches.group(1)
        return extracted_url
    else:
        return None


# Define the JavaScript function as a string
javascript_function = """function clickButton() {
var button = document.getElementById("launcher");
button.click();
}"""

address = "0.0.0.0"
port = config("INFERENCE_PORT")
public_url = ngrok.connect(addr=f'{address}:{port}')
public_url = extract_url_from_string(str(public_url))
print("Public URL for inference:", public_url)


def stop_process_at_port(port, signal=SIGTERM):
    """
    Stop a process using a given port and signal.
    """
    try:
        for proc in process_iter():
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    proc.send_signal(signal)
                    print(f"Process {proc.name()} (PID: {proc.pid}) using port {port} terminated.")
                    break
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def get_or_create_eventloop():
    """
    Get or create an asyncio event loop for the current thread.
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


async def on_button_click(ui_generator):
    """
    Launch the Gradio interface.
    """
    stop_process_at_port(7861)
    interface = ui_generator.create_ui()
    await interface.launch(server_name="127.0.0.1", server_port=int(config("INFERENCE_PORT")))


async def enable_inference(model):
    """
    Enable inference and return the public URL.
    """
    loop = get_or_create_eventloop()
    ui_generator = StableDiffusionUIGenerator(model=model)

    def run_on_button_click():
        asyncio.run(on_button_click(ui_generator))

    gradio_thread = multiprocessing.Process(target=run_on_button_click)
    gradio_thread.start()
    return public_url


# Define a function to fetch models from the database with pagination
def fetch_models(page=1, per_page=10, filter_status=None):
    """
    Fetch models from the database with pagination and optional filtering.
    """
    conn = sqlite3.connect('models.db')  # Replace with your actual database path
    cursor = conn.cursor()

    if filter_status == "Incomplete":
        cursor.execute(
            "SELECT * FROM models WHERE approval_status=? LIMIT ? OFFSET ?",
            ("Incomplete", per_page, (page - 1) * per_page),
        )
    elif filter_status == "Discarded":
        cursor.execute(
            "SELECT * FROM models WHERE approval_status IN (?, ?, ?) LIMIT ? OFFSET ?",
            ("Overtrained", "Undertrained", "Unusable", per_page, (page - 1) * per_page),
        )
    elif filter_status == "Submitted":
        cursor.execute(
            "SELECT * FROM models WHERE approval_status=? LIMIT ? OFFSET ?",
            ("Approved", per_page, (page - 1) * per_page),
        )
    else:
        cursor.execute("SELECT * FROM models LIMIT ? OFFSET ?", (per_page, (page - 1) * per_page))

    models = cursor.fetchall()
    conn.close()
    return models


# Define the Gradio interface
def review_models():
    """
    Define the Gradio interface for reviewing models.
    """
    review_blocks = gr.Blocks()

    with review_blocks as demo2:
        with gr.Row():
            with gr.Column():

                def update_table(filter):
                    """
                    Update the model list table based on the selected filter.
                    """
                    try:
                        model_list = filtered_models(filter_status=filter)
                        columns = ['model_id', 'model_name', 'Time', 'approval_status']
                        df = pd.DataFrame(model_list, columns=columns)
                        return gr.List.update(df)
                    except Exception as e:
                        print("Error in update_checkpoints:", e)

                async def model_evaluation(evt: gr.SelectData):
                    """
                    Handle model evaluation when selected from the table.
                    """
                    print(f"You selected {evt.value} at {evt.index} from {evt.target}")
                    model_id = evt.index[0]
                    model_name = evt.value
                    public_url = await enable_inference(model=model_name)
                    N = 7
                    ramdom_str = random.choices(string.ascii_letters, k=N)
                    return gr.Textbox.update(value=ramdom_str)

                def open_link(text_url):
                    """
                    Open a link when text_url changes.
                    """
                    print("Opening URL", text_url)

                def open_link2(text_url):
                    """
                    Open a link when a button is clicked.
                    """
                    print("Opening URL on button click", text_url)
                    import webbrowser
                    url = extract_url_from_string(public_url)
                    webbrowser.get("chrome").open_new_tab(url)

                def filtered_models(page=1, per_page=10, filter_status=None):
                    """
                    Filter and sort models based on status and pagination.
                    """
                    models = fetch_models(page, per_page, filter_status)
                    sorted_models = sorted(models, key=lambda x: x[2], reverse=True)
                    return sorted_models

                model_list = filtered_models(filter_status="Incomplete")
                columns = ['model_id', 'model_name', 'Time', 'approval_status']
                df = pd.DataFrame(model_list, columns=columns)
                filter = gr.Radio(label="Filter", choices=["Incomplete", "Submitted", "Discarded"], value="Incomplete")
                page = gr.Slider(label="Page", minimum=1, maximum=10, value=1, visible=False)
                table = gr.List(
                    df,
                    type='array',
                    label="Model List"
                )
                text_url = gr.Textbox(value="", visible=False)

                btn = gr.Button(value="launcher", label="launcher", interactive=True,
                                link=public_url, elem_id="launcher", visible=False)
                btn.click(fn=open_link2, inputs=[text_url], outputs=None)
                filter.change(update_table, filter, table)
                table.select(model_evaluation, None, text_url)
                text_url.change(open_link, text_url, None, _js=javascript_function)

    return demo2


def main():
    interface = review_models()

    def launch_interface():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            interface.launch(server_name="0.0.0.0", server_port=8090)

        finally:
            loop.close()

    interface_thread = threading.Thread(target=launch_interface)
    interface_thread.start()
    address = "127.0.0.1"
    port = 8090
    public_url = ngrok.connect(addr=f'{address}:{port}')
    print("Public URL for reviewing:", public_url)


if __name__ == "__main__":
    main()
