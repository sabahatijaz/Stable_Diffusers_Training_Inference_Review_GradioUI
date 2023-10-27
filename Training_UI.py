import os
import asyncio
import subprocess
import threading
import webbrowser
import re
import sqlite3
from collections import Counter
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import tracemalloc
import torch
from PIL import Image
from decouple import config
from pyngrok import ngrok
import gradio as gr
from Inference_UI import StableDiffusionUIGenerator
from SDTrainingPipeline import SDTrainingPipeline
from schema import check_database_schema, create_schema

tracemalloc.start()

# Global variable to store training logs
training_logs = []

# Set environment variables
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# Global variable to store thread output
thread_output = 0
model_name_in = ""


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


async def df_checker():
    """
    Check and create the database schema if it doesn't exist.
    """
    database_path = config("DATABASE_NAME")

    expected_models_table_sql = '''
        CREATE TABLE models (
            model_id INTEGER PRIMARY KEY,
            model_name TEXT UNIQUE NOT NULL
        )
    '''

    expected_checkpoints_table_sql = '''
        CREATE TABLE checkpoints (
            checkpoint_id INTEGER PRIMARY KEY,
            model_id INTEGER NOT NULL,
            checkpoint_name TEXT NOT NULL,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    '''

    expected_schema = [expected_models_table_sql, expected_checkpoints_table_sql]

    res = check_database_schema(database_path, expected_schema)
    if res == 0:
        print("Creating DB schema")
        database_path = config("DATABASE_NAME")
        conn = sqlite3.connect(database_path)
        create_schema(conn)
        conn.close()


async def on_button_click(ui_generator):
    """
    Launch the Gradio interface.
    """
    interface = ui_generator.create_ui()
    await interface.launch(server_name="127.0.0.1", server_port=int(config("INFERENCE_PORT")))


async def enable_inference():
    """
    Enable inference and return the public URL.
    """
    loop = get_or_create_eventloop()
    ui_generator = StableDiffusionUIGenerator()

    def run_on_button_click():
        asyncio.run(on_button_click(ui_generator))

    gradio_thread = threading.Thread(target=run_on_button_click)
    gradio_thread.start()
    address = "0.0.0.0"
    port = config("INFERENCE_PORT")
    public_url = ngrok.connect(addr=f'{address}:{port}')
    print("Public URL for inference:", public_url)
    return public_url


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


async def main():
    """
    Main program logic.
    """
    loop1 = get_or_create_eventloop()
    await df_checker()
    public_url = await enable_inference()
    public_url = extract_url_from_string(str(public_url))
    auth_token = config("PYNGROK_AUTH_TOKEN")
    ngrok_auth_command = ["ngrok", "authtoken", auth_token]

    try:
        subprocess.run(ngrok_auth_command, check=True)
        print("ngrok authentication token set successfully!")
    except subprocess.CalledProcessError as e:
        print("Error setting ngrok authentication token:", e)

    def run_training(model_name, class_token, instance_token, train_imgs,
                     reg_imgs, n_epochs, check_pt_num):
        """
        Run the training pipeline.
        """
        pipeline = SDTrainingPipeline(model_name, class_token, instance_token, train_imgs, reg_imgs, n_epochs,
                                      check_pt_num)
        val = pipeline.run()
        print(val)
        return val

    def thread_function(model_name, class_token, instance_token, train_imgs,
                        reg_imgs, n_epochs, check_pt_num):
        """
        Thread function to run training in a separate thread.
        """
        global thread_output
        thread_output = run_training(model_name, class_token, instance_token, train_imgs, reg_imgs, n_epochs, check_pt_num)

    def object_identification(direc):
        """
        Identify objects in images.
        """
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print("blip2 loaded")

        all_answers = []
        for filename in os.listdir(direc):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(direc, filename)
                image = Image.open(image_path)

                prompt = "Question: Is the person wearing sneakers, boots, or other? Just give one word answer Answer:"
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                print(generated_text)
                answer = generated_text
                all_answers.append(answer)

        most_common_answer = Counter(all_answers).most_common(1)[0][0]
        return most_common_answer

    def manager(model_name, n_epochs, check_pt_num, training_logs_output):
        """
        Manager function to coordinate training and log display.
        """
        loop = get_or_create_eventloop()
        global model_name_in
        model_name_in = model_name
        train_imgs = config("training_images_dir")
        reg_imgs = config("regularization_images_dir")
        class_token = object_identification(reg_imgs)
        instance_token = config("instance_token_input")

        t1 = threading.Thread(target=thread_function, args=(model_name, class_token, instance_token, train_imgs,
                                                            reg_imgs, n_epochs, check_pt_num))
        print("Starting thread!")
        t1.start()

    def open_link():
        """
        Open the public URL in the browser.
        """
        import webbrowser
        url = extract_url_from_string(public_url)
        webbrowser.get("chrome").open_new_tab(url)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_name_input = gr.Textbox(value="test", label="Model Name")
                n_epochs_input = gr.Number(value=int(config("DEFAULT_NUM_EPOCHS")), label="Number of Epochs")
                check_pt_num_input = gr.Number(value=int(config("DEFAULT_CHECK_PT_NUM")), label="Checkpoint Number")
                run_button = gr.Button(value="Run Model", label="Run Model")
                inference_button = gr.Button(value="Inference", label="Inference", interactive=True, link=public_url)
            with gr.Column():
                training_logs_output = gr.Textbox(label="Training Logs", lines=1000)

            run_button.click(
                fn=manager,
                inputs=[model_name_input, n_epochs_input, check_pt_num_input, training_logs_output],
                outputs=[training_logs_output],
            )
            inference_button.click(
                fn=open_link
            )

            def get_event_file_path(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.startswith("events.out.tfevents."):
                        return os.path.join(folder_path, filename)

                return None

            def display_event_file_logs(training_logs_output, model_name_input, n_epochs_input, check_pt_num_input):
                """
                Display training logs from event files.
                """
                import tensorflow.compat.v1 as tf
                tf.disable_v2_behavior()
                try:
                    if thread_output == 1:
                        pass  # GENERATE NOTIFICATION
                    global model_name_in
                    folder_path = f"{model_name_in}/logs/dreambooth"
                    event_file_path = get_event_file_path(folder_path)

                    captured_logs = []
                    for event in tf.train.summary_iterator(event_file_path):
                        for value in event.summary.value:
                            captured_logs.append(f"Iteration Step: {event.step}\nTag: {value.tag}")
                            if value.HasField('simple_value'):
                                captured_logs.append(f"Simple Value: {value.simple_value}")

                    decoded_content = '\n'.join(captured_logs)
                    time.sleep(2)
                    return decoded_content
                except Exception as err:
                    print("Exception occurred in display_event_file_logs:", err)
                    return "Once the model runs, your logs will be displayed here! Stay with us!"

            demo.load(fn=display_event_file_logs,
                      inputs=[training_logs_output, model_name_input, n_epochs_input, check_pt_num_input],
                      outputs=[training_logs_output], show_error=True, every=1)

            def launch_interface():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    demo.queue(concurrency_count=5, max_size=20).launch(max_threads=20, show_error=True)
                finally:
                    loop.close()
                    loop1.close()

            interface_thread = threading.Thread(target=launch_interface)
            interface_thread.start()

            address = "127.0.0.1"
            port = config("TRAINING_PORT")
            public_url = ngrok.connect(addr=f'{address}:{port}')
            print("Public URL for training:", public_url)


if __name__ == "__main__":
    loop = get_or_create_eventloop()
    asyncio.run(main())
