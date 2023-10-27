import gradio as gr
import numpy as np
import requests
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from decouple import config
import torch
import subprocess
import sqlite3
import pandas as pd
import threading


class StableDiffusionUIGenerator:
    """
    A class for generating a Gradio UI for image generation using Stable Diffusion models.
    """

    def __init__(self, model=None):
        self.prompt = ""
        self.models = self.get_all_models()
        self.ref_img_path = "ref.png"
        if model is None:
            self.model = ""
            self.ckpt_paths = self.init_ckpt_paths()
        else:
            self.model = model
            self.ckpt_paths = self.init_ckpt_paths(model=model)

        self.ckpt = ""

    def init_ckpt_paths(self, model=None):
        try:
            if not self.models:
                return []
            else:
                if model is None:
                    ckpts = self.get_checkpoints_for_model(self.models[0])
                else:
                    ckpts = self.get_checkpoints_for_model(model)
                return ckpts
        except Exception as e:
            print("Error in init_ckpts:", e)

    def get_all_models(self):
        try:
            conn = sqlite3.connect(config("DATABASE_NAME"))
            cursor = conn.cursor()

            try:
                cursor.execute("SELECT * FROM models")
                models = cursor.fetchall()
                sorted_models = sorted(models, key=lambda x: x[2], reverse=True)
                model_list = [model for model in sorted_models]
                columns = ['model_id', 'model_name', 'Time', 'approval_status']

                # Create a DataFrame from the list of tuples
                df = pd.DataFrame(model_list, columns=columns)
                return list(df["model_name"].values)

            except sqlite3.Error as e:
                print("Error:", e)
                return []

            finally:
                conn.close()
        except Exception as e:
            print("Error in get all models:", e)

    def get_checkpoints_for_model(self, model_name):
        try:
            conn = sqlite3.connect(config("DATABASE_NAME"))
            cursor = conn.cursor()

            try:
                # Get the model_id for the model
                cursor.execute("SELECT model_id FROM models WHERE model_name = ?", (model_name,))
                model_id = cursor.fetchone()

                if model_id:
                    # Retrieve checkpoints for the model
                    cursor.execute("SELECT checkpoint_name FROM checkpoints WHERE model_id = ?", (model_id[0],))
                    checkpoints = cursor.fetchall()
                    return [checkpoint[0] for checkpoint in checkpoints]
                else:
                    return None  # Model not found

            except sqlite3.Error as e:
                print("Error:", e)
                return None

            finally:
                conn.close()
        except Exception as e:
            print("Error in get_checkpoints_for_model:", e)

    def generate_grid(self, generated_images):
        """
        Generate a grid of images.

        Args:
            generated_images (list): List of generated images.

        Returns:
            numpy.ndarray: Combined grid image.
        """
        try:
            grid_size = (2, 1)
            grid_images = []
            count = 0
            for i in range(grid_size[0]):
                row_images = []
                for j in range(grid_size[1]):
                    generated_image = generated_images[count]
                    row_images.append(generated_image)
                    count += 1
                grid_images.append(row_images)

            return np.hstack([np.vstack(row) for row in grid_images])
        except Exception as e:
            print("Error in generate_grid:", e)

    def sdxl_generator(self):
        """
        Generate images using the Stable Diffusion XL model.

        Args:
            model_path (str): Path to the model checkpoint.

        Returns:
            numpy.ndarray: Combined grid image.
        """
        try:
            model_id = "runwayml/stable-diffusion-v1-5"
            unet = UNet2DConditionModel.from_pretrained(self.ckpt, subfolder="unet")
            text_encoder = CLIPTextModel.from_pretrained(self.ckpt, subfolder="text_encoder")
            pipe = DiffusionPipeline.from_pretrained(
                model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16, use_safetensors=True
            )
            pipe.to("cuda")

            seed = int(config("SEED"))
            n_steps = int(config("N_STEPS"))
            generator = torch.Generator(device="cuda").manual_seed(seed)

            generated_imgs = []

            for _ in range(2):
                image = pipe(
                    prompt=self.prompt,
                    num_inference_steps=n_steps,
                    cross_attention_kwargs={"scale": 1.0}
                ).images[0]
                generated_imgs.append(image)

            grid = self.generate_grid(generated_imgs)
            return grid
        except Exception as e:
            print("Error in sdxl_generator:", e)

    def update_model_status(self, model_name, approval_status):
        try:
            conn = sqlite3.connect(config("DATABASE_NAME"))
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE models SET approval_status = ? WHERE model_name = ?",
                (approval_status, model_name)
            )
            conn.commit()
        except Exception as e:
            print("Error in update_model_status:", e)
        finally:
            conn.close()

    def create_ui(self):
        try:
            inference_blocks = gr.Blocks()

            def generate_grids(selected_model, checkboxes, prompt):
                try:
                    self.model = dropdown
                    self.ckpt = checkboxes
                    self.prompt = prompt
                    grid = self.sdxl_generator()
                    ref_img = Image.open(self.ref_img_path)
                    return grid, ref_img
                except Exception as e:
                    print("Error:", e)
                    return None, None

            dropdown = gr.Dropdown(self.models, label="Models", value=self.model)
            checkboxes = gr.Radio(choices=self.ckpt_paths, label="Select Checkpoints")

            with inference_blocks as demo1:
                with gr.Row():
                    with gr.Column():
                        dropdown = gr.Dropdown(self.models, label="Models", value=self.model)
                        checkboxes = gr.Radio(choices=self.ckpt_paths, label="Select Checkpoints", value='str')
                        prompt = gr.Textbox(label="Prompt")
                        with gr.Column():
                            with gr.Row():
                                button_run = gr.Button(value="RunModel")
                                reload_button = gr.Button(value="Reload", size='sm')
                                submit_button = gr.Button(value="Submit", size='sm')

                    with gr.Column():
                        grids_out = gr.Image(type="pil", label="grid")
                        reference_img = gr.Image(type="pil", label="reference Image")

                    with gr.Column():
                        Overtrained = gr.Button(value="Overtrained")
                        Undertrained = gr.Button(value="Undertrained")
                        Unusable = gr.Button(value="Unusable")
                        Approved = gr.Button(value="Approved")
                        Next_Product = gr.Button(value="Next Product")

                        def update_checkpoints(selected_model):
                            try:
                                checkpoints = self.get_checkpoints_for_model(selected_model)
                                if checkpoints:
                                    self.ckpt_paths = checkpoints
                                    checkboxes.choices = self.ckpt_paths
                                    return gr.Radio.update(choices=self.ckpt_paths, label="Select Checkpoints")
                                else:
                                    self.ckpt_paths = ['no checkpoint']
                                    checkboxes.choices = self.ckpt_paths
                                    return gr.Radio.update(choices=self.ckpt_paths, label="Select Checkpoints")
                            except Exception as e:
                                print("Error in update_checkpoints:", e)

                        def submit_act(dropdown, checkboxes):
                            try:
                                api_url = config("API_URL")
                                # presigned_upload_url = call_api(api_url)
                                files = {'file': open(checkboxes, 'rb')}
                                r = requests.post(response['url'], data=response['fields'], files=files)
                                print(r.status_code)
                            except Exception as err:
                                print("error in uploading checkpoint to server: ", err)

                        def get_next_model_name(model_list, model_name):
                            try:
                                index = model_list.index(model_name)
                                if index < len(model_list) - 1:
                                    return model_list[index + 1]
                            except ValueError:
                                pass
                            return model_name

                        def next_button_reload():
                            try:
                                model = get_next_model_name(self.models, self.model)
                                self.model = model
                                self.prompt = ""
                                self.models = self.get_all_models()
                                checkpoints = self.get_checkpoints_for_model(self.model)
                                dropdown.choices = self.models
                                checkboxes.choices = checkpoints
                                prompt = ""
                                return gr.Dropdown.update(value=self.model), gr.Radio.update(
                                    choices=checkpoints, label="Select Checkpoints", value='str'), gr.Textbox.update(
                                    label="Prompt"), None, None
                            except Exception as e:
                                print("Error in reload:", e)

                        def reload():
                            try:
                                self.prompt = ""
                                self.models = self.get_all_models()
                                checkpoints = self.get_checkpoints_for_model(self.models[0])
                                dropdown.choices = self.models
                                checkboxes.choices = checkpoints
                                prompt = ""
                                return gr.Dropdown.update(choices=self.models, label="Models"), gr.Radio.update(
                                    choices=checkpoints, label="Select Checkpoints", value='str'), gr.Textbox.update(
                                    label="Prompt"), None, None
                            except Exception as e:
                                print("Error in reload:", e)

                        def handle_button_click(button_value, model):
                            selected_model = model
                            button_value = button_value
                            if button_value == "Overtrained":
                                self.update_model_status(selected_model, "Overtrained")
                            elif button_value == "Undertrained":
                                self.update_model_status(selected_model, "Undertrained")
                            elif button_value == "Unusable":
                                self.update_model_status(selected_model, "Unusable")
                            elif button_value == "Approved":
                                self.update_model_status(selected_model, "Approved")

                        Overtrained.click(fn=handle_button_click, inputs=[Overtrained, dropdown])
                        Undertrained.click(fn=handle_button_click, inputs=[Undertrained, dropdown])
                        Unusable.click(fn=handle_button_click, inputs=[Unusable, dropdown])
                        Approved.click(fn=handle_button_click, inputs=[Approved, dropdown])
                        Next_Product.click(fn=next_button_reload, inputs=None,
                                           outputs=[dropdown, checkboxes, prompt, reference_img, grids_out])

                        dropdown.change(update_checkpoints, dropdown, checkboxes)
                        button_run.click(fn=generate_grids,
                                         inputs=[dropdown, checkboxes, prompt],
                                         outputs=[grids_out, reference_img])
                        reload_button.click(reload, inputs=[],
                                            outputs=[dropdown, checkboxes, prompt, reference_img, grids_out])
                        submit_button.click(reload, inputs=[dropdown, checkboxes],
                                            outputs=[])
            return demo1

def run_gradio(ui_generator):
    interface = ui_generator.create_ui()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7861)

def main():
    ui_generator = StableDiffusionUIGenerator()
    interface = ui_generator.create_ui()
    interface.launch(share=True, server_name="0.0.0.0", server_port=7861)

if __name__ == "__main__":
    main()
