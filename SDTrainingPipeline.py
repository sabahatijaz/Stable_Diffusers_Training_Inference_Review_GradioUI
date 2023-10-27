from zipfile import ZipFile
import os
import shutil
import gdown
import subprocess
import json
import requests
from urllib.parse import urlparse, parse_qs
import traceback
import sqlite3
from io import BytesIO
from decouple import config


class SDTrainingPipeline:
    """A class representing a text-to-image training pipeline."""

    def __init__(self, model_name, class_token, instance_token, train_imgs, reg_imgs, n_epochs=200, check_pt_num=25):
        """
        Initialize the training pipeline.

        Args:
            model_name (str): Name of the model.
            class_token (str): Class token.
            instance_token (str): Instance token.
            train_imgs (str): URL of the training images folder.
            reg_imgs (str): URL of the regularization images folder.
            n_epochs (int, optional): Number of training epochs. Defaults to 200.
            check_pt_num (int, optional): Number of epochs in between saving a model. Defaults to 25.
        """
        self.model_name = model_name
        self.class_token = class_token
        self.instance_token = instance_token
        self.train_imgs = train_imgs
        self.reg_imgs = reg_imgs
        self.n_epochs = n_epochs
        self.check_pt_num = check_pt_num

    def get_file_ids(self, folder_url):
        response = requests.get(folder_url)
        file_ids = []

        if response.status_code == 200:
            content = response.text
            start = content.find('window["INITIAL_DATA"]') + len('window["INITIAL_DATA"]') + 1
            end = content.find(';', start)
            initial_data = content[start:end]

            start = initial_data.find('"ids":["') + len('"ids":["')
            end = initial_data.find('"]}', start)
            ids_string = initial_data[start:end]
            file_ids = ids_string.split('","')

        return file_ids

    def download_and_save_files(self, file_ids):
        zip_buffer = BytesIO()

        with ZipFile(zip_buffer, 'w') as zipf:
            for file_id in file_ids:
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                response = requests.get(download_url, stream=True)

                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0

                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            zip_buffer.write(chunk)
                            downloaded_size += len(chunk)
                            print(f"Downloading file with ID {file_id}: {downloaded_size}/{total_size} bytes")

                    zipf.writestr(file_id, zip_buffer.getvalue())
                    print(f"Downloaded file with ID {file_id}")

        with open('downloaded_folder.zip', 'wb') as f:
            f.write(zip_buffer.getvalue())

        print("Folder downloaded and saved as downloaded_folder.zip")

    def run_cmd(self, command):
        try:
            # Run the CMD command
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            # Check if the command was successful
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr

        except subprocess.CalledProcessError as e:
            return False, str(e)

    def extract_zip_to_folder(self, zip_path, extract_path):
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted contents to {extract_path}")

    def download_and_extract_folder(self, folder_url, output_path):
        try:
            os.makedirs(output_path, exist_ok=True)

            file_ids = self.get_file_ids(folder_url)
            self.download_and_save_files(file_ids)

            zip_path = 'downloaded_folder.zip'
            extract_path = output_path

            self.extract_zip_to_folder(zip_path, extract_path)

            # os.remove(zip_path)

            print("Downloaded, extracted, and cleaned up the folder contents successfully.")
        except Exception as e:
            print("Error during folder download, extraction, or cleaning:", e)

    def delete_folders_except_list(self, directory, folders_to_keep):
        try:
            # Ensure the provided directory exists
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory '{directory}' does not exist.")

            # Get a list of all items (files and folders) in the directory
            all_items = os.listdir(directory)

            # Iterate through the items in the directory
            for item in all_items:
                item_path = os.path.join(directory, item)

                # Check if the item is a folder and not in the list of folders to keep
                if os.path.isdir(item_path) and item not in folders_to_keep:
                    # Delete the folder and its contents
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    def clear_extra_files_folders(self, folder_path):
        """
        Remove unwanted files and folders from the specified directory.

        Args:
            folder_path (str): Path to the directory to be cleaned.

        Raises:
            Exception: If there is an error while clearing files and folders.
        """
        allowed_extensions = ['.jpg', '.png', '.txt']

        try:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isfile(item_path):
                    file_extension = os.path.splitext(item)[1].lower()
                    if file_extension not in allowed_extensions:
                        os.remove(item_path)
                        print(f"Removed file: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Removed directory: {item_path}")
            print("Files and folders removed successfully.")
        except Exception as e:
            print("Error while clearing extra files and folders:", e)

    # Function to add a new model and its checkpoint
    def add_model_with_checkpoint(self, model_name, checkpoint_name):
        conn = sqlite3.connect('models.db')
        cursor = conn.cursor()

        try:
            # Insert the model name (if it doesn't exist)
            cursor.execute("INSERT OR IGNORE INTO models (model_name) VALUES (?)", (model_name,))

            # Get the model_id for the model
            cursor.execute("SELECT model_id FROM models WHERE model_name = ?", (model_name,))
            model_id = cursor.fetchone()[0]

            # Insert the checkpoint
            cursor.execute("INSERT OR IGNORE INTO checkpoints (model_id, checkpoint_name) VALUES (?, ?)",
                           (model_id, checkpoint_name))

            conn.commit()
            return True  # Success
        except sqlite3.Error as e:
            print("Error:", e)
            conn.rollback()
            return False  # Error

        finally:
            conn.close()

    def create_metadata(self, source_dir):
        """
        Create a metadata file containing image and corresponding text information.

        Args:
            source_dir (str): Path to the directory containing images and text files.

        Raises:
            Exception: If there is an error while creating the metadata file.
        """
        metadata_entries = []

        try:
            for filename in os.listdir(source_dir):
                if filename.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(source_dir, filename)
                    text_filename = os.path.splitext(filename)[0] + '.txt'
                    text_path = os.path.join(source_dir, text_filename)

                    if os.path.exists(text_path):
                        with open(text_path, 'r') as text_file:
                            text_content = text_file.read().strip()

                        metadata_entry = {
                            "file_name": image_path,
                            "text": text_content
                        }

                        metadata_entries.append(metadata_entry)

            meta_file_path = os.path.join(source_dir, 'metadata.jsonl')
            with open(meta_file_path, 'w') as meta_file:
                for entry in metadata_entries:
                    json.dump(entry, meta_file)
                    meta_file.write('\n')
            print("meta.jsonl file generated successfully.")
        except Exception as e:
            print("Error while creating metadata:", e)

    def check_directory_availability(self, path):
        if os.path.exists(path):
            if os.path.isdir(path):
                1
            else:
                0
        else:
            0

    def count_images_in_directory(self, directory_path, image_extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
        """
        Count the number of images in a given directory.

        Args:
            directory_path (str): The path to the directory containing the images.
            image_extensions (list): List of valid image file extensions.

        Returns:
            int: The number of images found in the directory.
        """
        image_count = 0

        # Check if the directory exists
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

        # Iterate through files in the directory
        for filename in os.listdir(directory_path):
            file_extension = os.path.splitext(filename)[-1].lower()
            if file_extension in image_extensions:
                image_count += 1

        return image_count

    def create_model_command(self, train_data_dir, reg_data_dir):
        """
        Create and execute the command for training the model.

        Args:
            train_data_dir (str): Path to the training data directory.
            reg_data_dir (str): Path to the regularization data directory.

        Raises:
            subprocess.CalledProcessError: If there is an error while executing the training command.
        """
        try:
            # class_image_count=self.count_images_in_directory(reg_data_dir)
            command = f'''accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
                          --pretrained_model_name_or_path="{config("PRETRAINED_MODEL_NAME")}" \
                          --train_text_encoder \
                          --instance_data_dir "{train_data_dir}" \
                          --class_data_dir "{reg_data_dir}" \
                          --with_prior_preservation --prior_loss_weight=1.0 \
                          --instance_prompt "a photo of {self.instance_token} {self.class_token}" \
                          --class_prompt "a photo of {self.class_token}" \
                          --output_dir {self.model_name} \
                          --resolution {int(config("IMAGE_RESOLUTION"))} \
                          --train_batch_size {int(config("TRAIN_BATCH_SIZE"))} \
                          --use_8bit_adam \
                          --gradient_checkpointing \
                          --learning_rate {config("LEARNING_RATE")} \
                          --lr_scheduler "{config("LR_SCHEDULER")}" \
                          --mixed_precision "{config("MIXED_PRECISION")}" \
                          --checkpointing_steps {int(self.check_pt_num)} \
                          --lr_warmup_steps {int(config("LR_WARMUP_STEPS"))} \
                          --num_class_images 200 \
                          --max_train_steps {int(self.n_epochs)}'''

            result = self.run_cmd(command)
            count = self.n_epochs
            check_pt_num_input = self.check_pt_num
            print("Command output:")
            print(result)
            ckpt_paths = []
            extra_models = []
            for i in range(10):
                if len(ckpt_paths) < 3:
                    path = f"{self.model_name}/checkpoint-{int(count)}"
                    print(path)
                    # if self.check_directory_availability(path):
                    ckpt_paths.append(path)
                    extra_models.append(f"checkpoint-{int(count)}")
                    count = count - check_pt_num_input
                    self.add_model_with_checkpoint(self.model_name, path)
            print(ckpt_paths)

            self.delete_folders_except_list(self.model_name, extra_models)
        except subprocess.CalledProcessError as e:
            print("Error executing the command:", e)

    def run(self):
        """Execute the complete training pipeline."""
        try:
            output_path = f'{self.model_name}_train_imgs'
            output_path2 = f'{self.model_name}_reg_imgs'

            # self.download_and_extract_folder(self.train_imgs, output_path=output_path)
            # self.clear_extra_files_folders(output_path)

            # self.download_and_extract_folder(self.reg_imgs, output_path=output_path2)
            # self.clear_extra_files_folders(output_path2)

            # self.create_metadata(output_path)
            self.create_model_command(self.train_imgs, self.reg_imgs)

            return 1
        except Exception as err:
            print("Error in Run Function:", err)
            traceback.print_exc()  # Print the traceback information
            return 0
