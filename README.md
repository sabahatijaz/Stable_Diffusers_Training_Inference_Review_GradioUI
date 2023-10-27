# Stable_Diffusers_Training_Inference_Review_GradioUI
Stable Diffusers 1.5 Model Training and Inference Generation Pipeline

Stable Diffusers <!-- Add a relevant image here -->

Welcome to the Stable Diffusers 1.5 Model Training and Inference Generation Pipeline repository! This cutting-edge solution empowers you to train, infer, and review stable diffuser models seamlessly. Whether you're fine-tuning existing models, generating inferences, or evaluating model performance, this pipeline has you covered.
Features

    User-Friendly UIs: Intuitive graphical interfaces for training, inference generation, and model review.
    Efficient Training Pipeline: Streamlined training process with stable diffusers models, ensuring quick and accurate model creation.
    Dynamic Inference Generation: Generate inferences using pre-trained models, providing rapid insights from your data.
    Intelligent Model Review: Review models effortlessly, marking them as "Overtrained," "Undertrained," or "Unusable" for precise evaluation.
    Database Integration: Utilizes SQLite schema for seamless storage and retrieval of model data.
    Automated Data Clearance: Included script to clear database data, ensuring a clean slate for new experiments.
    Pyngrok Integration: Features Pyngrok tunneling capabilities for secure access to UIs remotely.

## Setup Instructions
1. Clone the Repository

    
    git clone <repository-url>

    cd repo

2. Install Dependencies


    pip install -r requirements.txt

    apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

3. Configure Accelerate


    accelerate config default

4. Install Diffusers Library


    git clone https://github.com/huggingface/diffusers

    cd diffusers

    pip install -e .

    pip install -r examples/dreambooth/requirements.txt

5. Install SQLite


    apt-get install sqlite3

6. Prepare Data


    Obtain the instance and class directories and extract them.
    Provide the folder path containing images only as the input parameter during training.

## Usage 
### Training UI


Run the following command to launch the training UI:



    python Training_UI.py

Access the UI at http://localhost:7860 for training stable diffuser models.
### Inference UI

Run the following command to launch the inference UI:


    python Inference_UI.py

Access the UI at http://localhost:7861 for generating inferences from pre-trained models.
### Model Review UI

Run the following command to launch the model review UI:


    python Review_UI.py

Access the UI at http://localhost:5002 for reviewing and marking models based on their performance.
### Clear Database Data

To clear database data, run:


    python clear_db.py
