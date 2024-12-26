import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    _instance = None
    models_base_path = "./models"  # The base folder for models

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_model(self, model_name: str):
        """
        Loads a model from the specified directory within the models base path.
        """
        if self.model_name == model_name:
            logging.info(f"Model {model_name} already loaded.")
            return

        try:
            # Prepend the base path for the models directory
            model_path = os.path.join(self.models_base_path, model_name)

            # Check if the model path exists
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model path '{model_path}' does not exist.")

            logging.info(f"Loading model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            self.model_name = model_name
            logging.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the loaded model.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Please load the model first.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
