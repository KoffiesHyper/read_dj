import os
import torch
import librosa
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from transformers import Wav2Vec2ForCTC, AutoProcessor, AutoConfig
import numpy as np
import torch
import os
from safetensors.torch import load_file

class LoadModel:
    def __init__(self, model_path, device=None):
        """
        model_path: folder containing merged phoneme model (config.json + pytorch_model.bin)
        device: 'cuda' or 'cpu' (defaults to cuda if available)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load_model_and_processor(self):
        print(f"Loading merged model from: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        # Load the config to get the correct vocab_size
        config = AutoConfig.from_pretrained(self.model_path)
        correct_vocab_size = config.vocab_size

        # Initialize the model from scratch with the correct config
        self.model = Wav2Vec2ForCTC(config)

        # Load the model weights directly
        model_weights_path = os.path.join(self.model_path, "model.safetensors")
        if not os.path.exists(model_weights_path):
            model_weights_path = os.path.join(self.model_path, "pytorch_model.bin") # Fallback for older saves
        
        if os.path.exists(model_weights_path):
            state_dict = load_file(model_weights_path, device=str(self.device))
            self.model.load_state_dict(state_dict, strict=False) # strict=False to ignore lm_head mismatch initially
            print("Model weights loaded directly.")
        else:
            raise FileNotFoundError(f"No model weights found at {self.model_path}. Looked for model.safetensors or pytorch_model.bin")

        # Re-initialize lm_head to the correct vocab_size and load fine-tuned weights
        if hasattr(self.model, 'lm_head'):
            hidden_size = self.model.lm_head.in_features if hasattr(self.model.lm_head, "in_features") else self.model.config.hidden_size
            self.model.lm_head = torch.nn.Linear(hidden_size, correct_vocab_size, bias=True)
            
            lm_head_path = os.path.join(self.model_path, "lm_head_state_dict.bin")
            if os.path.exists(lm_head_path):
                self.model.lm_head.load_state_dict(torch.load(lm_head_path, torch.device('cpu')))
                print("Fine-tuned lm_head loaded and applied.")
            else:
                print(f"Warning: lm_head_state_dict.bin not found at {lm_head_path}. Using re-initialized lm_head.")
        self.model.to(self.device)
        self.model.eval()
        print("Model and processor loaded.")

    def get_model(self):
        return self.model
    
    def get_processor(self):
        return self.processor