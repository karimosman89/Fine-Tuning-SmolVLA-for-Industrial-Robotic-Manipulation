import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

class VLADataset(Dataset):
    def __init__(self, data_path, tokenizer, image_size=224):
        with open(data_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.tokenizer = tokenizer
        self.image_size = image_size
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load and preprocess image
        image = Image.open(item['image_path'])
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).float()  # Convert to CHW
        
        # Tokenize text (command + actions)
        text = f"{item['command']} {item['actions']}"
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        
        
        state = torch.zeros(7, dtype=torch.float32)
        action = torch.zeros(7, dtype=torch.float32)
        
        return {
            'pixel_values': image,
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'state': state,
            'action': action
        }
