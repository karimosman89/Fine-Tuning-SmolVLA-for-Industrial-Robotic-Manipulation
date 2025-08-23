import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from safety_wrapper import SafetyWrapper
import yaml

class SmolVLAInference:
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Setup safety wrapper
        self.safety_wrapper = SafetyWrapper()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((self.config['image_size'], self.config['image_size']))
        image = np.array(image) / 255.0
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()  # Add batch dimension
        return image.to(self.device)
    
    def infer(self, image_path, command):
        # Preprocess image
        pixel_values = self.preprocess_image(image_path)
        
        # Tokenize command
        inputs = self.tokenizer(
            command, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate actions
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_length=self.config['max_action_length'],
                num_beams=5,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode actions
        action_sequence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the action part (after the command)
        action_text = action_sequence[len(command):].strip()
        
        # Parse and validate actions
        safe_actions = self.safety_wrapper.validate(action_text)
        
        return safe_actions

if __name__ == "__main__":
    # Example usage
    inference = SmolVLAInference("../output", "../configs/model_config.yaml")
    actions = inference.infer("test_image.png", "Pick up the bolt and place it on the conveyor belt.")
    print("Safe actions:", actions)
