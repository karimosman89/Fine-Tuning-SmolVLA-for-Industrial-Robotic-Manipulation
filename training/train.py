import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
import yaml
from dataset import VLADataset
import dataclasses
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import sys
import traceback
import copy

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable wandb in CI environment
os.environ["WANDB_DISABLED"] = "true"

# Import local SmolVLA implementation
try:
    from lerobot.policies.smolvla.modeling_smolvla  import SmolVLAPolicy, SmolVLAConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig as ConfigClass
    from lerobot.policies.normalize import Normalize, Unnormalize
except ImportError as e:
    print(f"Error importing local modules: {e}")
    # Create minimal stubs for compatibility
    class SmolVLAPolicy:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    class SmolVLAConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

def load_configs():
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open('configs/train_config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    
    return model_config, train_config

def compute_dataset_stats(dataset, num_samples=1000):
    """Compute mean and std for dataset normalization"""
    # Use default values for demonstration
    stats = {
        'observation.image': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        'observation.image2': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        'observation.image3': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        'observation.state': {'mean': [0.0] * 7, 'std': [1.0] * 7},
        'action': {'mean': [0.0] * 7, 'std': [1.0] * 7}
    }
    return stats

def setup_model_and_tokenizer(model_config, train_config, dataset_stats=None):
    # Use AutoProcessor instead of AutoTokenizer for SmolVLA models
    vlm_model_name = model_config.get('vlm_model_name', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    processor = AutoProcessor.from_pretrained(vlm_model_name)
    tokenizer = processor.tokenizer
    
    # Add special tokens for actions
    action_tokens = ["MOVE_TO", "OPEN_GRIPPER", "CLOSE_GRIPPER", "DONE"]
    tokenizer.add_tokens(action_tokens, special_tokens=True)
    
    # Load the model using the correct class with dataset statistics
    try:
        # Instantiate a SmolVLAConfig object with no arguments
        config = SmolVLAConfig()
        
        # Pass the model_config values directly to the from_pretrained method
        model = SmolVLAPolicy.from_pretrained(
            model_config['model_name'],
            config=config,
            image_size=model_config['image_size'],
            vlm_model_name=vlm_model_name,
            use_peft=train_config.get('use_peft', True),
            lora_rank=train_config.get('lora_rank', 8)
        )
    except Exception as e:
        print(f"Failed to load SmolVLA model: {e}")
        print("Falling back to standard transformer model")
        # Fallback to standard transformer
        model = AutoModelForCausalLM.from_pretrained(model_config['model_name'])
    
    # Set dtype after loading if needed
    if train_config.get('use_fp16', False):
        model = model.half()
    
    # Check if the model has a resize_token_embeddings method
    if hasattr(model, 'resize_token_embeddings'):
        model.resize_token_embeddings(len(tokenizer))
    elif hasattr(model, 'model') and hasattr(model.model, 'resize_token_embeddings'):
        model.model.resize_token_embeddings(len(tokenizer))
    
    # Setup PEFT if enabled
    if train_config.get('use_peft', False):
        # For SmolVLA, we need to target the expert layers, not the VLM layers
        target_modules = [
            "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out",
            "state_proj", "vlm_with_expert.expert_layers"
        ]
        
        peft_config = LoraConfig(
            r=train_config.get('lora_rank', 16),
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        try:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"PEFT initialization failed: {e}. Continuing without PEFT.")
            train_config['use_peft'] = False
    
    return model, tokenizer


def convert_batch_format(batch, tokenizer):
    """
    Convert the batch from standard format to SmolVLA format.
    """
    # Create a new batch in SmolVLA format
    new_batch = {}
    
    # Convert images - use the same image for all three image fields
    if 'pixel_values' in batch:
        new_batch['observation.image'] = batch['pixel_values']
        new_batch['observation.image2'] = batch['pixel_values']
        new_batch['observation.image3'] = batch['pixel_values']
    
    # Convert text - use input_ids to reconstruct task text
    if 'input_ids' in batch:
        task_texts = []
        for input_ids in batch['input_ids']:
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            task_texts.append(text)
        new_batch['task'] = task_texts
    
    # Add state if available
    if 'state' in batch:
        new_batch['observation.state'] = batch['state']
    else:
        batch_size = batch['pixel_values'].shape[0]
        new_batch['observation.state'] = torch.zeros(batch_size, 7)
    
    return new_batch

def main():
    try:
        # Load configs
        model_config, train_config = load_configs()
        os.makedirs(train_config['output_dir'], exist_ok=True)
        
        # First, check if metadata exists
        if not os.path.exists(train_config['data_path']):
            raise FileNotFoundError(f"Metadata file not found: {train_config['data_path']}")
            
        # Set up tokenizer
        vlm_model_name = model_config.get('vlm_model_name', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
        processor = AutoProcessor.from_pretrained(vlm_model_name)
        tokenizer = processor.tokenizer
        
        # Add special tokens for actions
        action_tokens = ["MOVE_TO", "OPEN_GRIPPER", "CLOSE_GRIPPER", "DONE"]
        tokenizer.add_tokens(action_tokens, special_tokens=True)
        
        # Create dataset
        dataset = VLADataset(
            data_path=train_config['data_path'],
            tokenizer=tokenizer,
            image_size=model_config['image_size']
        )
        
        print(f"Dataset loaded with {len(dataset)} samples")
        
        # Compute dataset statistics
        dataset_stats = compute_dataset_stats(dataset)
        
        # Load model
        model, _ = setup_model_and_tokenizer(model_config, train_config, dataset_stats)
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=train_config['batch_size'],
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=float(train_config['learning_rate']),
            weight_decay=0.01
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'] * len(train_dataloader)
        )
        
        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        
        for epoch in range(train_config['num_epochs']):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_config['num_epochs']}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device and convert format
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch = convert_batch_format(batch, tokenizer)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Handle different model types
                if hasattr(model, 'forward'):
                    # SmolVLA model
                    loss, _ = model(batch)
                else:
                    # Standard transformer model
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Logging
                if batch_idx % train_config['logging_steps'] == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")
                
                # Save checkpoint
                if batch_idx % train_config['save_steps'] == 0:
                    checkpoint_dir = f"{train_config['output_dir']}/checkpoint-{epoch}-{batch_idx}"
                    os.makedirs(checkpoint_dir, exist_ok=True)  
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
            
            # Print epoch summary
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")
        
        # Save final model
        model.save_pretrained(train_config['output_dir'])
        tokenizer.save_pretrained(train_config['output_dir'])
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        # Ensure output directory exists and create an error file
        os.makedirs('output', exist_ok=True)
        with open('output/error.log', 'w') as f:
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
        # Create a dummy model file to allow the workflow to continue
        with open('output/dummy.model', 'w') as f:
            f.write("Dummy model - training failed")
        # Re-raise to fail the step
        raise e

if __name__ == "__main__":
    main()
