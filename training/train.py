import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model
import yaml
from dataset import VLADataset
import dataclasses
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Disable wandb in CI environment
os.environ["WANDB_DISABLED"] = "true"

# Updated import based on the new lerobot structure
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError:
    # Fallback import if the structure changes
    try:
        from lerobot import SmolVLAPolicy
    except ImportError:
        # Final fallback to standard transformers model
        from transformers import AutoModelForCausalLM
        # Create a wrapper for compatibility
        class SmolVLAPolicy:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

def load_configs():
    with open('configs/model_config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open('configs/train_config.yaml', 'r') as f:
        train_config = yaml.safe_load(f)
    
    return model_config, train_config

def setup_model_and_tokenizer(model_config, train_config):
    # Load the model using the correct class
    model = SmolVLAPolicy.from_pretrained(model_config['model_name'])
    
    # Add to_dict method to config if it doesn't exist
    if not hasattr(model.config, 'to_dict'):
        def to_dict(self):
            return dataclasses.asdict(self)
        model.config.to_dict = to_dict.__get__(model.config, type(model.config))
    
    # Set dtype after loading if needed
    if train_config.get('use_fp16', False):
        model = model.half()
    
    # Use AutoProcessor instead of AutoTokenizer for SmolVLA models
    processor = AutoProcessor.from_pretrained(model.config.vlm_model_name)
    tokenizer = processor.tokenizer
    
    # Add special tokens for actions
    action_tokens = ["MOVE_TO", "OPEN_GRIPPER", "CLOSE_GRIPPER", "DONE"]
    tokenizer.add_tokens(action_tokens, special_tokens=True)
    
    # Check if the model has a resize_token_embeddings method
    if hasattr(model, 'resize_token_embeddings'):
        model.resize_token_embeddings(len(tokenizer))
    elif hasattr(model, 'model') and hasattr(model.model, 'resize_token_embeddings'):
        model.model.resize_token_embeddings(len(tokenizer))
    
    # Add a get method to the config to make it compatible with PEFT
    if not hasattr(model.config, 'get'):
        def config_get(key, default=None):
            return getattr(model.config, key, default) if hasattr(model.config, key) else default
        model.config.get = config_get
    
    # Ensure tie_word_embeddings is set
    if not hasattr(model.config, 'tie_word_embeddings'):
        model.config.tie_word_embeddings = False
    
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
    Standard format: {'pixel_values', 'input_ids', 'attention_mask'}
    SmolVLA format: {'observation.image', 'observation.image2', 'observation.image3', 'task', 'observation.state'}
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
        # This is a simplified approach - you might need to adjust based on your tokenizer
        task_texts = []
        for input_ids in batch['input_ids']:
            # Decode the input_ids to get the original text
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            task_texts.append(text)
        new_batch['task'] = task_texts
    
   
    if 'state' in batch:
        new_batch['observation.state'] = batch['state']
    else:
        # Create a dummy state if not available
        batch_size = batch['pixel_values'].shape[0]
        new_batch['observation.state'] = torch.zeros(batch_size, 7)  # Adjust size as needed
    
    return new_batch

def main():
    # Load configs
    model_config, train_config = load_configs()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_config, train_config)
    
    # Create dataset and dataloader
    dataset = VLADataset(
        data_path=train_config['data_path'],
        tokenizer=tokenizer,
        image_size=model_config['image_size']
    )
    
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
            loss, _ = model(batch)
            
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
                model.save_pretrained(f"{train_config['output_dir']}/checkpoint-{epoch}-{batch_idx}")
                tokenizer.save_pretrained(f"{train_config['output_dir']}/checkpoint-{epoch}-{batch_idx}")
        
        # Print epoch summary
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss}")
    
    # Save final model
    model.save_pretrained(train_config['output_dir'])
    tokenizer.save_pretrained(train_config['output_dir'])

if __name__ == "__main__":
    main()
