import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,  # Changed from AutoTokenizer to AutoProcessor
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
import yaml
from dataset import VLADataset

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
    
    # Setup PEFT if enabled
    if train_config.get('use_peft', False):
        peft_config = LoraConfig(
            r=train_config.get('lora_rank', 16),
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

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
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=train_config['output_dir'],
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        per_device_eval_batch_size=train_config['batch_size'],
        learning_rate=float(train_config['learning_rate']),
        warmup_steps=train_config['warmup_steps'],
        logging_steps=train_config['logging_steps'],
        save_steps=train_config['save_steps'],
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=train_config.get('use_fp16', False),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(train_config['output_dir'])

if __name__ == "__main__":
    main()
