
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

class VLABinPickingDataset(Dataset):
    def __init__(self, metadata_path, image_dir, processor):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.image_dir, os.path.basename(item["image_path"]))
        image = Image.open(image_path).convert("RGB")
        language_command = item["language_command"]
        action_sequence = item["action_sequence"]

        # Process inputs
        # The processor handles both image and text tokenization
        inputs = self.processor(images=image, text=language_command, return_tensors="pt")

        # For the action sequence, we need to tokenize it as target labels
        # Assuming the model's tokenizer can handle the action sequence format
        labels = self.processor.tokenizer(action_sequence, return_tensors="pt").input_ids

        # Flatten inputs for the model
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs["labels"] = labels.squeeze()

        return inputs

def collate_fn(batch):
    # Pad inputs and labels to the maximum length in the batch
    # This is a simplified collate_fn. In a real scenario, you'd handle padding more robustly
    # especially for different input types (pixel_values, input_ids, attention_mask)
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence([example["input_ids"] for example in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([example["attention_mask"] for example in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([example["labels"] for example in batch], batch_first=True, padding_value=-100) # -100 for ignore_index

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def fine_tune_smolvla(
    model_name="SmolVLA/SmolVLA-450M",
    data_dir="./data/synthetic_data",
    output_dir="./models/fine_tuned_smolvla",
    epochs=3,
    batch_size=4,
    learning_rate=1e-4,
    use_lora=True
):
    accelerator = Accelerator()

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name)

    if use_lora:
        # LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Common targets for attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM" # Or "SEQ_2_SEQ_LM" depending on model architecture
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    dataset = VLABinPickingDataset(
        metadata_path=os.path.join(data_dir, "dataset_metadata.json"),
        image_dir=data_dir,
        processor=processor
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    model.train()
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 10 == 0:
                accelerator.print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    processor.save_pretrained(output_dir)
    accelerator.print(f"Fine-tuning complete. Model saved to {output_dir}")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have run data_preparation.py first to generate synthetic data
    fine_tune_smolvla(
        model_name="SmolVLA/SmolVLA-450M",
        data_dir="./data/synthetic_data",
        output_dir="./models/fine_tuned_smolvla",
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        use_lora=True
    )

    print("Fine-tuning script finished.")


