
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from scripts.data_preparation import RoboticSimulation, generate_natural_language_command, generate_action_sequence # Import for simulation

class VLABinPickingTestDataset(Dataset):
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

        inputs = self.processor(images=image, text=language_command, return_tensors="pt")
        labels = self.processor.tokenizer(action_sequence, return_tensors="pt").input_ids

        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs["labels"] = labels.squeeze()

        return inputs

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence([example["input_ids"] for example in batch], batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence([example["attention_mask"] for example in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([example["labels"] for example in batch], batch_first=True, padding_value=-100)

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def evaluate_smolvla(
    model_path="./models/fine_tuned_smolvla",
    test_data_dir="./data/synthetic_data", # Assuming test data is also in synthetic_data for now
    batch_size=4
):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    model.eval()

    dataset = VLABinPickingTestDataset(
        metadata_path=os.path.join(test_data_dir, "dataset_metadata.json"),
        image_dir=test_data_dir,
        processor=processor
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_examples = 0
    successful_actions = 0

    sim = RoboticSimulation() # Initialize simulation for ASR evaluation

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Generate predictions
            generated_ids = model.generate(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=50 # Max length for generated action sequence
            )

            for i in range(generated_ids.shape[0]):
                total_examples += 1
                predicted_action_sequence = processor.tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                true_action_sequence = processor.tokenizer.decode(batch["labels"][i][batch["labels"][i] != -100], skip_special_tokens=True)

                print(f"\n--- Example {total_examples} ---")
                print(f"True Action: {true_action_sequence}")
                print(f"Predicted Action: {predicted_action_sequence}")

                # Simulate execution to determine Action Success Rate (ASR)
                # This is a placeholder. A real ASR would involve a robust simulation environment.
                if sim.execute_action(predicted_action_sequence):
                    successful_actions += 1

    asr = (successful_actions / total_examples) * 100 if total_examples > 0 else 0
    print(f"\n--- Evaluation Results ---")
    print(f"Total Examples: {total_examples}")
    print(f"Successful Actions: {successful_actions}")
    print(f"Action Success Rate (ASR): {asr:.2f}%")

if __name__ == "__main__":
    # Example usage:
    # Ensure you have fine-tuned the model using fine_tune.py first
    evaluate_smolvla(
        model_path="./models/fine_tuned_smolvla",
        test_data_dir="./data/synthetic_data" # Use the same synthetic data for testing for now
    )
    print("Evaluation script finished.")


