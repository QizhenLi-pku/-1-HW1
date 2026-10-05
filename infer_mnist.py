import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification
def collate(batch):
    images, labels = zip(*batch)
    return [image.convert("RGB") for image in images], torch.tensor(labels)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    model_id = "microsoft/resnet-18"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ResNetForImageClassification.from_pretrained(model_id).to(device)
    model.eval()
    dataset = MNIST(root="data", train=False, download=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)
    print(f"Model: {model_id}; device: {device}; test samples: {len(dataset)}")
    print("WARNING: ImageNet and MNIST labels differ; this is raw-index agreement.")

    correct = total = 0
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="MNIST test inference"):
            # The checkpoint processor handles resizing/cropping to 224x224,
            # rescaling to [0, 1], and ImageNet normalization.
            inputs = processor(images=images, return_tensors="pt")
            assert inputs["pixel_values"].shape[1:] == (3, 224, 224)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            predictions = model(**inputs).logits.argmax(dim=-1).cpu()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    result = {
        "model": model_id,
        "dataset": "MNIST",
        "split": "test",
        "samples": total,
        "correct_raw_indices": correct,
        "accuracy": correct / total,
        "accuracy_percent": 100.0 * correct / total,
        "metric": "raw ImageNet index == MNIST digit label (label spaces differ)",
        "finetuned": False,
        "input_shape": [3, 224, 224],
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Raw-index baseline accuracy: {result['accuracy_percent']:.4f}% ({correct}/{total})")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
