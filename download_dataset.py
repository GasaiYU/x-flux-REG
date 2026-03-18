import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def download_dataset(save_dir="./data/midjourney-v6-llava"):
    """
    Downloads the brivangl/midjourney-v6-llava dataset from Hugging Face
    and saves the images and metadata locally.
    """
    print(f"Downloading dataset to {save_dir}...")
    
    # Create directories
    images_dir = os.path.join(save_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load the dataset (streaming mode to avoid downloading everything into RAM at once)
    # Note: This dataset is large (~986k rows). We'll download it normally.
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("brivangl/midjourney-v6-llava", split="train")
    
    print(f"Total items in dataset: {len(dataset)}")
    
    # Create a metadata file
    metadata_path = os.path.join(save_dir, "metadata.jsonl")
    
    print("Saving images and metadata...")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(dataset)):
            try:
                # Get the image and metadata
                image = item["image"]
                prompt = item.get("prompt", "")
                llava_caption = item.get("llava", "")
                item_id = item.get("id", f"{i:09d}")
                
                # Save image
                image_filename = f"{item_id}_{i}.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                # Convert to RGB if necessary and save
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(image_path, quality=95)
                
                # Write metadata
                import json
                metadata = {
                    "file_name": f"images/{image_filename}",
                    "prompt": prompt,
                    "llava_caption": llava_caption,
                    "id": item_id
                }
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                
            except Exception as e:
                print(f"\nError processing item {i}: {e}")

if __name__ == "__main__":
    download_dataset()
