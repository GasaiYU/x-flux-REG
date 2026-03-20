import os
import argparse
import torch
from PIL import Image, ExifTags
from src.flux.xflux_pipeline import XFluxPipeline

def main(args):
    # Set local model paths if provided
    if args.model_path:
        from src.flux.util import configs
        
        flux_path = os.path.join(args.model_path, "flux1-dev.safetensors")
        ae_path = os.path.join(args.model_path, "ae.safetensors")
        
        # Update the configs dictionary directly since it was already initialized on import
        if args.name in configs:
            configs[args.name].ckpt_path = flux_path
            configs[args.name].ae_path = ae_path
            print(f"Set local model paths in configs: FLUX_DEV={flux_path}, AE={ae_path}")
        
        os.environ["FLUX_DEV"] = flux_path
        os.environ["AE"] = ae_path

    # Initialize the pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device:
        device = args.device
        
    print(f"Initializing XFluxPipeline with model_type='{args.name}', device='{device}', offload={args.offload}")
    xflux_pipeline = XFluxPipeline(args.name, device, args.offload)
    
    # Read prompts from the text file
    if not os.path.exists(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(prompts)} prompts in {args.prompt_file}.")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate images for each prompt
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] Generating image for prompt: {prompt}")
        
        # Call the pipeline
        img = xflux_pipeline(
            prompt=prompt,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed if args.seed != -1 else torch.Generator(device="cpu").seed(),
            true_gs=args.true_gs,
            timestep_to_start_cfg=1, # Added this to match Gradio demo default
        )
        
        # Save the image
        filename = os.path.join(args.output_dir, f"output_{i:04d}.jpg")
        
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = args.name
        
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        print(f"Saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux Text-to-Image Inference")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing prompts (one per line)")
    parser.add_argument("--output_dir", type=str, default="output/t2i_results", help="Directory to save generated images")
    parser.add_argument("--name", type=str, default="flux-dev", help="Model name")
    parser.add_argument("--device", type=str, default="", help="Device to use (e.g., cuda, cpu)")
    parser.add_argument("--model_path", type=str, default="", help="Local path to the Flux.1-dev model directory")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    
    # Generation parameters
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--true_gs", type=float, default=3.5, help="True guidance scale")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    
    args = parser.parse_args()
    main(args)
