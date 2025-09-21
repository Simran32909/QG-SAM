import os
import shutil
import random
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

def split_dataset(base_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Splits a dataset of images and masks into train, validation, and test sets.

    Args:
        base_dir (str): The path to the Kvasir-SEG dataset directory.
        output_dir (str): The path to save the split dataset.
        train_ratio (float): The proportion of the dataset to allocate for training.
        val_ratio (float): The proportion of the dataset to allocate for validation.
        seed (int): Random seed for reproducibility.
    """
    console = Console()
    console.print(f"[bold cyan]Starting dataset split...[/bold cyan]")
    
    # Define source directories
    images_src = Path(base_dir) / 'images'
    masks_src = Path(base_dir) / 'masks_png'

    # Check if source directories exist
    if not images_src.is_dir() or not masks_src.is_dir():
        console.print(f"[bold red]Error:[/bold red] Source 'images' or 'masks' directory not found in '{base_dir}'")
        return

    # Get list of image files
    image_files = [f for f in os.listdir(images_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        console.print(f"[bold red]Error:[/bold red] No images found in '{images_src}'. Please populate the directory.")
        return
        
    console.print(f"Found {len(image_files)} images to split.")

    # Shuffle the files for a random split
    random.seed(seed)
    random.shuffle(image_files)

    # Calculate split indices
    num_images = len(image_files)
    train_end = int(num_images * train_ratio)
    val_end = train_end + int(num_images * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    sets = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    # Create destination directories and copy files
    output_path = Path(output_dir)
    with Progress() as progress:
        main_task = progress.add_task("[blue]Copying files...", total=num_images)
        for set_name, files in sets.items():
            img_dest = output_path / set_name / 'images'
            mask_dest = output_path / set_name / 'masks'
            os.makedirs(img_dest, exist_ok=True)
            os.makedirs(mask_dest, exist_ok=True)

            for filename in files:
                img_src_path = images_src / filename
                # Assume mask has the same filename
                mask_src_path = masks_src / filename 

                if not mask_src_path.exists():
                    # If mask has a different extension (e.g. .png vs .jpg), try to find it
                    mask_found = False
                    for ext in ['.png', '.jpg', '.jpeg']:
                        potential_mask_path = masks_src / (Path(filename).stem + ext)
                        if potential_mask_path.exists():
                            mask_src_path = potential_mask_path
                            mask_found = True
                            break
                    if not mask_found:
                        progress.console.print(f"[yellow]Warning: Mask for {filename} not found. Skipping.[/yellow]")
                        progress.update(main_task, advance=1)
                        continue
                
                # Copy files
                shutil.copy(img_src_path, img_dest / filename)
                shutil.copy(mask_src_path, mask_dest / mask_src_path.name)
                progress.update(main_task, advance=1)
    
    console.print("\n[bold green]Dataset split successfully![/bold green]")
    console.print(f"  - Training set: {len(train_files)} images")
    console.print(f"  - Validation set: {len(val_files)} images")
    console.print(f"  - Test set: {len(test_files)} images")
    console.print(f"Split data is located in: [cyan]'{output_dir}'[/cyan]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split an image segmentation dataset into train, val, and test sets.")
    parser.add_argument(
        "base_dir", 
        type=str, 
        help="Path to the source dataset directory (e.g., 'Kvasir-SEG/')."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory where the split dataset will be saved."
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion for the training set.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion for the validation set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    args = parser.parse_args()

    # The test ratio is inferred from the train and val ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("The sum of train_ratio and val_ratio must be less than 1.")

    split_dataset(args.base_dir, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
