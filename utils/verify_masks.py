import os
import cv2
import numpy as np
import argparse
from rich.console import Console
from rich.table import Table

def verify_masks(masks_dir, num_to_check=10):
    """
    Checks mask images in a directory to see if they are binary or contain
    artifacts from lossy compression.

    Args:
        masks_dir (str): Path to the directory containing mask images.
        num_to_check (int): The number of images to sample from the directory.
                            Set to 0 to check all images.
    """
    console = Console()
    console.print(f"[bold cyan]Starting mask verification for directory:[/bold cyan] {masks_dir}")

    if not os.path.isdir(masks_dir):
        console.print(f"[bold red]Error:[/bold red] Directory not found at '{masks_dir}'")
        return

    image_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    if not image_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No image files found in the directory.")
        return

    if num_to_check > 0 and len(image_files) > num_to_check:
        # Check a random sample of N images
        import random
        files_to_check = random.sample(image_files, num_to_check)
        console.print(f"Found {len(image_files)} images. Checking a random sample of {num_to_check}.")
    else:
        # Check all images
        files_to_check = image_files
        console.print(f"Found {len(image_files)} images. Checking all of them.")

    table = Table(title="Mask Pixel Value Analysis")
    table.add_column("Filename", justify="left", style="cyan", no_wrap=True)
    table.add_column("Unique Values", justify="left", style="magenta")
    table.add_column("Status", justify="center", style="green")

    unclean_masks = []

    for filename in files_to_check:
        try:
            filepath = os.path.join(masks_dir, filename)
            # Read the image in grayscale
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                table.add_row(filename, "COULD NOT READ", "[bold red]Error[/bold red]")
                unclean_masks.append(filename)
                continue

            unique_values = np.unique(mask)
            
            # Check if the mask is perfectly binary (only 0 and 255)
            is_clean = set(unique_values).issubset({0, 255})
            
            status = "[bold green]Clean[/bold green]" if is_clean else "[bold yellow]Needs Binarization[/bold yellow]"
            if not is_clean:
                unclean_masks.append(filename)
            
            # For display, truncate long lists of unique values
            values_str = np.array2string(unique_values, separator=', ')
            if len(values_str) > 70:
                values_str = values_str[:70] + "..."

            table.add_row(filename, values_str, status)

        except Exception as e:
            table.add_row(filename, f"ERROR: {e}", "[bold red]Failed[/bold red]")
            unclean_masks.append(filename)
    
    console.print(table)
    console.print("\n[bold]Verification Summary:[/bold]")

    if not unclean_masks:
        console.print("[bold green]All checked masks appear to be clean (binary). You are good to go![/bold green]")
    else:
        console.print(f"[bold yellow]Found {len(unclean_masks)} masks that are not clean.[/bold yellow]")
        console.print("These masks contain pixel values other than 0 and 255, likely due to JPEG compression.")
        console.print("You should binarize these masks and save them in a lossless format like PNG before using them for training or evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify that mask images are binary (black and white).")
    parser.add_argument(
        "masks_dir", 
        type=str, 
        help="Path to the directory containing the mask images."
    )
    parser.add_argument(
        "--num_check",
        type=int,
        default=1000,
        help="Number of random images to check. Set to 0 to check all images in the directory."
    )
    args = parser.parse_args()
    
    verify_masks(args.masks_dir, args.num_check)
