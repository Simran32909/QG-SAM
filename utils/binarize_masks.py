import os
import cv2
import argparse
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn

def binarize_masks(source_dir, dest_dir):
    """
    Reads all images from a source directory, binarizes them, and saves
    them as PNG files in a destination directory.

    Args:
        source_dir (str): Path to the directory with original masks.
        dest_dir (str): Path to the directory where cleaned PNG masks will be saved.
    """
    console = Console()
    
    if not os.path.isdir(source_dir):
        console.print(f"[bold red]Error:[/bold red] Source directory not found at '{source_dir}'")
        return

    try:
        os.makedirs(dest_dir, exist_ok=True)
        console.print(f"[bold cyan]Ensured destination directory exists:[/bold cyan] {dest_dir}")
    except OSError as e:
        console.print(f"[bold red]Error:[/bold red] Could not create destination directory '{dest_dir}'. Reason: {e}")
        return

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    if not image_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No image files found in the source directory.")
        return

    console.print(f"Found {len(image_files)} images to process.")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("({task.completed}/{task.total})"),
    ) as progress:
        task = progress.add_task("[green]Binarizing masks...", total=len(image_files))

        for filename in image_files:
            try:
                source_path = os.path.join(source_dir, filename)
                
                # Create new filename with .png extension
                base_name = os.path.splitext(filename)[0]
                dest_filename = f"{base_name}.png"
                dest_path = os.path.join(dest_dir, dest_filename)

                # Read image in grayscale
                mask = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    progress.console.print(f"[yellow]Skipping {filename} (could not read).[/yellow]")
                    progress.update(task, advance=1)
                    continue

                # Binarize the image. Any pixel value > 0 will be set to 255.
                _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                
                # Save the result as a lossless PNG
                cv2.imwrite(dest_path, binary_mask)

            except Exception as e:
                progress.console.print(f"[red]Failed to process {filename}: {e}[/red]")
            
            progress.update(task, advance=1)

    console.print("\n[bold green]Processing complete![/bold green]")
    console.print(f"Cleaned masks have been saved to [cyan]'{dest_dir}'[/cyan] in PNG format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binarize mask images and save them in a lossless PNG format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "source_dir", 
        type=str, 
        help="Path to the source directory containing the original mask images."
    )
    parser.add_argument(
        "dest_dir",
        type=str,
        help="Path to the destination directory where cleaned PNG masks will be saved."
    )
    args = parser.parse_args()
    
    binarize_masks(args.source_dir, args.dest_dir)
