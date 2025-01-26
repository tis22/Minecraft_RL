import os
import csv
from tqdm import tqdm
import re

# Set folder path
source_folder = "./training_20241226_131530"

output_csv = "image_paths.csv"
image_extensions = {".png"}

def get_sorted_image_paths(folder):
    """
    Collect image file paths from the folder and sort them by episode and step.
    --------
    
    Args:
    folder (str):
        The path to the folder containing the images.
    ---------
    
    Returns:
    list:
        A sorted list of image file paths.
    """
    image_paths = []
    
    # Calculate the number of files for the tqdm-bar
    total_files = sum(len(files) for _, _, files in os.walk(folder))

    # Go through all files in the folder (recursively)
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
                pbar.update(1)

    def extract_episode_and_step(file_name):
        """
        Extract the episode and step numbers from the file name.
        --------
        
        Args:
        file_name (str):
            The file name of the image from which the episode and step should be extracted.
        ---------
        
        Returns:
        tuple:
            A tuple containing the episode and step numbers (episode, step).
            If the file name does not match the expected format, (float('inf'), float('inf')) is returned.
        """
        match = re.match(r"image_(\d+)_(\d+)\.png", file_name)
        if match:
            episode = int(match.group(1))
            step = int(match.group(2))
            return (episode, step)
        return (float('inf'), float('inf'))  # If the format does not fit, sort to the back.

    # Sort the list by episode and step.
    image_paths.sort(key=lambda x: extract_episode_and_step(os.path.basename(x)))
    return image_paths

def write_paths_to_csv(paths, output_file):
    """
    Write the image file paths to a CSV file.
    --------
    
    Args:
    paths (list):
        A list of image file paths to be written to the CSV file.
    output_file (str):
        The path to the output file where the image paths will be written.
    ---------
    
    Returns:
        None.
    """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_path"])
        for path in tqdm(paths, desc="Writing to CSV"):
            writer.writerow([path])

if __name__ == "__main__":
    print("Collect and sort image file paths...")
    sorted_image_paths = get_sorted_image_paths(source_folder)
    print(f"{len(sorted_image_paths)} image files found.")
    print("Writing paths to CSV-file...")
    write_paths_to_csv(sorted_image_paths, output_csv)
    print(f"Done! Paths saved to '{output_csv}'.")
