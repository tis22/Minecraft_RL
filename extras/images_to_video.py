import cv2
import csv
from tqdm import tqdm

def images_to_video(csv_path, video_file, framerate=30):
    """
    Create a video from images listed in a CSV file.
    --------

    Args:
    csv_path (str):
        The path to the CSV file containing the image paths.
    video_file (str):
        The output video file name, including the file extension (e.g. 'output.mp4').
    framerate (int, optional):
        The framerate for the video (Default = 30).
    ---------

    Returns:
        None.
    """
    image_paths = []
    
    # Loading image paths from CSV
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        image_paths = [row[0].strip() for row in reader]
    
    if not image_paths:
        print("No image paths found in the CSV.")
        return

    # Loading first image to determine the size
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"Error loading the first image: {image_paths[0]}")
        return

    height, width, _ = first_image.shape
    target_width = 320
    target_height = 320
    aspect_ratio = width / height
    if height < target_height or width < target_width:
        target_width = int(aspect_ratio * target_height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(video_file, fourcc, framerate, (target_width, target_height))
    
    if not video_writer.isOpened():
        print("Error opening the video file.")
        return

    print(f"Creating video: {video_file}")

    try:
        frames_written = 0
        current_episode = None

        for i, image_path in tqdm(enumerate(image_paths), desc="Processing images", ncols=100):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading {image_path}")
                continue

            # Image processing
            image = cv2.flip(image, 1)
            image = cv2.flip(image, -1)

            # Extract the file name from the image path
            # Extracts only 'image_<episode>_<step>.png'
            file_name = image_path.split('/')[-1].strip()  

            # Check for the expected format and get episode number and step number
            if file_name.startswith('image') and file_name.count('_') == 2 and file_name.endswith('.png'):
                try:
                    name_parts = file_name.replace('image_', '').replace('.png', '').split('_')
                    episode_number = int(name_parts[0])
                    step_number = int(name_parts[1])
                except (IndexError, ValueError) as e:
                    print(f"Error parsing episode or step number in file: {file_name} -> {e}")
                    continue
            else:
                print(f"Unexpected file format: {file_name}")
                continue

            if episode_number != current_episode:
                current_episode = episode_number

            high_res_image = cv2.resize(image, (target_width * 2, target_height * 2))
            font_scale = target_height / 400
            margin = 5
            vertical_gap = 20

            episode_text = f"Episode {current_episode}"
            step_text = f"Step {step_number}"

            episode_position = (margin, vertical_gap)
            step_position = (margin, vertical_gap + 30)

            # Add the numbers to the frame
            cv2.putText(high_res_image, episode_text, episode_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            cv2.putText(high_res_image, step_text, step_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), 2)

            image_resized = cv2.resize(high_res_image, (target_width, target_height))
            video_writer.write(image_resized)
            frames_written += 1

        video_writer.release()

    except KeyboardInterrupt:
        print("Process interrupted. Releasing video writer...")
        video_writer.release()

    if frames_written == 0:
        print("No frames were added to the video!")
    else:
        print(f"Video created successfully with {frames_written} frames.")

if __name__ == "__main__":
    csv_path = "image_paths.csv"
    video_file = "output.mp4"
    framerate = 30

    images_to_video(csv_path, video_file, framerate)