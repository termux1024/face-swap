import os
import cv2

def frames_to_video(frames_folder, output_path, fps=30):
    images = [img for img in os.listdir(frames_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()
    if not images:
        print("No image frames found in the folder.")
        return

    first_frame = cv2.imread(os.path.join(frames_folder, images[0]))
    height, width, layers = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(frames_folder, image))
        if frame is None:
            print(f"Warning: Could not read {image}, skipping.")
            continue
        video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    frames_folder = "VideoSwapping/swapped_frames"
    output_path = "VideoSwapping/output.mp4"
    fps = 16
    frames_to_video(frames_folder, output_path, fps)