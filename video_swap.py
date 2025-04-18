import cv2
import os
import shutil
from local_swap import FaceSwapper

def extract_frames(video_path, frames_dir):
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        idx += 1
    cap.release()
    return frame_paths

def frames_to_video(frames_dir, output_video_path, fps):
    frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    if not frames:
        print("No frames found in directory.")
        return
    first_frame = cv2.imread(frames[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
    out.release()

def main():
    # Use files from VideoSwapping folder
    video_path = os.path.join("VideoSwapping", "data_dst.mp4")
    source_image_path = os.path.join("VideoSwapping", "data_src.jpg")
    frames_dir = os.path.join("VideoSwapping", "video_frames")
    swapped_dir = os.path.join("VideoSwapping", "swapped_frames")
    output_video_path = os.path.join("VideoSwapping", "output_swapped_video.mp4")
    source_face_idx = 1
    dest_face_idx = 1

    # Extract frames
    print("Extracting frames from video...")
    frame_paths = extract_frames(video_path, frames_dir)

    # Prepare output directory
    if not os.path.exists(swapped_dir):
        os.makedirs(swapped_dir)

    # Initialize face swapper
    swapper = FaceSwapper()

    # Swap faces on each frame
    print("Swapping faces on frames...")
    for idx, frame_path in enumerate(frame_paths):
        try:
            swapped = swapper.swap_faces(
                source_path=source_image_path,
                source_face_idx=source_face_idx,
                target_path=frame_path,
                target_face_idx=dest_face_idx
            )
            out_path = os.path.join(swapped_dir, f"swapped_{idx:05d}.jpg")
            cv2.imwrite(out_path, swapped)
        except Exception as e:
            print(f"\nFrame {idx}: {e}")
            # Optionally, copy the original frame if swap fails
            cv2.imwrite(os.path.join(swapped_dir, f"swapped_{idx:05d}.jpg"), cv2.imread(frame_path))
        print(f"Swapping frame {idx+1}/{len(frame_paths)}", end='\r')
    print()  # Move to the next line after the loop

    # Get FPS from original video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Combine swapped frames into video
    print("Combining swapped frames into video...")
    frames_to_video(swapped_dir, output_video_path, fps)
    print(f"Done! Output video saved as {output_video_path}")

    # Ask user if they want to keep the extracted frames and swapped images
    answer = input("Do you want to keep the extracted frames and swapped images? (y/n): ").strip().lower()
    if answer == 'n':
        try:
            shutil.rmtree(frames_dir)
            shutil.rmtree(swapped_dir)
            print("Temporary folders deleted.")
        except Exception as e:
            print(f"Error deleting folders: {e}")
    else:
        print("Temporary folders kept.")

if __name__ == "__main__":
    main()