import os
import cv2
from SinglePhoto import FaceSwapper  # Use the shared class

def main():
    base_folder = "SingleSrcMultiDst"
    dst_folder = os.path.join(base_folder, "dst")
    output_folder = os.path.join(base_folder, "output")
    source_img_name = "data_src.jpg"  # Change as needed

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    source_img_path = os.path.join(base_folder, source_img_name)
    if not os.path.exists(source_img_path):
        print(f"Could not find source image: {source_img_path}")
        return

    swapper = FaceSwapper()

    # Ask user for source_face_idx (default 1)
    try:
        user_input = input("Enter the source face index (starting from 1, default is 1): ")
        source_face_idx = int(user_input) if user_input.strip() else 1
        if source_face_idx < 1:
            print("Invalid index. Using default value 1.")
            source_face_idx = 1
    except ValueError:
        print("Invalid input. Using default value 1.")
        source_face_idx = 1

    # Ask user for each target_face_idx for each destination image
    dest_images = [f for f in os.listdir(dst_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    dest_face_indices = {}
    for filename in dest_images:
        while True:
            try:
                idx_str = input(f"Enter target_face_idx for destination image '{filename}' (default 1): ")
                if idx_str.strip() == "":
                    idx = 1
                else:
                    idx = int(idx_str)
                if idx < 1:
                    print("Index must be >= 1.")
                    continue
                dest_face_indices[filename] = idx
                break
            except ValueError:
                print("Please enter a valid integer.")

    for filename in dest_images:
        target_img_path = os.path.join(dst_folder, filename)
        if not os.path.exists(target_img_path):
            print(f"Could not find target image: {target_img_path}")
            continue
        try:
            result = swapper.swap_faces(
                source_path=source_img_path,
                source_face_idx=source_face_idx,
                target_path=target_img_path,
                target_face_idx=dest_face_indices[filename]
            )
            output_path = os.path.join(output_folder, f"swapped_{filename}")
            cv2.imwrite(output_path, result)
            print(f"Swapped face saved to: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()