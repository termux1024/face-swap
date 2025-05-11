import cv2
import os
from SinglePhoto import FaceSwapper  # Updated import

def main():
    src_dir = os.path.join("MultiSrcMultiDst", "src")
    dst_dir = os.path.join("MultiSrcMultiDst", "dst")
    output_root = os.path.join("MultiSrcMultiDst", "output")

    os.makedirs(output_root, exist_ok=True)

    swapper = FaceSwapper()

    # Get all source and destination image paths
    source_images = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    dest_images = [os.path.join(dst_dir, f) for f in os.listdir(dst_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Ask user for target_face_idx for each destination image
    dest_face_indices = {}
    for dest_path in dest_images:
        dest_name = os.path.basename(dest_path)
        while True:
            try:
                idx_str = input(f"Enter target_face_idx for destination image '{dest_name}' (default 1): ")
                if idx_str.strip() == "":
                    idx = 1
                else:
                    idx = int(idx_str)
                if idx < 1:
                    print("Index must be >= 1.")
                    continue
                dest_face_indices[dest_path] = idx
                break
            except ValueError:
                print("Please enter a valid integer.")

    for source_path in source_images:
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        source_output_dir = os.path.join(output_root, source_name)
        os.makedirs(source_output_dir, exist_ok=True)

        for dest_path in dest_images:
            dest_name = os.path.splitext(os.path.basename(dest_path))[0]
            target_face_idx = dest_face_indices[dest_path]
            try:
                result = swapper.swap_faces(
                    source_path=source_path,
                    source_face_idx=1,
                    target_path=dest_path,
                    target_face_idx=target_face_idx
                )
                output_path = os.path.join(source_output_dir, f"{source_name}_to_{dest_name}.jpg")
                cv2.imwrite(output_path, result)
                print(f"Swapped {source_name} -> {dest_name}: {output_path}")
            except Exception as e:
                print(f"Error swapping {source_name} to {dest_name}: {str(e)}")

if __name__ == "__main__":
    main()