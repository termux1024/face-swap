import os
import cv2
import insightface
from insightface.app import FaceAnalysis

class FaceSwapper:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx', download=True, download_zip=True
        )

    def swap_faces(self, source_img, source_face_idx, target_img, target_face_idx):
        source_faces = self.app.get(source_img)
        target_faces = self.app.get(target_img)

        source_faces = sorted(source_faces, key=lambda x: x.bbox[0])
        target_faces = sorted(target_faces, key=lambda x: x.bbox[0])

        if len(source_faces) < source_face_idx or source_face_idx < 1:
            raise ValueError(f"Source image contains {len(source_faces)} faces, but requested face {source_face_idx}")
        if len(target_faces) < target_face_idx or target_face_idx < 1:
            raise ValueError(f"Target image contains {len(target_faces)} faces, but requested face {target_face_idx}")

        source_face = source_faces[source_face_idx - 1]
        target_face = target_faces[target_face_idx - 1]

        result = self.swapper.get(target_img, target_face, source_face, paste_back=True)
        return result

def main():
    base_folder = "MultiplePics"
    dst_folder = os.path.join(base_folder, "dst")
    output_folder = os.path.join(base_folder, "output")
    source_img_name = "data_src.jpg"  # Change as needed
    source_face_idx = 1  # Change as needed
    target_face_idx = 1  # Change as needed

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    source_img_path = os.path.join(base_folder, source_img_name)
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        print(f"Could not read source image: {source_img_path}")
        return

    swapper = FaceSwapper()

    for filename in os.listdir(dst_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            target_img_path = os.path.join(dst_folder, filename)
            target_img = cv2.imread(target_img_path)
            if target_img is None:
                print(f"Could not read target image: {target_img_path}")
                continue
            try:
                result = swapper.swap_faces(
                    source_img=source_img,
                    source_face_idx=source_face_idx,
                    target_img=target_img,
                    target_face_idx=target_face_idx
                )
                output_path = os.path.join(output_folder, f"swapped_{filename}")
                cv2.imwrite(output_path, result)
                print(f"Swapped face saved to: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()