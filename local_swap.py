import cv2
import insightface
from insightface.app import FaceAnalysis
import os

class FaceSwapper:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(
            'inswapper_128.onnx', download=True, download_zip=True
        )

    def swap_faces(self, source_path, source_face_idx, target_path, target_face_idx):
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(target_path)

        if source_img is None or target_img is None:
            raise ValueError("Could not read one or both images")

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
    # Paths relative to root
    source_path = os.path.join("SinglePhoto", "data_src.jpg")
    target_path = os.path.join("SinglePhoto", "data_dst.jpg")
    output_dir = os.path.join("SinglePhoto", "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    swapper = FaceSwapper()

    try:
        result = swapper.swap_faces(
            source_path=source_path,
            source_face_idx=1,
            target_path=target_path,
            target_face_idx=1
        )
        output_path = os.path.join(output_dir, "swapped_face.jpg")
        cv2.imwrite(output_path, result)
        print(f"Face swap completed successfully. Result saved to: {output_path}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()