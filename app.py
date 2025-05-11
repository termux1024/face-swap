import gradio as gr
import os
import cv2
from SinglePhoto import FaceSwapper

wellcomingMessage = """
    <h1>Face Swapping Suite</h1>
    <p>All-in-one face swapping: single photo, video, multi-source, and multi-destination!</p>
"""

swapper = FaceSwapper()

def swap_single_photo(src_img, src_idx, dst_img, dst_idx):
    try:
        src_path = "tmp_src.jpg"
        dst_path = "tmp_dst.jpg"
        cv2.imwrite(src_path, src_img)
        cv2.imwrite(dst_path, dst_img)
        result = swapper.swap_faces(src_path, int(src_idx), dst_path, int(dst_idx))
        return result
    except Exception as e:
        return f"Error: {e}"

def swap_video(src_img, src_idx, video, dst_idx):
    # Save source image
    src_path = "tmp_src.jpg"
    cv2.imwrite(src_path, src_img)
    # Save video
    video_path = "tmp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video.read())
    # Extract frames
    frames_dir = "tmp_video_frames"
    swapped_dir = "tmp_swapped_frames"
    output_video_path = "tmp_output_video.mp4"
    from VideoSwapping import extract_frames, frames_to_video
    frame_paths = extract_frames(video_path, frames_dir)
    os.makedirs(swapped_dir, exist_ok=True)
    for idx, frame_path in enumerate(frame_paths):
        out_path = os.path.join(swapped_dir, f"swapped_{idx:05d}.jpg")
        try:
            swapped = swapper.swap_faces(src_path, int(src_idx), frame_path, int(dst_idx))
            cv2.imwrite(out_path, swapped)
        except Exception:
            cv2.imwrite(out_path, cv2.imread(frame_path))
    # Combine frames to video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames_to_video(swapped_dir, output_video_path, fps)
    return output_video_path

def swap_multi_src_single_dst(src_imgs, dst_img, dst_idx):
    # src_imgs: list of images
    results = []
    dst_path = "tmp_dst.jpg"
    cv2.imwrite(dst_path, dst_img)
    for i, src_img in enumerate(src_imgs):
        src_path = f"tmp_src_{i}.jpg"
        cv2.imwrite(src_path, src_img)
        try:
            result = swapper.swap_faces(src_path, 1, dst_path, int(dst_idx))
            results.append(result)
        except Exception as e:
            results.append(f"Error: {e}")
    return results

def swap_multi_src_multi_dst(src_imgs, dst_imgs, dst_indices):
    # src_imgs, dst_imgs: lists of images; dst_indices: list of indices
    results = []
    for i, src_img in enumerate(src_imgs):
        src_path = f"tmp_src_{i}.jpg"
        cv2.imwrite(src_path, src_img)
        for j, dst_img in enumerate(dst_imgs):
            dst_path = f"tmp_dst_{j}.jpg"
            cv2.imwrite(dst_path, dst_img)
            try:
                result = swapper.swap_faces(src_path, 1, dst_path, int(dst_indices[j]))
                results.append(result)
            except Exception as e:
                results.append(f"Error: {e}")
    return results

with gr.Blocks() as demo:
    gr.Markdown(wellcomingMessage)
    with gr.Tab("Single Photo Swapping"):
        gr.Interface(
            fn=swap_single_photo,
            inputs=[
                gr.Image(label="Source Image"),
                gr.Number(value=1, label="Source Face Index"),
                gr.Image(label="Destination Image"),
                gr.Number(value=1, label="Destination Face Index"),
            ],
            outputs=gr.Image(label="Swapped Image"),
        )
    with gr.Tab("Video Swapping"):
        gr.Interface(
            fn=swap_video,
            inputs=[
                gr.Image(label="Source Image"),
                gr.Number(value=1, label="Source Face Index"),
                gr.Video(label="Target Video"),
                gr.Number(value=1, label="Destination Face Index"),
            ],
            outputs=gr.Video(label="Swapped Video"),
        )
    with gr.Tab("MultiSrc SingleDst"):
        gr.Interface(
            fn=swap_multi_src_single_dst,
            inputs=[
                gr.Gallery(label="Source Images", type="numpy", columns=3),
                gr.Image(label="Destination Image"),
                gr.Number(value=1, label="Destination Face Index"),
            ],
            outputs=gr.Gallery(label="Swapped Images"),
        )
    with gr.Tab("MultiSrc MultiDst"):
        gr.Interface(
            fn=swap_multi_src_multi_dst,
            inputs=[
                gr.Gallery(label="Source Images", type="numpy", columns=3),
                gr.Gallery(label="Destination Images", type="numpy", columns=3),
                gr.Textbox(label="Destination Face Indices (comma-separated, e.g. 1,1,2)"),
            ],
            outputs=gr.Gallery(label="Swapped Images"),
        )

if __name__ == "__main__":
    demo.launch()
