import urllib.request

from PIL import Image

from facefusion.filesystem import is_video

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36'),('Connection', 'keep-alive')]
urllib.request.install_opener(opener)

from .utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil
from facefusion.core import run
import tempfile, requests, uuid, folder_paths, os
class WD_FaceFusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "single_source_image": ("IMAGE",),  # Single source image
                "device": (["cpu", "cuda"], {"default": "cpu"}),  # Execution provider
                "face_detector_score": ("FLOAT", {"default": 0.65, "min": 0, "max": 1, "step": 0.02}),
                # Face detector score
                "mask_blur": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),  # Face mask blur
                "landmarker_score": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                # Face landmarker score
                "face_enhance_blend": ("FLOAT", {"default": 30, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "WDTRIP"

    def execute(self, image, single_source_image, device, face_detector_score, mask_blur, landmarker_score,face_enhance_blend):
        source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        tensor_to_pil(single_source_image).save(source_path)
        source_paths=[source_path]

        target_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        batch_tensor_to_pil(image)[0].save(target_path)

        output_dir = folder_paths.get_output_directory()
        full_output_folder,filename,_,subfolder,_, = folder_paths.get_save_image_path("WD_", output_dir)
        file = f"{uuid.uuid4()}.{target_path.split('.')[-1]}"
        output_path = os.path.join(full_output_folder, file)

        run(source_paths,
            target_path,
            output_path,
            provider=[device],
            detector_score=face_detector_score,
            mask_blur=mask_blur,
            face_enhance_blend=face_enhance_blend,
            landmarker_score=landmarker_score,
            thread_count=1)
        result=batched_pil_to_tensor([Image.open(output_path)])
        return (result,)
class WD_FaceFusion_Video:
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "single_source_image": ("IMAGE",),  # Single source image
                "device": (["cpu", "cuda"], {"default": "cuda"}),  # Execution provider
                "video_url": ("STRING", {
                    "default": "https://wdduoduo-videos.oss-cn-hangzhou.aliyuncs.com/test/None/painter/pose7_20241114183403.mp4",
                    "defaultBehavior": "input"
                }),
                "face_detector_score": ("FLOAT", {"default": 0.65, "min": 0, "max": 1, "step": 0.02}),
                # Face detector score
                "mask_blur": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.05}),  # Face mask blur
                "landmarker_score": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                # Face landmarker score
                "face_enhance_blend": ("FLOAT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "thread_count":("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("SCENE_VIDEO","STRING")
    RETURN_NAMES = ("scenes_video","file_path")
    FUNCTION = "execute"
    CATEGORY = "WDTRIP"

    def execute(self, video_url, single_source_image, device, face_detector_score, mask_blur, landmarker_score,face_enhance_blend,thread_count):
        # Download the video to a temporary file

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            response = requests.get(video_url.strip(), stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            target_path = temp_file.name
        output_dir = folder_paths.get_output_directory()
        full_output_folder,filename,_,subfolder,_, = folder_paths.get_save_image_path("WD_", output_dir)
        file = f"{uuid.uuid4()}.{target_path.split('.')[-1]}"
        output_path = os.path.join(full_output_folder, file)
        source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        tensor_to_pil(single_source_image).save(source_path)
        source_paths=[source_path]
        from facefusion.core import run
        run(source_paths,
            target_path,
            output_path,
            provider=[device],
            detector_score=face_detector_score,
            mask_blur=mask_blur,
            face_enhance_blend=face_enhance_blend,
            landmarker_score=landmarker_score,
            thread_count=thread_count)

        return {"ui": {"unfinished_batch": [True]},"result":(output_path,output_path)}



NODE_CLASS_MAPPINGS = {
    "WD_FaceFusion": WD_FaceFusion,
    "WD_FaceFusion_Video": WD_FaceFusion_Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_FaceFusion": "WD_FaceFusion",
    "WD_FaceFusion_Video": "WD_FaceFusion_Video",
}
