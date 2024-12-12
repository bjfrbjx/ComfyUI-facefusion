import mimetypes
import urllib.request

from PIL import Image

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent',
                      'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36'),
                     ('Connection', 'keep-alive')]
urllib.request.install_opener(opener)

from .utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil
import tempfile, requests, uuid, os
try:
    import folder_paths
except:
    folder_paths=None

# =================================
def get_mime_type(file_path):
    # 获取文件的 MIME 类型
    mime_type, _ = mimetypes.guess_type(file_path)

    # 如果无法猜测类型，返回默认类型
    if mime_type is None:
        return 'application/octet-stream'

    return mime_type

def facefusion_run(source_path, target_path: str, output_path, provider, detector_score=0.6, mask_blur=0.3,
                   face_enhance_blend=0., landmarker_score=0.5, thread_count=1):
    from facefusion.vision import detect_image_resolution, pack_resolution, detect_video_resolution, detect_video_fps
    from facefusion.filesystem import is_video, is_image
    from facefusion import state_manager
    from facefusion.core import conditional_process
    the_processors = ['face_swapper', ]
    if face_enhance_blend > 0.:
        the_processors.append('face_enhancer')
    apply_state_item = state_manager.set_item
    apply_state_item('processors', the_processors)
    apply_state_item('face_detector_angles', [0])
    #apply_state_item('command', 'headless-run')

    # ===

    apply_state_item('execution_thread_count', thread_count, )
    apply_state_item('face_enhancer_blend', face_enhance_blend)
    apply_state_item('source_paths', source_path)
    apply_state_item('target_path', target_path)
    apply_state_item('output_path', output_path)
    apply_state_item('execution_providers', provider)
    apply_state_item('face_detector_score', detector_score)
    apply_state_item('face_mask_blur', mask_blur)
    apply_state_item('face_landmarker_score', landmarker_score)
    apply_state_item('face_detector_model', 'yoloface', )
    apply_state_item('face_detector_size', '640x640', )
    apply_state_item('face_landmarker_model', '2dfan4', )
    apply_state_item('face_selector_mode', 'reference', )
    apply_state_item('face_selector_order', 'large-small', )
    apply_state_item('reference_face_position', 0, )
    apply_state_item('reference_face_distance', 0.6, )
    apply_state_item('reference_frame_number', 0, )
    apply_state_item('face_mask_types', ['box'], )
    apply_state_item('face_mask_blur', 0.3, )
    apply_state_item('face_mask_padding', (0, 0, 0, 0), )
    apply_state_item('temp_frame_format', 'png', )
    apply_state_item('output_image_quality', 80, )
    apply_state_item('output_audio_encoder', 'aac', )
    apply_state_item('output_video_encoder', 'libx264', )
    apply_state_item('output_video_preset', 'veryfast', )
    apply_state_item('output_video_quality', 80, )
    apply_state_item('age_modifier_model', 'styleganex_age', )
    apply_state_item('age_modifier_direction', 0, )
    apply_state_item('expression_restorer_model', 'live_portrait', )
    apply_state_item('expression_restorer_factor', 80, )
    apply_state_item('face_editor_model', 'live_portrait', )
    apply_state_item('face_enhancer_model', 'gfpgan_1.4', )
    apply_state_item('face_swapper_model', 'inswapper_128_fp16', )
    apply_state_item('face_swapper_pixel_boost', '128x128', )
    apply_state_item('frame_enhancer_model', 'span_kendata_x4', )
    apply_state_item('frame_enhancer_blend', 80, )
    apply_state_item('open_browser', False, )
    apply_state_item('execution_queue_count', 1, )
    apply_state_item('video_memory_strategy', None, )
    # apply_state_item('execution_device_id', '0')
    if is_image(image_path=target_path):
        image_resolution = detect_image_resolution(target_path)
        apply_state_item('output_image_resolution', pack_resolution(image_resolution))
    elif is_video(video_path=target_path):
        video_resolution = detect_video_resolution(target_path)
        apply_state_item('output_video_resolution', pack_resolution(video_resolution))
        apply_state_item('output_video_fps', int(detect_video_fps(target_path)))
    conditional_process()


class WD_FaceFusion:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "single_source_image": ("IMAGE",),  # Single source image
                "device": (["cpu", "cuda"], {"default": "cuda"}),  # Execution provider
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

    def execute(self, image, single_source_image, device, face_detector_score, mask_blur, landmarker_score,
                face_enhance_blend):
        source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        tensor_to_pil(single_source_image).save(source_path)
        source_paths = [source_path]

        target_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        batch_tensor_to_pil(image)[0].save(target_path)

        output_dir = folder_paths.get_output_directory()
        full_output_folder, _, _, _, _, = folder_paths.get_save_image_path("WD_", output_dir)
        file = f"{uuid.uuid4()}.{target_path.split('.')[-1]}"
        output_path = os.path.join(full_output_folder, file)

        facefusion_run(source_paths,
            target_path,
            output_path,
            provider=[device],
            detector_score=face_detector_score,
            mask_blur=mask_blur,
            face_enhance_blend=face_enhance_blend,
            landmarker_score=landmarker_score,
            thread_count=1)
        result = batched_pil_to_tensor([Image.open(output_path)])
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
                "thread_count": ("INT", {"default": 4, "min": 1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("SCENE_VIDEO",)
    RETURN_NAMES = ("scenes_video",)
    FUNCTION = "execute"
    CATEGORY = "WDTRIP"

    def execute(self, video_url, single_source_image, device, face_detector_score, mask_blur, landmarker_score,
                face_enhance_blend, thread_count):
        # Download the video to a temporary file

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            response = requests.get(video_url.strip(), stream=True)
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            target_path = temp_file.name
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, _, subfolder, _, = folder_paths.get_save_image_path("WD_", output_dir)
        file = f"{uuid.uuid4()}.{target_path.split('.')[-1]}"
        output_path = os.path.join(full_output_folder, file)
        source_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        tensor_to_pil(single_source_image).save(source_path)
        source_paths = [source_path]
        facefusion_run(source_paths,
            target_path,
            output_path,
            provider=[device],
            detector_score=face_detector_score,
            mask_blur=mask_blur,
            face_enhance_blend=face_enhance_blend,
            landmarker_score=landmarker_score,
            thread_count=thread_count)
        previews = [
            {
                "filename": file,
                "subfolder": "",
                "type": "output",
                "format": get_mime_type(output_path),
            }
        ]
        import logging
        logging.error(f"{previews},{output_path}")
        return {"ui": {"gifs": previews}, "result": (output_path,)}


NODE_CLASS_MAPPINGS = {
    "WD_FaceFusion": WD_FaceFusion,
    "WD_FaceFusion_Video": WD_FaceFusion_Video,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_FaceFusion": "WD_FaceFusion",
    "WD_FaceFusion_Video": "WD_FaceFusion_Video",
}
