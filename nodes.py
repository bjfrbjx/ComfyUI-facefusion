import urllib.request
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36'),('Connection', 'keep-alive')]
urllib.request.install_opener(opener)

from face_scripts.facefusion_swap import FaceFusionScript
from modules.processing import (
    StableDiffusionProcessingImg2Img
)
from .utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil


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
    CATEGORY = "FaceFusion"

    def execute(self, image, single_source_image, device, face_detector_score, mask_blur, landmarker_score,face_enhance_blend):
        pil_images = batch_tensor_to_pil(image)
        source = tensor_to_pil(single_source_image)
        script = FaceFusionScript()
        p = StableDiffusionProcessingImg2Img(pil_images)
        script.process(p=p,
                       img=source,
                       device=device,
                       face_detector_score=face_detector_score,
                       mask_blur=mask_blur,
                       imgs=None,
                       face_enhance_blend=face_enhance_blend,
                       landmarker_score=landmarker_score)
        result = batched_pil_to_tensor(p.init_images)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "WD_FaceFusion": WD_FaceFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD_FaceFusion": "WD_FaceFusion",
}
