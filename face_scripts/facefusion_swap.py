# coding=utf-8

import gradio as gr
from PIL import Image
from modules import scripts, images, scripts_postprocessing
from modules.processing import (
    StableDiffusionProcessing,
)

import face_scripts.facefusion_logging as logger
from face_scripts.fusion_swapper import swap_face
from face_scripts.facefusion_utils import get_timestamp
import facefusion.metadata as ff_metadata

print(
    f"[-] FaceFusion initialized. version: {ff_metadata.get('version')}"
)


class FaceFusionScript(scripts.Script):

    def process(
        self,
        p: StableDiffusionProcessing,
        img,
        device,
        face_detector_score,
        mask_blur,
        imgs,
        landmarker_score,
        face_enhance_blend:float,
        thread_count:int
    ):
        self.source = img
        self.face_enhance_blend=face_enhance_blend
        self.device = device
        self.face_detector_score = face_detector_score
        self.mask_blur = mask_blur
        self.source_imgs = imgs
        self.landmarker_score = landmarker_score
        self.thread_count=thread_count
        if self.source is None:
            logger.error(f"Please provide a source face")
        else:
            for i in range(len(p.init_images)):
                image = p.init_images[i]
                args = scripts.PostprocessImageArgs(image)
                self.postprocess_image(p, args)
                p.init_images[i] = args.image


    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        if self.source is not None:
            st = get_timestamp()
            logger.info("FaceFusion enabled, start process")
            image: Image.Image = script_pp.image
            landmarker_score = 0.5
            if self.landmarker_score:
                landmarker_score = self.landmarker_score
            result: Image.Image = swap_face(
                source_img= self.source,
                target_img=image,
                provider=self.device,
                detector_score=self.face_detector_score,
                mask_blur=self.mask_blur,
                landmarker_score=landmarker_score,
                face_enhance_blend=self.face_enhance_blend,
                source_imgs=self.source_imgs,
                thread_count=self.thread_count,
            )
            pp = scripts_postprocessing.PostprocessedImage(result)
            pp.info = {}
            p.extra_generation_params.update(pp.info)
            script_pp.image = pp.image
            et = get_timestamp()
            cost_time = (et - st) / 1000
            logger.info(f"FaceFusion process done, time taken: {cost_time} sec.")
