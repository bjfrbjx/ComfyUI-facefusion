# FaceFuison extension for ComfyUI

[FaceFusion](https://github.com/facefusion/facefusion) is a very nice face swapper and enhancer.

# install
将项目复制到'~/ComfyUI/custom_nodes'下，进入目录执行`pip install -r requirments.txt`

# download models
模型会在执行时自动下载，如有需要可以提前下载到项目文件夹下的`.assets/models/`路径下 【可以找代理网站加速，如https://gitproxy.click/】   
```
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/dfl_xseg.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fairface.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fan_68_5.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/kim_vocal_2.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.onnx
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.onnx

https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/2dfan4.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/arcface_w600k_r50.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/dfl_xseg.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fairface.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/fan_68_5.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/kim_vocal_2.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/bisenet_resnet_34.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/retinaface_10g.hash
https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/scrfd_2.5g.hash
```
![models.png](.github/models.png)
# play
![video.png](.github/video.png)
single_source_image: 换脸来源   
video和video_url：都是加载视频，一个是本地上传，一个是url加载    
reference_face_image：脸部追踪，当启用face_selector_mode=reference时，在多人视频中跟踪相似的脸进行替换   
face_enhance_blend:gfpgan1.4修正脸部的强度，0~100 ，0就是不启用修正   
thread_count: 视频帧换脸时的并发线程数   

![image.png](.github/image.png)
single_source_image: 换脸来源   
image：换脸目标   
reference_face_image：脸部追踪，当启用face_selector_mode=reference时，在多人图片中跟踪相似的脸进行替换   
face_enhance_blend:gfpgan1.4修正脸部的强度，0~100 ，0就是不启用修正  

## fix
1 yoloface 有时识别不出来人脸，导致闪帧，改用retinaface或者scrfd
![error_yoloface.png](.github/error_yoloface.png)