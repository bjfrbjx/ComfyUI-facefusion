# coding=utf-8

import enum
import tempfile
from typing import List, Optional

from facefusion.core import run


class DeviceProvider(enum.Enum):
    CPU = "cpu"
    GPU = "cuda"


def swap_face(
        source_paths: [List[str]],
        target_path: str,
        output_path: Optional[str] = None,
        provider: DeviceProvider = DeviceProvider.CPU,
        detector_score: float = 0.65,
        mask_blur: float = 0.7,
        face_enhance_blend: bool = True,
        landmarker_score: float = 0.5
) -> Optional[str]:
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    result = run(
        source_path=source_paths, target_path=target_path, output_path=output_path,
        provider=provider.value, detector_score=detector_score, mask_blur=mask_blur, face_enhance_blend=face_enhance_blend,
        landmarker_score=landmarker_score
    )
    return result
