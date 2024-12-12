import os
import sys

from facefusion.typing import AppContext


def detect_app_context() -> AppContext:
	frame = sys._getframe(1)

	res=_inner(frame)
	# from facefusion import logger
	# logger.error(f"xxxx {res}",__name__)
	return res

def _inner(frame):
	while frame:
		if os.path.join('facefusion', 'jobs') in frame.f_code.co_filename:
			return 'cli'
		if os.path.join('facefusion', 'uis') in frame.f_code.co_filename:
			return 'ui'
		frame = frame.f_back
	return 'cli'
