#!/usr/bin/env python3

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GRADIO_SERVER_PORT'] = '7860'
os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'
import urllib.request
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36'),('Connection', 'keep-alive')]
urllib.request.install_opener(opener)
from facefusion import core

if __name__ == '__main__':
	core.cli()
