"ATR585 ToupTek Astro camera - live fedd viewer"
"Controls exposure and gain"

import sys, os, ctypes, threading
import numpy as np
import cv2

Script_dir = os.path.dirname(os.path.abspath(__file__))
if Script_dir not in sys.path:
    sys.path.insert(0, Script_dir)

import toupcam

frame_ready = threading.Event()
frame_lock = threading.Lock()
latest_frame = None
cam_width = cam_height = 0


def on(event, cam):
    global latest_frame, cam_width, cam_height
    if event == toupcam.TOUPCAM_EVENT_IMAGE:
        buf = (ctypes.c_ushort * (cam_width * cam_height))()
        info = toupcam.ToupcamFrameInfoV2()
        try:
            cam.PullImageV3(buf, 0, 16, -1, info)
        except toupcam.HRESULTException:
            return
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(cam_height, cam_width)
        arr8 = (arr >> 4).astype(np.uint8)
        with frame_lock:
            latest_frame = arr8
        frame_ready.set()
    elif event == toupcam.TOUPCAM_EVENT_DISCONNECTED:
        print("[camera] Disconnected!")
        frame_ready.set()