import cv2
import numpy as np
def cropping(frame,ix,iy,ox,oy):
    croped_frame = frame[iy:oy,ix:ox]
    croped_gray_frame = cv2.cvtColor(croped_frame,cv2.COLOR_BGR2GRAY)
    return croped_gray_frame

def cropping_color(frame,ix,iy,ox,oy):
    croped_frame = frame[iy:oy,ix:ox]
    cv2.imshow("frame_color",croped_frame)
    return croped_frame

def cropping_mask(frame,ix,iy,ox,oy,upper,lower):
    croped_frame = frame[iy:oy,ix:ox]
    croped_hsv_frame = cv2.cvtColor(croped_frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(croped_hsv_frame,lower,upper)
    new_mask_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = "uint8")
    new_mask_frame[iy:oy,ix:ox] = mask
    return new_mask_frame

def dst_cropping(frame,ix,iy,ox,oy):
    croped_frame = frame[iy:oy,ix:ox]
    new_mask_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = "uint8")
    new_mask_frame[iy:oy,ix:ox] = croped_frame
    return new_mask_frame
