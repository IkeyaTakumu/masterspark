import cv2
import numpy as np
import pdb

"""
gerobiの回転中心の座標は
center_x = 0.1453125 test_center_x/window_width * gerobi_window_width
center_y = 0.4986111 test_center_y/window_height * gerobi_window_height
それぞれの係数は倍率なので変える必要なし
"""

def gerobi(kuromaga, gerobiga, croped_window_center_x, croped_window_center_y, frame_width, frame_height, degree, scale=1.0, n = 2):
    angle = -1*degree
    frame_height,frame_width = kuromaga.shape[:2]
    window_size = (frame_width,frame_height)
    resized_gerobiga = cv2.resize(gerobiga,window_size)
    gerobi_size = tuple(np.array([frame_width, frame_height]))
    gerobi_center = tuple(np.array([0.1453125 * frame_width,0.4986111 * frame_height]))

    tx = croped_window_center_x - gerobi_center[0]
    ty = croped_window_center_y - gerobi_center[1]
    moved_matrix = np.array([[1, 0, tx],
                             [0, 1, ty]])

    rotation_matrix = cv2.getRotationMatrix2D((croped_window_center_x,croped_window_center_y), angle, scale)
    rotation_matrix = np.float32(rotation_matrix)

    moved_gerobiga = cv2.warpAffine(resized_gerobiga, moved_matrix, gerobi_size, flags=cv2.INTER_LINEAR)
    rotated_moved_gerobiga = cv2.warpAffine(moved_gerobiga, rotation_matrix, gerobi_size, flags=cv2.INTER_LINEAR)
    kamehameha = cv2.bitwise_or(kuromaga,rotated_moved_gerobiga,rotated_moved_gerobiga)
    return kamehameha
