import cv2
import numpy as np

#ビーム動画の合成
def gerobi(kuromaga, gerobiga, x, y, width, height):
    point_x = x - 280
    point_y = y - (height/2)
    orgHeight,orgWidth = kuromaga.shape[:2]
    size = (orgWidth,orgHeight)
    gerobiga = cv2.resize(gerobiga,size)
    sizes = tuple(np.array([gerobiga.shape[1], gerobiga.shape[0]]))
    rad = 0
    move_x = point_x
    move_y = point_y
    matrix = [
              [np.cos(rad),  -1 * np.sin(rad), move_x],
              [np.sin(rad),   np.cos(rad), move_y]
          ]
    affine_matrix = np.float32(matrix)
    gerobiga = cv2.warpAffine(gerobiga, affine_matrix, sizes, flags=cv2.INTER_LINEAR)
    kamehameha = cv2.bitwise_or(kuromaga,gerobiga,gerobiga)

    return kamehameha
