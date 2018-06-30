import cv2
import numpy as np
"""
back_ground:世界遺産の部分
frame:取得した動画のフレーム
thresh_upper:閾値上
thresh_lower:閾値下
戻り値：合成した画像が帰る
機能：frameから指定の色領域をマスクし，背景と合成
"""
def composition_chrom(back_ground, frame, thresh_upper, thresh_lower):
    kernel = np.ones((10, 10), np.uint8)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, thresh_lower, thresh_upper)
    #cv2.imshow('mae_frame_hsv',mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('ato_frame_hsv',mask)
    inv_mask = cv2.bitwise_not(mask)
    res1 = cv2.bitwise_and(frame,frame,mask=inv_mask)
    #cv2.imshow('res1',res1)
    res2 = cv2.bitwise_and(back_ground,back_ground,mask=mask)
    #cv2.imshow('res2',res2)
    disp = cv2.bitwise_or(res1,res2)
    #cv2.imshow('disp',disp)
    return disp
"""
lower_w:白領域閾値下
upper_w:白領域閾値上
logo:logoの画像をリサイズしnp.arrayで格納
row,cols,channel:logoの高さ，幅，チャンネル
logo_gray:logoのグレースケール化
ret,masklogo:wakaran,logoの二値化
masklogo_inv:bit反転(Nが白，背景黒)
logo_fg:logo縁とグレー部分以外を黒塗りするかっこいい！！
chrom:logoを埋め込む画像
roi:画像を埋め込む部分を切り取る
chrom_bg:切り取った画像にlogoの黒領域を確保
dst:黒領域を確保した部分にlogoを埋め込み完璧な合成を作成する
chrom:元の画像にlogoを合成した部分を埋め込む
機能：logoに関するクラス，logoの初期設定や画像左上に埋め込む処理を持つ
"""
#logoに関するクラス，今回は未使用
class logos():
    def __init__(self,logo_path,window_width,window_height):
        self.lower_w = np.array([0,0,0])
        self.upper_w = np.array([2,2,2])
        self.logo = cv2.resize(cv2.imread(logo_path),(int(window_width/10), int(window_width/10)))
        self.rows,self.cols,self.channels = self.logo.shape
        self.logo_gray = cv2.cvtColor(self.logo,cv2.COLOR_BGR2GRAY)
        self.ret,self.masklogo = cv2.threshold(self.logo_gray, 220, 255, cv2.THRESH_BINARY)
        self.masklogo_inv = cv2.bitwise_not(self.masklogo)
        self.logo_fg = cv2.bitwise_and(self.logo,self.logo,mask = self.masklogo_inv)

    def composition_logo(self,chrom):
        mask_logo = cv2.inRange(self.logo, self.lower_w, self.upper_w)
        roi = chrom[0:self.rows,0:self.cols]
        chrom_bg = cv2.bitwise_and(roi,roi,mask = self.masklogo)
        dst = cv2.bitwise_or(chrom_bg, self.logo_fg)
        chrom[0:self.rows, 0:self.cols] = dst
        return chrom
