import cv2
import numpy as np
from copy import copy
from crop_hand import dst_cropping,cropping
import math
import time
import threading
from mutagen.mp3 import MP3 as mp3
import pygame

"""
手検出範囲のバウンディングボックスに関するクラス
draw_box():バウンディングボックスの描画
hand_in_box():バウンディングボックス内に手があるか,肌色の検出で判断
move_box_point():バウンディングボックスの移動（w,a,s,d）
"""
class BDBox:
    def __init__(self,
                 x = 100 ,#バウンディングボックスの左角のx座標
                 y = 100,#バウンディングボックスの左角のy座標
                 width = 200,#バウンディングボックスの幅
                 height = 200,#バウンディングボックスの高さ
                 ac = 5,#移動速さ
                 color = (255,0,0),
                 track_mode = "cam"):#バウンディングボックスの色

        if ((x - width) < 0) or ((y - height) < 0):
            x = width
            y = height

        self.track_mode = track_mode
        self.point_num = 500
        self.flag_hand = False
        self.pre_flag_hand = False
        self.start_time = 0
        self.upperleft_x = x
        self.upperleft_y = y
        self.box_width = width
        self.box_height = height
        self.box_degree = 0
        self.ac = ac
        self.pixnum = width*height
        self.color = color
        self.kernel = np.ones((15, 15), np.uint8)#クロージング処理で利用するカーネルサイズ
        self.criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,20,0.03)#終了条件
        self.over_x = False
        self.over_y = False
        self.flag_track = False

    def draw_box(self,disp,frame):#青枠の描画．ここの処理は再考の余地あり
        disp_cp = copy(disp)
        self.frame = frame
        if not self.flag_track:
            self.upperright_x = self.upperleft_x + self.box_width
            self.upperright_y = self.upperleft_y
            self.downright_x = self.upperright_x
            self.downright_y = self.upperright_y + self.box_height
            self.center_p_x = self.upperleft_x + int(self.box_width/2)
            self.center_p_y = self.upperleft_y + int(self.box_height/2)
            upperleft = (self.upperleft_x,self.upperleft_y)
            downright = (self.downright_x,self.downright_y)
            cv2.rectangle(disp_cp,(upperleft),(downright),self.color,10)
        else:
            disp_cp = self.tracking(disp_cp)
        return disp_cp

    def hand_in_box(self, image, thresh_hand_upper, thresh_hand_lower):#手が青枠内にあるか判定
        #print("左上：({},{})　右下：({},{})".format(self.upperleft_x,self.upperleft_y,self.downright_x,self.downright_y))
        self.image_crop = image[self.upperleft_y:self.downright_y,self.upperleft_x:self.downright_x]
        self.image_crop_hsv = cv2.cvtColor(self.image_crop, cv2.COLOR_BGR2HSV)
        self.hand_mask = cv2.bitwise_not(cv2.inRange(self.image_crop_hsv, thresh_hand_lower, thresh_hand_upper))
        self.hand_mask = cv2.morphologyEx(self.hand_mask, cv2.MORPH_OPEN, self.kernel)
        """
        青枠領域とそのmask画像の表示（デバッグ用）
        """
        if int(self.hand_mask.sum()/255) > int(self.pixnum*0.2):#肌色領域の面積で手があるか判断
            self.color = (0,255,0)
            self.flag_hand = True
        else:
            pygame.mixer.music.stop()
            self.color = (251,38,91)
            self.flag_hand = False
            self.start_time = 0
        if (not self.pre_flag_hand) and self.flag_hand:
            self.start_time = time.time()
            pygame.mixer.music.play(-1)
        self.pre_flag_hand = self.flag_hand

    def move_box_point(self,image):#w,a,s,dが押されたときに青枠の移動と追跡開始のフラグ作成
        key = cv2.waitKey(1)&0xff
        if key == ord("w"):
            if (self.upperleft_y) > 0:
                self.upperleft_y -= self.ac
        if key == ord("s"):
            if (self.downright_y + self.ac) < image.shape[0]:
                self.upperleft_y += self.ac
        if key == ord("a"):
            if (self.upperleft_x) > 0:
                self.upperleft_x -= self.ac
        if key == ord("d"):
            if (self.downright_x) < image.shape[1]:
                self.upperleft_x += self.ac
        if key == ord("e"):
            exit()
        if key == ord("k"):
            self.start_time = 0
            self.flag_track = False

        if (time.time() - self.start_time)>2 and (self.start_time != 0) and self.flag_track == False:
            print("start_tracking")
            self.initialize_tracking()
            self.flag_track = True
            self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def initialize_tracking(self):
        #ヒストグラムの生成
        self.roi_hist = cv2.calcHist([self.image_crop_hsv],[0],self.hand_mask,[256],[0,256])
        #ノルム正規化
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)
        if self.track_mode == "lk":
            self.pre_frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
            self.pre_croped_frame = cropping(self.frame,self.upperleft_x,self.upperleft_y,self.upperleft_x + self.box_width,self.upperleft_y + self.box_height)

    def tracking(self,img):
        if self.track_mode == "cam":
            #imageのHSV変換
            hsv_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            #tをトリガーとして取得した手の色ヒストグラムを特徴量として，画像の類似度を判定
            dst = cv2.calcBackProject([hsv_img],[0],self.roi_hist,[0,180], 1)
            dst = dst_cropping(dst,self.upperleft_x,self.upperleft_y,self.upperleft_x + self.box_width,self.upperleft_y + self.box_height)
            ret_mean, track_window = cv2.meanShift(dst,(self.upperleft_x,self.upperleft_y,self.box_width,self.box_height),self.term_crit)
            #物体の検出
            ret_cam, _ = cv2.CamShift(dst,(self.upperleft_x,self.upperleft_y,self.box_width,self.box_height), self.term_crit)
            #座標取得
            pts = cv2.boxPoints(ret_cam)
            pts = np.int0(pts)
            #バウンディングボックスの傾きを計算
            radian = math.atan2([pts][0][1][1] - [pts][0][0][1] , [pts][0][1][0] - [pts][0][0][0])
            self.box_degree = radian *(180/np.pi)
            #物体検出で取得した座標を次のboxの座標へぶちこむ
            self.upperleft_x = track_window[0]
            self.upperleft_y = track_window[1]
            self.downright_x = self.upperleft_x + self.box_width
            self.downright_y = self.upperleft_y + self.box_height
            self.center_p_x = int((pts[1][0] + pts[3][0])/2)
            self.center_p_y = int((pts[3][1] + pts[1][1])/2)
            img_dst = cv2.rectangle(img, (track_window[0],track_window[1]), (track_window[0]+track_window[2], track_window[1]+track_window[3]), self.color, 10)
        elif self.track_mode == "lk":
            #imageのHSV変換
            hsv_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            #tをトリガーとして取得した手の色ヒストグラムを特徴量として，画像の類似度を判定
            dst = cv2.calcBackProject([hsv_img],[0],self.roi_hist,[0,180], 1)
            dst = dst_cropping(dst,self.upperleft_x,self.upperleft_y,self.upperleft_x + self.box_width,self.upperleft_y + self.box_height)
            #物体の検出
            ret_cam, _ = cv2.CamShift(dst,(self.upperleft_x,self.upperleft_y,self.box_width,self.box_height), self.term_crit)
            #座標取得
            pts = cv2.boxPoints(ret_cam)
            pts = np.int0(pts)
            #バウンディングボックスの傾きを計算
            radian = math.atan2([pts][0][1][1] - [pts][0][0][1] , [pts][0][1][0] - [pts][0][0][0])
            self.box_degree = radian *(180/np.pi)

            gray_frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
            now_croped_frame = cropping(self.frame,self.upperleft_x,self.upperleft_y,self.upperleft_x + self.box_width,self.upperleft_y + self.box_height) #現在フレームをクロップ
            self.pre_croped_feature = cv2.goodFeaturesToTrack(self.pre_croped_frame,self.point_num,0.1,5) #過去フレームをクロップ

            for i in range(self.pre_croped_feature.shape[0]):
                if i == 0:
                    feature_x = self.pre_croped_feature[i][0][0]
                    feature_y = self.pre_croped_feature[i][0][1]
                    self.pre_feature = np.array([[[feature_x + self.upperleft_x,feature_y + self.upperleft_y]]], dtype = "float32")
                else:
                    feature_x = self.pre_croped_feature[i][0][0]
                    feature_y = self.pre_croped_feature[i][0][1]
                    self.pre_feature = np.concatenate([self.pre_feature,np.array([[[feature_x + self.upperleft_x,feature_y + self.upperleft_y]]], dtype = "float32")],axis=0)

            now_feature, status, err = cv2.calcOpticalFlowPyrLK(self.pre_frame, gray_frame, self.pre_feature, nextPts = None, winSize = (10, 10), maxLevel = 4, criteria = self.criteria, flags = 0)
            for i in range(now_feature.shape[0]):
                pre_x = self.pre_feature[i][0][0]
                pre_y = self.pre_feature[i][0][1]
                now_x = now_feature[i][0][0]
                now_y = now_feature[i][0][1]
                dx = now_x - pre_x
                dy = now_y - pre_y
                if i == 0:
                    all_dx = np.array(dx)
                    all_dy = np.array(dy)
                else:
                    all_dx = np.append(all_dx,dx)
                    all_dy = np.append(all_dy,dy)

                img_dst = cv2.circle(img,(int(now_x),int(now_y)),2,(0,0,255),-1)

            median_dx = int(np.median(all_dx))
            median_dy = int(np.median(all_dy))

            if 0 < self.upperleft_x + median_dx:
                self.upperleft_x += median_dx
            else:
                self.upperleft_x = 0
            if 0 < self.upperleft_y + median_dy:
                self.upperleft_y += median_dy
            else:
                self.upperleft_y = 0
            if img.shape[1] > self.upperleft_x + self.box_width + median_dx:
                self.downright_x += median_dx
                if self.over_x:
                    self.downright_x = img.shape[1] - 1
                    self.over_x = False
                elif (self.downright_x - self.upperleft_x) < self.box_width:
                    self.downright_x = self.upperleft_x + self.box_width
            else:
                self.downright_x = img.shape[1] - 1
                self.over_x = True

            if img.shape[0] > self.upperleft_y + self.box_height + median_dy:
                self.downright_y += median_dy
                if self.over_y:
                    self.downright_y = img.shape[0] - 1
                    self.over_y = False
                elif (self.downright_y - self.upperleft_y) < self.box_height:
                    self.downright_y = self.upperleft_y + self.box_height
            else:
                self.downright_y = img.shape[0] - 1
                self.over_y = True

            self.pre_croped_frame = now_croped_frame.copy()
            self.pre_frame = gray_frame.copy()
            self.center_p_x = self.upperleft_x + int(self.box_width/2)
            self.center_p_y = self.upperleft_y + int(self.box_height/2)
            upperleft = (self.upperleft_x,self.upperleft_y)
            downright = (self.downright_x,self.downright_y)
            cv2.rectangle(img_dst,(upperleft),(downright),self.color,10)
        return img_dst

    def mp3_load(self):
        filename = "./data/master_sound_cut.wav"
        pygame.mixer.init()
        pygame.mixer.music.load(filename)

"""
usage

back = cv2.imread("D:\sekkeiseisaku\sekai.png")
bdbox = BDBox(x = 100,y = 500,width = 100,height = 100,ac = 10)
while True:
    cp_back = copy(back)
    bdbox.move_box_point(cp_back)
    bdbox_back = bdbox.draw_box(cp_back)
    cv2.imshow("back_ground",bdbox_back)
cv2.destroyAllWindows()
"""
