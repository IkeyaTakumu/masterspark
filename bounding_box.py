import cv2
import numpy as np
from copy import copy

"""
手検出範囲のバウンディングボックスに関するクラス
draw_box():バウンディングボックスの描画
hand_in_box():バウンディングボックス内に手があるか,肌色の検出で判断
move_box_point():バウンディングボックスの移動（w,a,s,d）
"""
class BDBox:
    def __init__(self,
                 x = 400 ,#バウンディングボックスの右角のx座標
                 y = 400,#バウンディングボックスの右角のy座標
                 width = 200,#バウンディングボックスの幅
                 height = 200,#バウンディングボックスの高さ
                 ac = 5,#移動速さ
                 color = (255,0,0)):#バウンディングボックスの色

        if ((x - width) < 0) or ((y - height) < 0):
            x = width
            y = height

        self.flag_hand = False
        self.upperright_x = x
        self.upperright_y = y
        self.box_width = width
        self.box_height = height
        self.ac = ac
        self.pixnum = width*height
        self.color = color
        self.kernel = np.ones((15, 15), np.uint8)#クロージング処理で利用するカーネルサイズ
        self.flag_track = False

    def draw_box(self,image):#青枠の描画．ここの処理は再考の余地あり
        image_cp = copy(image)
        if not self.flag_track:
            self.upperleft_x = self.upperright_x - self.box_width
            self.upperleft_y = self.upperright_y
            self.downright_x = self.upperright_x
            self.downright_y = self.upperright_y + self.box_height
            self.center_p_x = self.upperleft_x + int(self.box_width/2)
            self.center_p_y = self.upperleft_y + int(self.box_height/2)
            upperleft = (self.upperleft_x,self.upperleft_y)
            downright = (self.downright_x,self.downright_y)
            cv2.rectangle(image_cp,(upperleft),(downright),self.color,10)
        else:
            image_cp = self.tracking(image_cp)
        return image_cp

    def hand_in_box(self, image, thresh_hand_upper, thresh_hand_lower):#手が青枠内にあるか判定
        self.image_crop = image[self.upperleft_y:self.downright_y,self.upperleft_x:self.downright_x]
        self.image_crop_hsv = cv2.cvtColor(self.image_crop, cv2.COLOR_BGR2HSV)
        self.hand_mask = cv2.inRange(self.image_crop_hsv, thresh_hand_lower, thresh_hand_upper)
        self.hand_mask = cv2.morphologyEx(self.hand_mask, cv2.MORPH_CLOSE, self.kernel)
        """
        青枠領域とそのmask画像の表示（デバッグ用）
        """
        #cv2.namedWindow("hand", cv2.WINDOW_NORMAL)
        #cv2.imshow("hand",self.image_crop)
        #cv2.namedWindow("hand_mask", cv2.WINDOW_NORMAL)
        #cv2.imshow("hand_mask",self.hand_mask)
        if int(self.hand_mask.sum()/255) > int(self.pixnum*0.2):#肌色領域の面積で手があるか判断
            self.color = (0,255,0)
            self.flag_hand = True
        else:
            self.color = (251,38,91)
            self.flag_hand = False

    def move_box_point(self,image):#w,a,s,dが押されたときに青枠の移動と追跡開始のフラグ作成
        key = cv2.waitKey(1)&0xff
        if key == ord("w"):
            if (self.upperleft_y) > 0:
                self.upperright_y -= self.ac
        if key == ord("s"):
            if (self.downright_y + self.ac) < image.shape[0]:
                self.upperright_y += self.ac
        if key == ord("a"):
            if (self.upperleft_x) > 0:
                self.upperright_x -= self.ac
        if key == ord("d"):
            if (self.downright_x) < image.shape[1]:
                self.upperright_x += self.ac
        if key == ord("e"):
            exit()
        if key == ord("t"):
            print("start_tracking")
            self.initialize_tracking()
            self.flag_track = True
            self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def initialize_tracking(self):
        #ヒストグラムの生成
        self.roi_hist = cv2.calcHist([self.image_crop_hsv],[0],self.hand_mask,[256],[0,256])
        #ノルム正規化
        cv2.normalize(self.roi_hist,self.roi_hist,0,255,cv2.NORM_MINMAX)

    def tracking(self,img):
        #imageのHSV変換
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #tをトリガーとして取得した手の色ヒストグラムを特徴量として，画像の類似度を判定
        dst = cv2.calcBackProject([hsv_img],[0],self.roi_hist,[0,180], 1)
        #物体の検出
        ret, track_window = cv2.meanShift(dst,(self.upperleft_x,self.upperleft_y,self.box_width,self.box_height), self.term_crit)
        #物体検出で取得した座標を次のboxの座標へぶちこむ
        x,y,w,h = track_window
        self.upperleft_x = x
        self.upperleft_y = y
        self.downright_x = self.upperleft_x + self.box_width
        self.downright_y = self.upperleft_y + self.box_height
        self.center_p_x = self.upperleft_x + int(self.box_width/2)
        self.center_p_y = self.upperleft_y + int(self.box_height/2)
        img_dst = cv2.rectangle(img, (x,y), (x+w, y+h), 255, 2)
        return img_dst

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
