#coding:utf-8
import cv2
import numpy as np
from bounding_box import BDBox
from generate_chrom import composition_chrom
from params import *
from gerobi_kansu import gerobi
import serial
from opt import parse_opts

#緑：Hue : 60	Saturation : 100	Brightness : 71
lower_green = np.array([20, 50, 40])
upper_green = np.array([90, 255, 255])
lower_w = np.array([0,0,0])
upper_w = np.array([2,2,2])
lower_hand = np.array([0,10,40])
upper_hand = np.array([50,150,255])

#入力>255であれば255を返し，0 <= 入力 <= 255であれば入力をそのまま返し，入力 < 0であれば0を返す
def lamp(x):
    if x > 255:
        return 255
    elif (x >= 0) and (x <= 255):
        return x
    elif x < 0:
        return 0

"""
左クリック時：GBの閾値を変える
右クリック時：肌色の閾値を変える
どちらもカーソルのHSVの上下30ほど
"""
#マウス操作があった時の処理
def EstHsvThreash(event,x,y,flags,params):
    global upper_green
    global lower_green
    global upper_hand
    global lower_hand
    img_hsv = cv2.cvtColor(disp,cv2.COLOR_BGR2HSV)
    #左クリックされたとき
    if event == cv2.EVENT_LBUTTONDOWN:
        pixVal_hsv = img_hsv[y][x]
        upper_H = lamp(pixVal_hsv[0] + 30)
        lower_H = lamp(pixVal_hsv[0] - 30)
        upper_green = np.array([upper_H,255,255])
        lower_green = np.array([lower_H,90,30])
    #右クリックされたとき
    if event == cv2.EVENT_RBUTTONDOWN:
        pixVal_hand_hsv = img_hsv[y][x]
        upper_H = lamp(pixVal_hand_hsv[0] + 30)
        lower_H = lamp(pixVal_hand_hsv[0] - 30)
        upper_S = lamp(pixVal_hand_hsv[1] + 50)
        lower_S = lamp(pixVal_hand_hsv[1] - 50)
        upper_V = lamp(pixVal_hand_hsv[2] + 50)
        lower_V = lamp(pixVal_hand_hsv[2] - 50)
        upper_hand = np.array([upper_H,upper_S,upper_V])
        lower_hand = np.array([lower_H,lower_S,lower_V])
        print("upper_hand:{}".format(upper_hand))
        print("lower_hand:{}".format(lower_hand))

if __name__ == '__main__':
    opt = parse_opts()
    #ser = serial.Serial('COM3',9600,timeout=None)
    #ser.write(bytes("q","utf-8"))
    cap = cv2.VideoCapture(opt.camera_id)#webカメラ用のオブジェクト作成
    masterspark = cv2.VideoCapture(beam_path)
    end_flag, c_frame = cap.read()#初期フレームの読込
    gerobi_flag, gerobi_frame = masterspark.read()#ゲロビフレーム読み込み
    height, width, channels = c_frame.shape#フレームの高さ，幅，チャンネル取得
    back = cv2.imread(back_ground_path)#背景とロゴの読み込み
    back = cv2.resize(back,(opt.window_width,opt.window_height))#背景を指定サイズにリサイズ
    back_hsv = cv2.cvtColor(back, cv2.COLOR_BGR2HSV)#背景をHSV変換
    bdbox = BDBox(x = opt.bdbox_xp,y = opt.bdbox_yp,width = opt.bdbox_width,height = opt.bdbox_height,ac = 20,track_mode = opt.track_mode)#BDBのオブジェクト生成
    bdbox.mp3_load()
    cv2.namedWindow("gousei")
    cv2.setMouseCallback("gousei", EstHsvThreash)

    while end_flag == True:
        img = cv2.resize(c_frame,(opt.window_width,opt.window_height))#キャプチャをリサイズ
        disp = composition_chrom(back, img, upper_green, lower_green)#キャプチャと背景をクロマキー合成
        disp_with_box = bdbox.draw_box(disp,img)#BoundingBoxの合成
        bdbox.move_box_point(disp_with_box)#boxを動かす
        bdbox.hand_in_box(img, upper_green, lower_green)#box内に手があるか
        if bdbox.flag_hand:#box内に手がある場合実行
            if gerobi_flag:#ビーム動画が再生されているとき
                disp_with_box = gerobi(disp_with_box, gerobi_frame, bdbox.center_p_x, bdbox.center_p_y,opt.window_width,opt.window_height,bdbox.box_degree)
                #ser.write(bytes("s","utf-8"))
                gerobi_flag, gerobi_frame = masterspark.read()#ゲロビフレーム読み込み
            elif not gerobi_flag:#ビーム動画が全フレーム再生されたとき
                masterspark.release()#ビーム動画のオブジェクトを開放
                masterspark = cv2.VideoCapture(beam_path)#再び動画のオブジェクトを生成
                gerobi_flag, gerobi_frame = masterspark.read()#ゲロビフレーム読み込み
        #else:
            #ser.write(bytes("q","utf-8"))
        cv2.imshow("gousei", disp_with_box)#フレーム表示処理
        key = cv2.waitKey(INTERVAL)#一定時間キー入力を待つ
        if key == ESC_KEY:#Escキーが押されたら終了
            break
        if key == ord("c"):#cキーが押されたらキャプチャ
            cv2.imwrite("./capture/caputure.jpg",disp_with_box)
            print("captureしました")
        end_flag, c_frame = cap.read()#フレームの更新

    cv2.destroyAllWindows()
    cap.release()
    #ser.close()
