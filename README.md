# 概要
webカメラで取得した画像上の枠内に手が入ったことを検知して，ビームを発射するプログラム．クロマキー合成で宇宙空間をフレームに合成しているためグリーンバックを用意して実行して下さい．

# 操作方法
* w,a,s,dで枠を移動
* 枠内に手がある状態でtキーを押下すると手を追跡してビームがでる
* escキーで終了

# 実行
`python main.py --window_width 1280 --window_height 720 --bdbox_xp 10 --bdbox_yp 10 --bdbox_width 200 --bdbox_height 200 --camera_id 0`

* --window_height:ウィンドウの高さを設定
* --window_width:ウィンドウの幅を設定
* --bdbox_xp:青枠の初期位置を設定
* --bdbox_yp:青枠の初期位置を設定
* --bdbox_width:青枠の幅を設定
* --bdbox_height:青枠の高さを設定
* --camera_id:webカメラのidを設定
# 環境
* python 3.5.4
* opencv 3.3.1
* numpy 1.14.0

# 実行画面
<img src = "https://user-images.githubusercontent.com/37826053/42122640-aca1b7e6-7c7f-11e8-9bcd-50193faef0e5.jpg" width="400px" height="225">

