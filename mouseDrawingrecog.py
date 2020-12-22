'''
* 실행 순서
's' 입력
-> 입력할 숫자를 키보드로 입력
-> 마우스로 검은 창에 숫자 입력
-> 'space bar' 입력
-> 입력창에서 예측 결과 확인
-> 'c'를 입력해 입력화면 초기화 및 다시 마우스로 숫자 입력 후 'space bar' 입력을 원하는 만큼 반복
'''
import cv2
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

mode   = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE
x_img = np.zeros(shape=(28, 28), dtype=np.uint8)
x = np.zeros(shape=(28, 28), dtype=np.uint8)
temp = np.zeros(shape=(20, 20), dtype=np.uint8)

## 오늘 날짜로 폴더 생성
import time
def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d"%(now.tm_year, now.tm_mon, now.tm_mday)
    return s

def make_folder(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

##신경망
def init_network():
    with open("my_pickle.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
        for layer in network.layers.values():
            x = layer.forward(x)
        
        return x
    
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON: ##왼쪽 마우스를 누른채로 움직일 때
            cv2.circle(dst, (x, y), 12, (255, 255, 255), -1) ##dst에 (x,y) 위치에 원을 흰색으로 채워
    elif event == cv2.EVENT_RBUTTONDOWN:
        print("(" + str(x) + ", " + str(y) + ")")
    cv2.imshow('dst', dst)           
    
dst = np.zeros(shape=(512, 512, 3), dtype=np.uint8) ##3 : color/ 흑백은 1
result = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
cv2.namedWindow('dst') ##showWindow
cv2.setMouseCallback('dst', onMouse)

count = 0
good_cnt = 0
cor_result = 10
imgcnt = 0

prompt = '''------------------------------------------------------------------------------
* 시작하려면 's'를 누르고  0~9 사이의 숫자를 입력 후 Enter를 쳐주세요!! *
* 'c': Clear, 'v': 맞춘 횟수 및 확률 확인, ' ': Predict, 'Esc': Exit, 'r' : Save_image *
------------------------------------------------------------------------------
'''
print(prompt)

while True:
    key = cv2.waitKey(25)    
    if key == 27: 
        break
    elif key == ord('v'):
            print(">>> 맞춘 횟수, 확률 |", good_cnt, "/", count,",", good_cnt/count*100,"%")
            print("\n")
            count = 0
            print(prompt)
    elif key == ord('s'):
        dst[:,:] = 0
        imgcnt+=1
        
        if count != 0 and count < 100:
            print(">>> 맞춘 횟수, 확률 |", good_cnt, "/", count,",", good_cnt/count*100,"%")
            print("\n")
            count = 0
        cor_result = int(input("* Enter the answer between 0 to 9 : "))
        print("\n")       
        good_cnt = 0
    elif key == ord('c'):
        dst[:,:] = 0
    elif key == ord(' '):
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
##        _, contours, _ = cv2.findContours(gray, mode, method) ## 버전이 달라서 아래의 코드로 수정
        contours, _ = cv2.findContours(gray, mode, method) ##conected component를 찾는다.

        for i, cnt in enumerate(contours):
            x, y, width, height = cv2.boundingRect(cnt)
            drawimg = cv2.rectangle(dst, (x, y), (x+width, y+height), (0,0,255), 2) #2:두께
            ax = int(x + width/2)
            ay = int(y + height/2)
            
            if width > height:
                r = int(width/2)
            else:
                r = int(height/2)

        cv2.rectangle(dst, (ax-r, ay-r), (ax+r, ay+r), (255, 0, 0), 2)
        temp = gray[ay-r:ay+r, ax-r:ax+r]
        temp = cv2.resize(temp, (20,20))
        x_img[4:24,4:24] = temp
        x = np.reshape(x_img,(-1, 28*28))
        x = x.astype(np.float32)/255

        network = init_network()
        y = predict(network, x)
        p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.

        count+=1

        if cor_result != 10:
            print("# 정답: %d | %d번째 입력: %d" %(cor_result, count, p))
            print("\n")

            if cor_result == p:
                good_cnt+=1
            else:
                root_dir = "fi/"
                today = get_today()
                work_dir = root_dir+today
                make_folder(work_dir)
                imgname = work_dir+"/"+str(imgcnt)+"-"+str(count)+"_c"+str(cor_result)+"-_p"+str(p)+".png"
                cv2.imwrite(imgname, x_img)

            cv2.putText(dst, str(p), (ax+r+10, ay), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 5)
            cv2.namedWindow('result')
            cv2.imshow('result',x_img)
        else:
            cv2.putText(dst, "Push the 's' button!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    elif key == ord('r'):
        imgname = "fi/"+str(imgcnt)+"-"+str(count)+"_c"+str(cor_result)+"-_p"+str(p)+".png"
        cv2.imwrite(imgname, x_img)    
    cv2.imshow('dst',dst)

cv2.destroyAllWindows()
