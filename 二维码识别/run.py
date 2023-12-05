import cv2
from pyzbar.pyzbar import decode
import numpy as np
# img = cv2.imread('./1.png')
# print(decode(img))

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

with open('./DataList.text') as f:
    # 读取文件中的身份信息
    DataList = f.read().splitlines()
while True:
    success,img = cap.read()
    for barcode in decode(img):
        print(barcode.data)
        # barcode的属性都是bytes类型,要转为str类型
        mydata = barcode.data.decode('utf-8')
        if mydata in DataList:
            text = "Access"
            color = (0,255,0)
        else:
            text ="Un-Access"
            color=(0,0,255)
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,color,5)
        pts2 = barcode.rect
        cv2.putText(img,text,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
    cv2.imshow("result",img)
    p=cv2.waitKey(1)
    if(p==27):
        cv2.destroyAllWindows()
        break
cap.release()

