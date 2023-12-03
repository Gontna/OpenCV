import cv2
import pytesseract

img = cv2.imread('./1.png')
# pytesseract 读取RGB opencv 是BGR
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgH,imgW,_ = img.shape

# boxes=pytesseract.image_to_boxes(img)
# print(boxes) #这时box是一个字符串
#
# for box in boxes.splitlines(): # 按行分割成多个字符串
#     # 以空格分割成字符列表
#     box = box.split(' ')
#     print(box)
#     # image_to_box 的返回值是左下角坐标点和右上角坐标点
#     x,y,xW,yH = int(box[1]),int(box[2]),int(box[3]),int(box[4])
#     cv2.rectangle(img,(x,imgH-y),(xW,imgH-yH),(0,0,255),2)

textDatas = pytesseract.image_to_data(img)
# print(textDatas)
textDatas = textDatas.splitlines()
for i,data in enumerate(textDatas):

    data = data.split()
    print(data)
    if i!=0 and len(data)==12:
        # 这四个值是左上角坐标点和右下角坐标点
        x,y,w,h = int(data[6]),int(data[7]),int(data[8]),int(data[9])
        cv2.rectangle(img,(x,y),(x+w,h+y),(0,0,255),2)
        cv2.putText(img,data[11],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow("result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
