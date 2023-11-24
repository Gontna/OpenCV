# python run.py -i images/credit_card_03.png -t images/ocr_a_reference.png
import argparse
import myutils
import numpy as np

from show import show
import cv2
import myutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-t", "--template", required=True,
                help="path to template OCR-A img")
args = vars(ap.parse_args())
img = cv2.imread(args["template"])
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# 读取一个模板图像
img = cv2.imread(args["template"])
show('img', img)

# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show('ref', ref)

# 二值图像 返回两个数据 1是处理后的图像 二是阈值处理后的阈值
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
show('ref', ref)

# 轮廓检测
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
show('img', img)
# 轮廓个数
# print(np.array(refCnts).shape)
# 轮廓排序
refCnts = myutils.sort_contours(refCnts, "left-to-right")[0]

digits = {}

# 遍历每一个轮廓并裁剪轮廓尺寸
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    # 截取y~y+h 和x~x+w的区域
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # 每个数字对应一个模板
    digits[i] = roi

# 定义卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取输入图像
image = cv2.imread(args["image"])
image = myutils.resize(image, width=300)
show("image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show("gray", gray)

# 礼帽 取出亮度高的区域
topHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
show("topHat", topHat)

# sobel算子计算边缘
gradX = cv2.Sobel(topHat, cv2.CV_32F, dx=1, dy=0, ksize=-1)
# gradX = cv2.convertScaleAbs(gradX)
gradX = np.absolute(gradX)
# 归一化处理
(minValue, maxValue) = (np.min(gradX), np.max(gradX))
print(minValue, maxValue)
gradX = (255 * (gradX - minValue) / (maxValue - minValue))
gradX = gradX.astype("uint8")
show("gradx", gradX)

# 将数字连在一起,先腐蚀后膨胀
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
show("grad", gradX)

# 二值操作,使数字区域分块,每一块都连在一起
# 阈值设置成零和cv2.THRESH_OTSU搭配使用,图片有两个主体时(双峰)自动判断阈值
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
show("thrsh", thresh)

# 闭操作,使缝隙被填充
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
show("thrsh", thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
show("cur_img", cur_img)
locs = []

# 过滤多余轮廓
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    # 通过轮廓长宽比来判断轮廓是否符合要求
    ar = w / float(h)

    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            # 保留符合特征的轮廓
            locs.append((x, y, w, h))

# 轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    # g根据坐标提取到每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    show("轮廓", group)

    # 预处理 得到每一个小轮廓
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    show("1", group)
    ctns, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 轮廓排序
    digitCnts = myutils.sort_contours(ctns, "left-to-right")[0]

    # 计算每一组中的一个数值
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        # 需要与模板大小一直
        roi = cv2.resize(roi, (57, 88))
        show('roi', roi)

        # 计算得分
        scores = []

        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

print("crad Type{}".format(FIRST_NUMBER[output[0]]))
# print("ID{}".format(output))
print("ID:{}".format("".join(output)))
show("image", image)
