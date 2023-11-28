import argparse
import numpy as np
import cv2

ag = argparse.ArgumentParser()
ag.add_argument("-i", "--image",
                required=True, help="Path to image to be scanned")
args = vars(ag.parse_args())


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img, dim, interpolation=inter)
    return resized


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # 按顺序找到对应坐标 0 1 2 3 左上，右上，右下，左下
    # 计算坐标 的x,y的和
    sum = pts.sum(axis=1)  # 对每一行进行相加 axis=0则是对每一列进行相加
    rect[0] = pts[np.argmin(sum)]  # np.argmin 的返回值是一个索引
    rect[2] = pts[np.argmax(sum)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    :param image: 变换图片
    :param pts: 原始坐标
    :return: 变换后的图片
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算宽高
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2)
    heightB = np.sqrt((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))

    # 变换之后的坐标位置
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # dsize 要求int类型
    return warped

image = cv2.imread(args["image"])

# 是小票图片则旋转
if args["image"] == 'images/receipt.jpg':
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
# 计算resize的比例
ratio = image.shape[0] / 500.0
orig = image.copy()

image = resize(orig, height=500)
# 预处理
# 灰度图,高斯滤波去除噪音点
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

show("edged", edged)

# 轮廓检测
ctns = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
# 按面积或周长来排序轮廓
cnts = sorted(ctns, key=cv2.contourArea, reverse=True)[:5]
# cv2.drawContours(image,cnts,-1,(0,255,0),2)
# show("ln",image)

for c in cnts:
    peri = cv2.arcLength(c, True)
    # 轮廓近似 目的是找到最外面的轮廓
    # 返回值是顶点坐标
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

# 展示轮廓
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
show("outline", image)

# 透视变换 将图片"扶正" 需要两组坐标点 一组是原始坐标,另一组是变换坐标即[(0,0),(w,0),(0,h),(w,h)]
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)  # 因为用到的图片是经过变换后的,所以坐标位置也要经过一次变换
# 二值处理
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite('scan.jpg', ref)

# show("rotation",resize(cv2.rotate(ref, cv2.ROTATE_90_COUNTERCLOCKWISE),height=650))
cv2.imshow("original",resize(orig,height=650))
cv2.imshow("warped",resize(warped,height=650))
cv2.imshow("ref",resize(ref,height=650))
cv2.waitKey(0)