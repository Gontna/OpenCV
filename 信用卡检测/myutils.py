import cv2


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == 'bottom-to-top':
        reverse = True
    if method == "bottom-top" or method == "top-to-bottom":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    # zip将cnts和bundingBox 组合成一个元组列表 每一个元组都是cnts中的一个轮廓和boundingBox中的边框
    (ctns, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # b:[1][i]是拿出来每一个轮廓的边框
    return ctns, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
