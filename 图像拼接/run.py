import cv2
import numpy as np


class Stitcher:
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageB, imageA) = images
        # 检测两张图片中的关键点
        (kpsA, featureA) = self.detectAndDescribe(imageA)
        (kpsB, featureB) = self.detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = self.matchKeypoints(kpsA, kpsB, featureA, featureB, ratio, reprojThresh)

        # 如果返回结果为空,没有匹配成功的特征带点，退出算法
        if M is None:
            return None

        # 否则提取匹配结果
        #H是3*3视角变换矩阵
        (matches,H,status) = M
        # 将图片A进行视角变换,result 是变换后的图片
        result = cv2.warpPerspective(imageA,H,(imageA.shape[1]+imageB.shape[1],imageA.shape[0]))
        self.show('result',result)

        # 将图片B传入result图片最左端
        result[0:imageB.shape[0],0:imageB.shape[1]] = imageB
        self.show('result',result)
        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

            # 返回匹配结果
        return result
    def detectAndDescribe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor =cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)# kp是keypoint类型对象
        # 将计算结果转为numpy数组
        kps = np.float32([kp.pt for kp in kps]) # kp.pt 是坐标信息
        # 返回特征点集， 以及对应的特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featureA, featureB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()
        # 使用KNN 检测来自Ab图的sift特征配对 K=2
        rawMatches = matcher.knnMatch(featureA, featureB, 2)

        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 储存两个点在featureA 和featureB 的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # 当筛选后的匹配对大于4是，计算视角变换矩阵
        if len(matches)>4:
            # 获取匹配对的坐标点
            ptsA = np.float32([kpsA[i] for (_,i) in matches])
            ptsB = np.float32([kpsB[i] for (i,_) in matches])

            #计算视角变换矩阵 statu是一个数组来表示匹配点是否是内点,如果是内点则为1
            (H,status) = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,reprojThresh)#RNSAC算法+H矩阵计算
            return (matches,H,status)
        # 如果匹配结果小于4 返回None
        return None
    def show(self,name,img):
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis

s = Stitcher()
imgA = cv2.imread('./left_01.png')
imgB = cv2.imread('./right_01.png')
(result,vis) = s.stitch([imgA,imgB],showMatches=True)
cv2.imshow("left",imgA)
cv2.imshow("right",imgB)
cv2.imshow("result",result)
cv2.imshow("vis",vis)
cv2.waitKey(0)
cv2.destroyAllWindows()