 # -*- coding: utf-8 -*-

import dlib  # 人脸识别的库dlib
import cv2  # 图像处理的库OpenCv
import numpy as np  # 数据处理的库 numpy
import time
import math
from scipy.spatial import distance as dist
from imutils import face_utils


class Fatigue_detecting:

    def __init__(self):
        """参数"""
        # 默认为摄像头0
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False  # False未打开摄像头，True摄像头已打开
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5
        # 闪烁阈值（秒）
        # 眼睛长宽比
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3
        # 打哈欠长宽比
        self.MAR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = 3
        # 瞌睡点头
        self.HAR_THRESH = 0.3
        self.NOD_AR_CONSEC_FRAMES = 5

        """计数"""
        # 初始化帧计数器和眨眼总数
        self.COUNTER = 0
        self.TOTAL = 0
        # 初始化帧计数器和打哈欠总数
        self.mCOUNTER = 0
        self.mTOTAL = 0
        # 初始化帧计数器和点头总数
        self.hCOUNTER = 0
        self.hTOTAL = 0
        # 离职时间长度
        self.oCOUNTER = 0
        # 初始化眨眼频率,点头频率，打哈欠频率
        self.frequency = 0
        self.hfrequency = 0
        self.yfrequency = 0
        # 初始化疲劳程度
        self.score = 0

        """姿态"""
        # 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                                      [1.330353, 7.122144, 6.903745],  # 29左眉右角
                                      [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                                      [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                                      [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                                      [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                                      [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                                      [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                                      [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                                      [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                                      [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                                      [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                                      [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                                      [0.000000, -7.415691, 4.070434]])  # 6下巴角

        # 相机坐标系(XYZ)：添加相机内参
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                  0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                  0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
        # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

        # 像素坐标系(xy)：填写凸轮的本征和畸变系数
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)

        # 重新投影3D点的世界坐标轴以验证结果姿势
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])
        # 绘制正方体12轴
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]

        self.running = False

    def __del__(self):
        pass

    def get_head_pose(self, shape):  # 头部姿态估计
        # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
        # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
        # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        # solvePnP计算姿势——求解旋转和平移矩阵：
        # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

        # 计算欧拉角calc euler angle
        # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

        return reprojectdst, euler_angle  # 投影误差，欧拉角（yaw/pitch/row）

    def eye_aspect_ratio(self, eye):
        """眼睛长宽比"""
        # 计算两组垂直眼睛标志（x，y）坐标之间的欧几里德距离
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # 计算水平之间的欧几里德距离
        C = dist.euclidean(eye[0], eye[3])
        # 眼睛长宽比的计算
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):  # 嘴巴长宽比
        A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
        B = dist.euclidean(mouth[4], mouth[8])  # 53, 57
        C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar

    def _learning_face(self, event):
        """dlib的初始化调用"""
        self.detector = dlib.get_frontal_face_detector()  # Dlib的人脸检测器
        self.predictor = dlib.shape_predictor('.\model\shape_predictor_68_face_landmarks.dat')  # Dlib的人脸特征点检测器
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)  # 调用摄像头

        # 分别获取左右眼面部标志的索引
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        while self.running:
            # 读取帧
            ret, frame = self.cap.read()
            flag, im_rd = self.cap.read()
            if not ret:
                break

            # 将帧调整为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测面部
            rects = self.detector(gray, 0)

            for rect in rects:
                # 获取面部特征点并转换为NumPy数组
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # 提取左眼和右眼坐标
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = self.eye_aspect_ratio(leftEye)
                rightEAR = self.eye_aspect_ratio(rightEye)

                # 计算两个眼睛的平均EAR
                ear = (leftEAR + rightEAR) / 2.0

                # 计算嘴巴的长宽比
                mouth = shape[mStart:mEnd]
                mar = self.mouth_aspect_ratio(mouth)

                # 检查是否满足眨眼阈值
                if ear < self.EYE_AR_THRESH:
                    self.COUNTER += 1
                else:
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        self.TOTAL += 1
                    self.COUNTER = 0

                # 检查是否满足打哈欠阈值
                if mar > self.MAR_THRESH:
                    self.mCOUNTER += 1
                else:
                    if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:
                        self.mTOTAL += 1
                    self.mCOUNTER = 0

                # 获取头部姿态
                reprojectdst, euler_angle = self.get_head_pose(shape)
                har = abs(euler_angle[0, 0])
                if har > self.HAR_THRESH:
                    self.hCOUNTER += 1
                else:
                    if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:
                        self.hTOTAL += 1
                    self.hCOUNTER = 0
                if self.score >= 30 and self.score <= 55:
                    cv2.putText(frame, "mid fatigue", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if self.score > 55 and self.score <= 75:
                    cv2.putText(frame, "moderate fatigue", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if self.score > 75:
                    cv2.putText(frame, "severe fatigue", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print(self.score)
                # 显示结果
                cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(frame, "Yawns: {}".format(self.mTOTAL), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(frame, "Nods: {}".format(self.hTOTAL), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                            2)
                #cv2.putText(frame, "severe fatigue", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 显示视频帧
            cv2.imshow("Frame", frame)

            # 如果按下 'q' 键，则退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        # 释放摄像头并关闭所有OpenCV窗口
        self.cap.release()
        cv2.destroyAllWindows()

    def count(self, event):
        while self.running:
            # 计算眨眼、打哈欠、点头的频率并更新疲劳分数
            # (这部分实现依赖于具体需求)
            time.sleep(1)  # 暂停1秒钟，模拟实际计算的时间间隔
            self.frequency = self.TOTAL / (time.time() / 60)  # 例子：计算每分钟眨眼次数
            self.hfrequency = self.hTOTAL / (time.time() / 60)  # 例子：计算每分钟点头次数
            self.yfrequency = self.mTOTAL / (time.time() / 60)  # 例子：计算每分钟打哈欠次数
            self.score = self.frequency + self.hfrequency + self.yfrequency  # 例子：计算疲劳分数
            if self.score >= 100:
                self.score = 100
            if self.score <= 0:
                self.score = 0
            if self.frequency > 0.47 and self.frequency < 0.61:
                self.score = self.score + 10
            if self.frequency > 0.62 and self.frequency < 0.95:
                self.score = self.score + 15
            if self.frequency > 0.96:
                self.score = self.score + 20
            if self.frequency < 0.47 and self.score >= 0:
                self.score = self.score - 5
            if self.yfrequency >= 0.2 and self.yfrequency <= 0.4:
                self.score = self.score + 10
            if self.yfrequency > 0.4 and self.yfrequency <= 0.6:
                self.score = self.score + 15
            if self.yfrequency > 0.6:
                self.score = self.score + 20
            if self.yfrequency < 0.2 and self.score >= 0:
                self.score = self.score - 10
            if self.hfrequency >= 0.2 and self.hfrequency <= 0.4:
                self.score = self.score + 15
            if self.hfrequency > 0.4 and self.hfrequency <= 0.6:
                self.score = self.score + 20
            if self.hfrequency > 0.6:
                self.score = self.score + 25
            if self.hfrequency < 0.2 and self.score >= 0:
                self.score = self.score - 20
            if self.score >= 100:
                self.score = 100
            if self.score <= 0:
                self.score = 0
    # def alarm(self, event):
    #     #while True:
    #     for i in range(500):
    #         #print("开始进入休眠")
    #         time.sleep(3)
    #         #print("结束休眠，进入新的一轮循环")
    #         if self.score >= 30 and self.score <= 55:
    #             self.m_textCtrl3.AppendText(
    #                 time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入轻度疲劳，请打起精神！！\n准备开始语音播报\n")
    #             pythoncom.CoInitialize()
    #             engine = client.Dispatch("SAPI.SpVoice")
    #             engine.Speak('警报警报，检测到您已进入轻度疲劳，请注意')
    #             # 语音播报内容
    #             # self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入轻度疲劳，请打起精神！！\n准备开始语音播报\n")
    #         if self.score > 55 and self.score <= 75:
    #             # 语音播报内容
    #             self.m_textCtrl3.AppendText(
    #                 time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入中度疲劳，请尽快打起精神！！！\n准备开始语音播报\n")
    #             pythoncom.CoInitialize()
    #             engine = client.Dispatch("SAPI.SpVoice")
    #             engine.Speak('警报警报，检测到您已进入中度疲劳，请尽快打起精神，否则即将自动报警')
    #         if self.score > 75:
    #             # 语音播报内容
    #             self.m_textCtrl3.AppendText(
    #                 time.strftime('%Y-%m-%d %H:%M ', time.localtime()) + u"警报警报已进入重度疲劳，请靠边停车，已为您自动报警\n准备开始语音播报\n")
    #             pythoncom.CoInitialize()
    #             engine = client.Dispatch("SAPI.SpVoice")
    #             engine.Speak('警报警报，检测到您已进入重度疲劳，请靠边停车，已为您自动报警')
    def camera_on(self, event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        import _thread
        self.running = True
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_face, (event,))
        _thread.start_new_thread(self.count, (event,))

    def off(self, event):
        """关闭摄像头，显示封面页"""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    fatigue_detector = Fatigue_detecting()
    event = None  # 替换为实际事件对象，如果有的话
    fatigue_detector.camera_on(event)

    # 让主线程持续运行，直到用户中断
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fatigue_detector.off(event)
