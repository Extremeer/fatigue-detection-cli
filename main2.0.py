import socket
import cv2
import numpy as np
import threading
import queue
import time
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

# 创建共享队列
frame_queue = queue.Queue()

# 创建VideoWriter对象
fps = 30  # 设置帧率
frame_width, frame_height = 640, 480  # 设置帧的宽度和高度
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定socket到一个可用的端口上
server_address = ('192.168.200.1', 8888)  # 设置PC端的IP地址和端口号
server_socket.bind(server_address)

# 监听连接
server_socket.listen(1)

def receive_frames():
    while True:
        # 接受客户端连接
        client_socket, client_address = server_socket.accept()  # 接收一次消息
        print('等待客户端连接...')
        print('客户端已连接:', client_address)

        # 接收图像数据流并解码
        image_data = b''
        while True:
            data = client_socket.recv(1024)
            if not data:
                print("No data")
                break
            image_data += data

            # 检查接收到的数据流是否已经包含完整的图片
            if image_data.endswith(b'\xff\xd9'):
                # 将接收到的数据流转换为图像
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                print("Get Pic")

                # 写入视频帧
                out.write(image)

                # 将图像写入队列
                frame_queue.put(image)

                # 显示解码后的图像
                cv2.imshow("1", image)
                cv2.waitKey(1)  # 设置适当的等待时间，单位为毫秒

                # 清空图像数据，准备接收下一张图片
                image_data = b''

        # 关闭客户端socket
        client_socket.close()

    # 关闭VideoWriter对象和服务器socket
    out.release()
    server_socket.close()

class Fatigue_detecting:
    def __init__(self):
        # 初始化参数和变量
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = 3
        self.MAR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = 3
        self.HAR_THRESH = 0.3
        self.NOD_AR_CONSEC_FRAMES = 5
        self.COUNTER = 0
        self.TOTAL = 0
        self.mCOUNTER = 0
        self.mTOTAL = 0
        self.hCOUNTER = 0
        self.hTOTAL = 0
        self.oCOUNTER = 0
        self.frequency = 0
        self.hfrequency = 0
        self.yfrequency = 0
        self.score = 0

        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                                      [1.330353, 7.122144, 6.903745],
                                      [-1.330353, 7.122144, 6.903745],
                                      [-6.825897, 6.760612, 4.402142],
                                      [5.311432, 5.485328, 3.987654],
                                      [1.789930, 5.393625, 4.413414],
                                      [-1.789930, 5.393625, 4.413414],
                                      [-5.311432, 5.485328, 3.987654],
                                      [2.005628, 1.409845, 6.165652],
                                      [-2.005628, 1.409845, 6.165652],
                                      [2.774015, -2.080775, 5.048531],
                                      [-2.774015, -2.080775, 5.048531],
                                      [0.000000, -3.116408, 6.097667],
                                      [0.000000, -7.415691, 4.070434]])
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                  0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                  0.0, 0.0, 1.0]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                        [10.0, 10.0, -10.0],
                                        [10.0, -10.0, -10.0],
                                        [10.0, -10.0, 10.0],
                                        [-10.0, 10.0, 10.0],
                                        [-10.0, 10.0, -10.0],
                                        [-10.0, -10.0, -10.0],
                                        [-10.0, -10.0, 10.0]])
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]

        self.running = False

    def __del__(self):
        pass

    def get_head_pose(self, shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,
                                            self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        return reprojectdst, euler_angle

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[4], mouth[8])
        C = dist.euclidean(mouth[0], mouth[6])
        mar = (A + B) / (2.0 * C)
        return mar

    def _learning_face(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

        while self.running:
            if not frame_queue.empty():
                frame = frame_queue.get()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 0)

                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:
                                    face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
                    rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:
                                     face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
                    mouth = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:
                                  face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]]

                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    mar = self.mouth_aspect_ratio(mouth)
                    ear = (leftEAR + rightEAR) / 2.0

                    if ear < self.EYE_AR_THRESH:
                        self.COUNTER += 1
                    else:
                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            self.TOTAL += 1
                        self.COUNTER = 0

                    if mar > self.MAR_THRESH:
                        self.mCOUNTER += 1
                    else:
                        if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:
                            self.mTOTAL += 1
                        self.mCOUNTER = 0

                    reprojectdst, euler_angle = self.get_head_pose(shape)
                    har = euler_angle[0, 0]

                    if har > self.HAR_THRESH:
                        self.hCOUNTER += 1
                    else:
                        if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:
                            self.hTOTAL += 1
                        self.hCOUNTER = 0

                    if ear < self.EYE_AR_THRESH and mar > self.MAR_THRESH and har > self.HAR_THRESH:
                        self.oCOUNTER += 1
                    else:
                        if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                            self.oCOUNTER = 0
                            self.score += 1

                cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Yawns: {}".format(self.mTOTAL), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Nods: {}".format(self.hTOTAL), (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Score: {}".format(self.score), (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Frame", frame)
                cv2.waitKey(1)  # 设置适当的等待时间，单位为毫秒

    def run(self):
        self.running = True
        self._learning_face()

# 创建并启动接收帧的线程
receive_thread = threading.Thread(target=receive_frames)
receive_thread.start()

# 创建并启动疲劳检测的实例
fatigue_detector = Fatigue_detecting()
fatigue_detector_thread = threading.Thread(target=fatigue_detector.run)
fatigue_detector_thread.start()

# 等待线程结束
receive_thread.join()
fatigue_detector_thread.join()

# 释放资源
cv2.destroyAllWindows()
