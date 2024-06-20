import socket
import cv2
import numpy as np

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

while True:
    # 接受客户端连接
    client_socket, client_address = server_socket.accept()  # 接收一次消息
    print('等待客户端连接...')
    #
    print('客户端已连接:', client_address)

    # 接收图像数据流并解码
    image_data = b''
    while True:
        data = client_socket.recv(1024)
        print()

        if not data:
            print("Nodata")
            break
        image_data += data

        # 检查接收到的数据流是否已经包含完整的图片
        if image_data.endswith(b'\xff\xd9'):
            # 将接收到的数据流转换为图像
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print("GetPic")

            # 写入视频帧
            out.write(image)

            # 显示解码后的图像
            cv2.imshow("1",image)
            cv2.waitKey(1)  # 设置适当的等待时间，单位为毫秒

            # 清空图像数据，准备接收下一张图片
            image_data = b''


    # 关闭客户端socket
    client_socket.close()

# 关闭VideoWriter对象和服务器socket
out.release()
server_socket.close()
