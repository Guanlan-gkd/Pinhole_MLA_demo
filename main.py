import cv2
import time
import numpy as np
# import src.utils as utils
# from src.getDensity import getFilteredDensity
from src.tracker import Tracker

# v4l2-ctl --list-devices 

def put_optical_flow_arrows_on_image(image, optical_flow, threshold=2.0):
    # Don't affect original image
    image = image.copy()

    scaled_flow = optical_flow * 3.0  # scale factor

    # Get start and end coordinates of the optical flow
    flow_start = np.stack(
        np.meshgrid(range(0, scaled_flow.shape[1], 30),
                    range(0, scaled_flow.shape[0], 30)), 2)
    flow_start[:,:,0] += 9
    flow_start[:,:,1] += 29
    flow_end = (scaled_flow[flow_start[:, :, 1], flow_start[:, :, 0], :] +
                flow_start).astype(np.int32)

    # Threshold values
    norm = np.linalg.norm(scaled_flow[flow_start[:, :, 1], flow_start[:, :,
                                                                      0], :],
                          axis=2)
    # print(norm.max(), norm.min())
    norm[norm < threshold] = 0
    # Draw all the nonzero values
    nz = np.nonzero(norm)
    norm = np.asarray((norm - norm.min())/ norm.max()* 255.0, dtype='uint8')
    # print(norm.max(), norm.min())
    color_image = cv2.applyColorMap(norm, cv2.COLORMAP_RAINBOW).astype('int')
    for i in range(len(nz[0])):
        y, x = nz[0][i], nz[1][i]
        cv2.arrowedLine(image,
                        pt1=tuple(flow_start[y, x]),
                        pt2=tuple(flow_end[y, x]),
                        color=(int(color_image[y, x, 0]), int(color_image[y, x, 1]),
                               int(color_image[y, x, 2])),
                        thickness=2,
                        tipLength=.3)
    return image


def compute_curl(vector_field):
    """
    计算二维向量场的旋度

    参数:
    vector_field (ndarray): 二维向量场，形状为 (m, n, 2)

    返回值:
    ndarray: 旋度场，形状为 (m, n)
    """

    # 获取向量场的大小
    m, n, _ = vector_field.shape

    # 提取 x 和 y 方向上的分量
    x_component = vector_field[:, :, 0]
    y_component = vector_field[:, :, 1]

    # 计算 x 和 y 方向上的梯度
    dx_dy = np.gradient(x_component)
    dy_dx = np.gradient(y_component)

    # 提取 x 和 y 方向上的梯度分量
    dx = dx_dy[1]
    dy = dy_dx[0]

    # 计算旋度
    curl = dy - dx

    return curl


def compute_div(vector_field):
    """
    计算二维向量场的旋度

    参数:
    vector_field (ndarray): 二维向量场，形状为 (m, n, 2)

    返回值:
    ndarray: 旋度场，形状为 (m, n)
    """

    # 获取向量场的大小
    m, n, _ = vector_field.shape

    # 提取 x 和 y 方向上的分量
    x_component = vector_field[:, :, 0]
    y_component = vector_field[:, :, 1]

    # 计算 x 和 y 方向上的梯度
    dx_dy = np.gradient(x_component)
    dy_dx = np.gradient(y_component)

    # 提取 x 和 y 方向上的梯度分量
    dx = dx_dy[1]
    dy = dy_dx[0]

    # 计算旋度
    div = dy + dx

    return div

saveflag = False
frame_count = 0
i = 0
cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

ncol = 4
nrow = 3

tracker = Tracker(adaptive=True,
                      cuda=False)  # cuda=True if using opencv cuda

vertices = [(64,210,354,502),(52,209,364),90,90]  # top left x, top left y, length, width

mag_bar = np.zeros((270,10,3),np.uint8)
pt1 = (0,0)
pt2 = (10,270)
cv2.rectangle(mag_bar,pt1,pt2,(255,0,0),-1)

T_bar = np.zeros((10,360,3),np.uint8)
pt1 = (0,0)
pt2 = (360,10)
cv2.rectangle(T_bar,pt1,pt2,(180,0,110),-1)

x_bar = np.zeros((10,360,3),np.uint8)
pt1 = (0,0)
pt2 = (360,10)
cv2.rectangle(x_bar,pt1,pt2,(0,255,0),-1)

y_bar = np.zeros((270,10,3),np.uint8)
pt1 = (0,0)
pt2 = (10,270)
cv2.rectangle(y_bar,pt1,pt2,(0,0,255),-1)

# cv2.imshow("x_bar",x_bar)
# cv2.imshow("y_bar",y_bar)
# cv2.imshow("mag_bar",mag_bar)
# cv2.imshow("T_bar",T_bar)
# cv2.waitKey()

# print(1)
while True:
    ret, frame = cap2.read()
    # print("\r", ret, end ="")
    if ret:
        # cv2.imshow("frame",frame)
        vstack = []
        # tile_list = []
        for i in range(nrow):
            hstack = []
            for j in range(ncol):
                left_top = (vertices[0][j],vertices[1][i])
                
                tile = frame[ left_top[1] : left_top[1] + vertices[2], 
                                left_top[0] : left_top[0] + vertices[2] ,
                                :]
                
                tile_flipped = cv2.flip(tile, -1)
                
                # tile_list.append(tile_flipped)
                if j == 0:
                    hstack = tile_flipped
                else:
                    hstack = np.hstack((hstack, tile_flipped))
            if i == 0:
                vstack = hstack
            else:
                vstack = np.vstack((vstack, hstack))

        # cv2.imshow("stack", vstack)
    
    vstack = cv2.flip(cv2.cvtColor(vstack, cv2.COLOR_BGR2GRAY),1)
    flow = tracker.track(vstack)
    # mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
    mag = compute_div(flow)
    mag_sum = np.sum(mag)/10000
    x_sum = -1 * np.sum(flow[:,:,0])/1000000
    y_sum = -1 * np.sum(flow[:,:,1])/1000000
    T_sum = np.sum(compute_curl(flow))/2000
    print(mag_sum)
    

    pt1 = (0,0)
    pt2 = (10,270)
    cv2.rectangle(mag_bar,pt1,pt2,(255,0,0),-1)

    pt1 = (0,0)
    pt2 = (360,10)
    cv2.rectangle(T_bar,pt1,pt2,(180,0,110),-1)

    pt1 = (0,0)
    pt2 = (360,10)
    cv2.rectangle(x_bar,pt1,pt2,(0,255,0),-1)

    pt1 = (0,0)
    pt2 = (10,270)
    cv2.rectangle(y_bar,pt1,pt2,(0,0,255),-1)

    index_mag = min(int(mag_sum * 270 *1.5), 270)
    # print(270 - index_mag)
    mag_bar[0: 270 - index_mag,:,:] = 0

    if y_sum >= 0:
        index_y = min(int(y_sum * 135 * 1.5), 135)
        y_bar[0: 135 - index_y,:,:] = 0
        y_bar[135::,:,:] = 0
    else:
        index_y = max(int(y_sum * 135 * 1.5), -135)        
        y_bar[135 + abs(index_y)::,:,:] = 0
        y_bar[0:135,:,:] = 0

    if x_sum >= 0:
        index_x = min(int(x_sum * 180 * 1.5), 180)
        x_bar[:, 0: 185 - index_x,:] = 0
        x_bar[:, 180::,:] = 0
    else:
        index_x = max(int(x_sum * 180 * 1.5), -180)        
        x_bar[:, 180 + abs(index_x)::,:] = 0
        x_bar[:,0:180,:] = 0

    if T_sum >= 0:
        index_T = min(int(T_sum * 180), 180)
        T_bar[:, 0: 185 - index_T,:] = 0
        T_bar[:, 180::,:] = 0
    else:
        index_T = max(int(T_sum * 180), -180)        
        T_bar[:, 180 + abs(index_T)::,:] = 0
        T_bar[:,0:180,:] = 0



    # # print(index_x)
    # cv2.rectangle(mag_bar,pt1,pt2,(255,0,0),-1)
    # mag_bar[0: 270 - index_mag,:,:] = 0

    # index_y = min(int(y_sum * 270 *1.5), 270)
    # print(270 - index_mag)
    # cv2.rectangle(mag_bar,pt1,pt2,(255,0,0),-1)
    # mag_bar[0: 270 - index_mag,:,:] = 0

    # index_T = min(int(T_sum * 270 *1.5), 270)
    # print(270 - index_mag)
    # cv2.rectangle(mag_bar,pt1,pt2,(255,0,0),-1)
    # mag_bar[0: 270 - index_mag,:,:] = 0



    arrows = put_optical_flow_arrows_on_image(
                cv2.cvtColor(vstack,cv2.COLOR_GRAY2BGR), flow[15:-15, 15:-15, :])
    # cv2.imshow('arrows', arrows)
    
    black_block = np.zeros((10,10,3),np.uint8)
    cv2.rectangle(T_bar,(178,0),(182,10),(255,255,0),-1)
    cv2.rectangle(x_bar,(178,0),(182,10),(255,255,0),-1)
    cv2.rectangle(y_bar,(0,133),(10,137),(255,255,0),-1)
    
    bar = np.hstack((
        np.vstack((black_block, mag_bar, black_block)), 
        np.vstack((T_bar, arrows, x_bar)), 
        np.vstack((black_block, y_bar, black_block))
                     ))
    cv2.imshow("bar", bar)

    key = cv2.waitKey(1)  & 0xFF

    if key == ord('q'):
        print("\n")
        break
    elif key == ord('r'):
        tracker.reset()
        print('reset')
    elif key == ord('s'):
        cv2.imwrite('./'+str(i)+'.jpg',bar)
        cv2.imwrite('./'+str(i)+'_frame.jpg',vstack)