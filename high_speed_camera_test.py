import cv2
import time
import numpy as np

# mtx = np.load('/home/zgl/Desktop/mtx/mtx_h.npy')
# dist = np.load('/home/zgl/Desktop/mtx/dist_h.npy')
# v4l2-ctl --list-devices 


frame_count = 0
i = 0
cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
saveflag = False
t1 = time.time()
while True:
    ret2, frame2 = cap2.read()
 
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("test", frame)
    # h,  w = frame.shape[:2]
    # newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))   
    # undistort6q
    # dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    # x,y,w,h = roi
    # dst = dst[y:y+h, x:x+w]
    # print("\r", ret2, end ="")
    if ret2:
        
        # img = np.hstack((np.vstack((frame2,frame3)),np.vstack((frame4,frame5))))
        cv2.imshow("frame2",frame2)
        if saveflag == True:
            frame_count += 1
            if frame_count == 3:
                i += 1
                frame_count = 0
                cv2.imwrite('./tube_bump/'+str(i)+'.jpg',frame2)
                print(i)
        # print(1.0/(time.time()-t1))
        # t1 = time.time()
    

        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break
        elif k & 0xFF == ord('s'):
            print("start saving")
            saveflag = True
                

  


