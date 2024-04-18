import cv2
import numpy as np


class Tracker:
    def __init__(self,
                 adaptive,
                 cuda,
                 lower_threshold=12.0,
                 upper_threshold=18.0):
        self.cuda = cuda
        if not cuda:
            # DIS method
            self.__tracking_inst = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

            self.__coarse_tracking_inst = cv2.DISOpticalFlow_create(
                cv2.DISOpticalFlow_PRESET_ULTRAFAST)
        else:
            # #cuda flow method
            self.__tracking_inst = 	cv2.FarnebackOpticalFlow.create()
            self.__coarse_tracking_inst = self.__tracking_inst
        self.__fine_base_frame_queue = []
        self.__coarse_base_frame_queue = []
        self.__fine_flow_queue = []
        self.__prev_fine_flow = None
        self.__prev_coarse_flow = None
        self.__adaptive = adaptive
        self.__lower_threshold = lower_threshold  # error lower than threshold will close the loop
        self.__upper_threshold = upper_threshold  # error greater than threshold will add new reference frame

    def track(self, frame):
        downScaled_size = (frame.shape[1] // 20, frame.shape[0] // 20)
        if self.cuda:
            frame = cv2.cuda_GpuMat(frame)
        if self.__adaptive:
            if not self.__fine_base_frame_queue:
                self.__fine_base_frame_queue.append(frame)
                x = np.linspace(0,
                                downScaled_size[0] - 1,
                                downScaled_size[0],
                                dtype='float32')
                y = np.linspace(0,
                                downScaled_size[1] - 1,
                                downScaled_size[1],
                                dtype='float32')
                self.map = np.array(np.meshgrid(x, y)).transpose((1, 2, 0))
                if self.cuda:
                    cuMat = cv2.cuda_GpuMat()
                    cuMat.upload(self.map)
                    self.map = cuMat
                    self.__coarse_base_frame_queue.append(
                        cv2.cuda.resize(frame,
                                        downScaled_size,
                                        interpolation=cv2.INTER_CUBIC))
                    self.__fine_flow_queue.append(
                        cv2.cuda_GpuMat(frame.size(), cv2.CV_32FC2,
                                        (0.0, 0.0)))
                    self.__prev_fine_flow = cv2.cuda_GpuMat(
                        frame.size(), cv2.CV_32FC2, (0.0, 0.0))
                    self.__prev_coarse_flow = cv2.cuda_GpuMat(
                        downScaled_size[1], downScaled_size[0], cv2.CV_32FC2,
                        (0.0, 0.0))
                else:
                    self.__coarse_base_frame_queue.append(
                        cv2.resize(frame,
                                   downScaled_size,
                                   interpolation=cv2.INTER_CUBIC))
                    self.__fine_flow_queue.append(
                        np.zeros((frame.shape[0], frame.shape[1], 2)))
                    self.__prev_fine_flow = np.zeros(
                        (frame.shape[0], frame.shape[1], 2))
                    self.__prev_coarse_flow = np.zeros(
                        (downScaled_size[1], downScaled_size[0], 2))
                return self.__fine_flow_queue[-1]
            else:
                if self.cuda:
                    downScaled_frame = cv2.cuda.resize(
                        frame, downScaled_size, interpolation=cv2.INTER_CUBIC)
                else:
                    downScaled_frame = cv2.resize(
                        frame, downScaled_size, interpolation=cv2.INTER_CUBIC)
                self.__checkLoopClosure(downScaled_frame)
                self.__prev_coarse_flow = self.__coarse_tracking_inst.calc(
                    self.__coarse_base_frame_queue[-1], downScaled_frame,
                    self.__prev_coarse_flow)
                self.__prev_fine_flow = self.__tracking_inst.calc(
                    self.__fine_base_frame_queue[-1], frame,
                    self.__prev_fine_flow)
                if self.cuda:
                    final_flow = cv2.cuda.add(self.__fine_flow_queue[-1],
                                              self.__prev_fine_flow)
                else:
                    final_flow = self.__fine_flow_queue[
                        -1] + self.__prev_fine_flow
                error = self.__getForwardBackwardError(
                    self.__coarse_base_frame_queue[-1], downScaled_frame,
                    self.__prev_coarse_flow)
                # print("Error, ", error)
                if error > self.__upper_threshold:
                    self.__fine_base_frame_queue.append(frame)
                    self.__coarse_base_frame_queue.append(downScaled_frame)
                    self.__fine_flow_queue.append(final_flow)
                    self.__prev_fine_flow = None
                    self.__prev_coarse_flow = None
                return final_flow
        else:
            if not self.__fine_base_frame_queue:
                self.__fine_base_frame_queue.append(frame)
            self.__prev_fine_flow = self.__tracking_inst.calc(
                self.__fine_base_frame_queue[-1], frame, self.__prev_fine_flow)
            return self.__prev_fine_flow

    def __getForwardBackwardError(self, I0, I1, flow):
        if self.cuda:
            flow_map = cv2.cuda.add(self.map, flow)
            map_x, map_y = cv2.cuda.split(flow_map)
            z = cv2.cuda.remap(I1,
                               map_x,
                               map_y,
                               interpolation=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
            t = I0.clone()
            t = t.convertTo(rtype=cv2.CV_32FC1, dst=t)
            z = z.convertTo(rtype=cv2.CV_32FC1, dst=z)
            rev = cv2.cuda.absSum(cv2.cuda.subtract(
                t, z))[0] / t.size()[0] / t.size()[1]
        else:
            flow_map = self.map + flow
            z = cv2.remap(I1,
                          flow_map[:, :, 0],
                          flow_map[:, :, 1],
                          interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
            rev = np.abs(z.astype('float32') - I0.astype('float32'))
            rev = rev.mean()
        return rev

    def __checkLoopClosure(
        self, downScaled_frame
    ):  # update (prune) the frame queue with loop closure detection
        for i in range(len(self.__coarse_base_frame_queue) - 2, -1, -1):
            f = self.__coarse_tracking_inst.calc(
                self.__coarse_base_frame_queue[i], downScaled_frame, None)
            error = self.__getForwardBackwardError(
                self.__coarse_base_frame_queue[i], downScaled_frame, f)
            if error < self.__lower_threshold:
                print("loop closed at frame ", str(i))
                self.__fine_base_frame_queue = self.__fine_base_frame_queue[:
                                                                            i +
                                                                            1]
                self.__coarse_base_frame_queue = self.__coarse_base_frame_queue[:
                                                                                i
                                                                                +
                                                                                1]
                self.__fine_flow_queue = self.__fine_flow_queue[:i + 1]
                return True
        return False

    def reset(self):
        self.__fine_base_frame_queue = []
        self.__coarse_base_frame_queue = []
        self.__fine_flow_queue = []
        self.__prev_fine_flow = None
        self.__prev_coarse_flow = None

    def getFrameQueueLength(self):
        return len(self.__coarse_base_frame_queue)