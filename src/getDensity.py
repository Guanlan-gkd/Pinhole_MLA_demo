import cupy as cp
import cv2
# import freud
import numpy as np

zero_level = None


def correctInvalidArea(density):
    # density = guided_filter(density, density, 5, 0.001, s=4)
    # invalid_ind = np.zeros_like(density, dtype='uint8')
    global zero_level
    if zero_level is None:
        zero_level = np.mean(density[60:-60, 60:-60])
    # threshold = zero_level - (density.max() - zero_level) / 120
    threshold = zero_level
    invalid_ind = twoClustering(density, threshold)
    # density = cv2.inpaint(density * 1e4, invalid_ind, 20,
    #                       cv2.INPAINT_TELEA) / 1e4
    density[np.where(invalid_ind > 0)] = zero_level  ## type 1 repair
    # density = guided_filter(density, density, 7, 0.005, s=4)
    density = cv2.ximgproc.guidedFilter(density, density, 7, 0.005)
    return density
    # return invalid_ind


def twoClustering(density, threshold):
    cluster = np.zeros_like(density, dtype='uint8')
    cluster[np.where(density < threshold)] = 1
    # kernel = np.ones((3, 3), np.uint8)
    # cluster = cv2.dilate(cluster, kernel, iterations=4)
    return cluster


def getFilteredDensity(flow, use_cuda):
    sigma = 3.0
    r_max = 9.0
    # tmp = time.time()
    if use_cuda:
        density = getDensityCuda(flow, sigma, r_max)
    else:
        density = getDensity(flow, sigma, r_max)
    # print("Density frequency: ", float(1.0 / (time.time() - tmp)))
    # tmp = time.time()
    density = correctInvalidArea(density)
    # print("Correct frequency: ", float(1.0 / (time.time() - tmp)))
    return density[15:-15, 15:-15]


def getDensity(flow, sigma, r_max):
    # returns the negative density
    x = np.linspace(-(flow.shape[1] - 1) / 2, (flow.shape[1] - 1) / 2,
                    flow.shape[1],
                    endpoint=True,
                    dtype='float32')
    y = np.linspace(-(flow.shape[0] - 1) / 2, (flow.shape[0] - 1) / 2,
                    flow.shape[0],
                    endpoint=True,
                    dtype='float32')
    coordinate = np.array(np.meshgrid(x, y), dtype='float32').transpose(
        (1, 2, 0))
    points = (coordinate + flow).reshape(-1, 2)
    points = np.hstack((points, np.zeros((points.shape[0], 1))))

    box = freud.box.Box(flow.shape[1] - 1, flow.shape[0] - 1)
    box.periodic = np.array((False, False, False))

    gd = freud.density.GaussianDensity((flow.shape[1], flow.shape[0]), r_max,
                                       sigma)
    system = freud.LinkCell(box, points, cell_width=r_max)
    # system = freud.AABBQuery(box, points)
    gd.compute(system)
    return -gd.density.T


class CudaArrayInterface:
    def __init__(self, gpu_mat):
        w, h = gpu_mat.size()
        type_map = {
            cv2.CV_8U: "u1",
            cv2.CV_8S: "i1",
            cv2.CV_16U: "u2",
            cv2.CV_16S: "i2",
            cv2.CV_32S: "i4",
            cv2.CV_32F: "f4",
            cv2.CV_64F: "f8",
        }
        self.__cuda_array_interface__ = {
            "version": 2,
            "shape": (h, w),
            "data": (gpu_mat.cudaPtr(), False),
            "typestr": type_map[gpu_mat.type()],
            "strides": (gpu_mat.step, gpu_mat.elemSize()),
        }


func_string = r"""
    extern "C"{
    __device__ float gaussian(float twoSigma2, float dx, float dy)
    {
        float pi = 3.1415926535897932384626;
        float distance = dx*dx+dy*dy;
        float result = expf(-distance/twoSigma2);
        result = result/(twoSigma2*pi);
        return result;
    }
    __global__ void density(const float *flowX, const float *flowY, float *result, int x_size, int y_size, float twoSigma2, float r_max)
    {
        int x = threadIdx.x;
        int y = blockIdx.x;
        float dest_x = float(x)+flowX[y*x_size+x];
        float dest_y = float(y) + flowY[y*x_size+x];
        for(int ind_x=ceilf(dest_x-r_max);ind_x<ceilf(dest_x+r_max); ++ind_x)
            for(int ind_y=ceilf(dest_y-r_max);ind_y<ceilf(dest_y+r_max); ++ind_y)
            {
                if(ind_x >= x_size || ind_x<0 || ind_y >= y_size || ind_y < 0)
                {
                    continue;
                }
                float dx = float(ind_x)-dest_x;
                float dy = float(ind_y)-dest_y;
                atomicAdd(result+ind_y*x_size+ind_x, -gaussian(twoSigma2, dx,dy));
            }
    }
    }
    """
funcupy = cp.RawModule(code=func_string).get_function("density")


def getDensityCuda(flow, sigma, r_max):
    twoSigma2 = cp.float32(2.0 * sigma**2)
    r_max = cp.float32(r_max)
    if type(flow).__module__ == np.__name__:
        flowx = cp.array(flow[:, :, 0], dtype=cp.float32)
        flowy = cp.array(flow[:, :, 1], dtype=cp.float32)
    else:
        flowx, flowy = cv2.cuda.split(flow)
        flowx = cp.array(CudaArrayInterface(flowx),
                         dtype=cp.float32,
                         copy=True)
        flowy = cp.array(CudaArrayInterface(flowy),
                         dtype=cp.float32,
                         copy=True)
    result = cp.zeros((flowx.shape[0], flowx.shape[1]), dtype=cp.float32)
    block_dim = (flowx.shape[1], )
    grid = (flowx.shape[0], )
    funcupy(grid, block_dim, (flowx, flowy, result, cp.int32(
        flowx.shape[1]), cp.int32(flowx.shape[0]), twoSigma2, r_max))
    return result.get()
