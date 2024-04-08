#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void process1(cv::cuda::PtrStepSz<uchar3> leftImage, cv::cuda::PtrStepSz<uchar3> rightImage, cv::cuda::PtrStepSz<uchar3> anaglyphImage, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
        uchar3 leftPixel = leftImage(dst_y, dst_x);
        uchar3 rightPixel = rightImage(dst_y, dst_x);
        uchar3 anaglyphPixel;
        anaglyphPixel.z = 0.299*leftPixel.z + 0.587*leftPixel.y +0.144*leftPixel.x;
        anaglyphPixel.y = 0;
        anaglyphPixel.x = 0.299*rightPixel.z + 0.587*rightPixel.y +0.144*rightPixel.x;
        anaglyphImage(dst_y, dst_x) = anaglyphPixel;
    }
}
__global__ void process2(cv::cuda::PtrStepSz<uchar3> leftImage, cv::cuda::PtrStepSz<uchar3> rightImage, cv::cuda::PtrStepSz<uchar3> anaglyphImage, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
        uchar3 leftPixel = leftImage(dst_y, dst_x);
        uchar3 rightPixel = rightImage(dst_y, dst_x);
        uchar3 anaglyphPixel;
        anaglyphPixel.z = 0.299*leftPixel.z + 0.587*leftPixel.y +0.144*leftPixel.x;
        anaglyphPixel.y = 0.299*rightPixel.z + 0.587*rightPixel.y +0.144*rightPixel.x;
        anaglyphPixel.x = 0.299*rightPixel.z + 0.587*rightPixel.y +0.144*rightPixel.x;
        anaglyphImage(dst_y, dst_x) = anaglyphPixel;
    }
}
__global__ void process3(cv::cuda::PtrStepSz<uchar3> leftImage, cv::cuda::PtrStepSz<uchar3> rightImage, cv::cuda::PtrStepSz<uchar3> anaglyphImage, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
        uchar3 leftPixel = leftImage(dst_y, dst_x);
        uchar3 rightPixel = rightImage(dst_y, dst_x);
        uchar3 anaglyphPixel;
        anaglyphPixel.z = leftPixel.z;
        anaglyphPixel.y = rightPixel.y;
        anaglyphPixel.x = rightPixel.x;
        anaglyphImage(dst_y, dst_x) = anaglyphPixel;
    }
}
__global__ void process4(cv::cuda::PtrStepSz<uchar3> leftImage, cv::cuda::PtrStepSz<uchar3> rightImage, cv::cuda::PtrStepSz<uchar3> anaglyphImage, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
        uchar3 leftPixel = leftImage(dst_y, dst_x);
        uchar3 rightPixel = rightImage(dst_y, dst_x);
        uchar3 anaglyphPixel;
        anaglyphPixel.z = 0.299*leftPixel.z + 0.587*leftPixel.y +0.144*leftPixel.x;
        anaglyphPixel.y = rightPixel.y;
        anaglyphPixel.x = rightPixel.x;
        anaglyphImage(dst_y, dst_x) = anaglyphPixel;
    }
}
__global__ void process5(cv::cuda::PtrStepSz<uchar3> leftImage, cv::cuda::PtrStepSz<uchar3> rightImage, cv::cuda::PtrStepSz<uchar3> anaglyphImage, int rows, int cols)
{
    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
        uchar3 leftPixel = leftImage(dst_y, dst_x);
        uchar3 rightPixel = rightImage(dst_y, dst_x);
        uchar3 anaglyphPixel;
        anaglyphPixel.z = 0.7*leftPixel.y +0.3*leftPixel.x;
        anaglyphPixel.y = rightPixel.y;
        anaglyphPixel.x = rightPixel.x;
        anaglyphImage(dst_y, dst_x) = anaglyphPixel;
    }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void createAnaglyphCUDA ( cv::cuda::GpuMat& d_Limg, cv::cuda::GpuMat& d_Rimg, int method, cv::cuda::GpuMat& d_result, int block_x, int block_y)
{

    const dim3 block(block_x, block_y);
    const dim3 grid(divUp(d_Limg.cols, block.x), divUp(d_Limg.rows, block.y));

    const int rows = d_Limg.rows;
    const int cols = d_Limg.cols;

    if (method == 1) {
        process1<<<grid, block>>>(d_Limg, d_Rimg, d_result, rows, cols);
    } else if (method == 2) {
        process2<<<grid, block>>>(d_Limg, d_Rimg, d_result, rows, cols);
    } else if (method == 3) {
        process3<<<grid, block>>>(d_Limg, d_Rimg, d_result, rows, cols);
    } else if (method == 4) {
        process4<<<grid, block>>>(d_Limg, d_Rimg, d_result, rows, cols);
    } else {
        process5<<<grid, block>>>(d_Limg, d_Rimg, d_result, rows, cols);
    }
}