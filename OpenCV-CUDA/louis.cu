#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ double determinantOfCovariance(cv::cuda::PtrStepSz<uchar3> image, int row, int col, int neighborhood, int rows, int cols) {
    // Define the neighborhood bounds
    int half_neighborhood = neighborhood / 2;
    int start_row = max(0, row - half_neighborhood);
    int end_row = min(rows - 1, row + half_neighborhood);
    int start_col = max(0, col - half_neighborhood);
    int end_col = min(cols - 1, col + half_neighborhood);

    // Extract the neighborhood pixels
    int sumR = 0, sumG = 0, sumB = 0;
    int counter = 0;
    for (int i = start_row; i <= end_row; ++i) {
        for (int j = start_col; j <= end_col; ++j) {
            sumR += image(i, j).z;
            sumG += image(i, j).y;
            sumB += image(i, j).x;
            counter++;
        }
    }

    // Compute the mean of the neighborhood
    double meanR = static_cast<double>(sumR) / (counter);
    double meanG = static_cast<double>(sumG) / (counter);
    double meanB = static_cast<double>(sumB) / (counter);

    // Compute the covariance matrix
    double3 covariance1 = {0., 0., 0.};
    double3 covariance2 = {0., 0., 0.};
    double3 covariance3 = {0., 0., 0.};
    int counter2 = 0;
    for (int i = start_row; i <= end_row; ++i) {
        for (int j = start_col; j <= end_col; ++j) {
            double diffR = image(i, j).z - meanR;
            double diffG = image(i, j).y - meanG;
            double diffB = image(i, j).x - meanB;
            covariance3.z += diffR * diffR;
            covariance3.y += diffR * diffG;
            covariance3.x += diffR * diffB;

            covariance2.z += diffG * diffR;
            covariance2.y += diffG * diffG;
            covariance2.x += diffG * diffB;

            covariance1.z += diffB * diffR;
            covariance1.y += diffB * diffG;
            covariance1.x += diffB * diffB;

            // covariance1 += (deviation.x * deviation.x, deviation.x * deviation.y, deviation.x * deviation.z);
            // covariance2 += (deviation.y * deviation.x, deviation.y * deviation.y, deviation.y * deviation.z);
            // covariance3 += (deviation.z * deviation.x, deviation.z * deviation.y, deviation.z * deviation.z);
            counter2++;
        }
    }
    covariance1 = {covariance1.x / counter2, covariance1.y / counter2, covariance1.z / counter2};
    covariance2 = {covariance2.x / counter2, covariance2.y / counter2, covariance2.z / counter2};
    covariance3 = {covariance3.x / counter2, covariance3.y / counter2, covariance3.z / counter2};

    // Compute the determinant of the covariance matrix
    return covariance1.x * covariance2.y * covariance3.z + 
           covariance1.y * covariance2.z * covariance3.x + 
           covariance1.z * covariance2.x * covariance3.y - 
           covariance1.z * covariance2.y * covariance3.x - 
           covariance1.y * covariance2.x * covariance3.z - 
           covariance1.x * covariance2.z * covariance3.y;
}


__global__ void process(cv::cuda::PtrStepSz<uchar3> d_Limg, cv::cuda::PtrStepSz<uchar3> d_Rimg, cv::cuda::PtrStepSz<uchar3> d_result, const int neighborhood, const float ratio, const int rows, const int cols) {

    const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
    const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (dst_x < cols && dst_y < rows) {
        float Ldet = determinantOfCovariance(d_Limg, dst_y, dst_x, neighborhood, rows, cols);
        float Rdet = determinantOfCovariance(d_Rimg, dst_y, dst_x, neighborhood, rows, cols);
        int Lkernel_size;
        int Rkernel_size;
        
        Lkernel_size = ratio / ((Ldet*Ldet/100000)+1);
        Rkernel_size = ratio / ((Rdet*Rdet/100000)+1);

        float3 Lsum = {0., 0., 0.};
        float3 Rsum = {0., 0., 0.};
        int Lcounter = 0;
        int Rcounter = 0;
        for (int h=-Lkernel_size/2;h<=Lkernel_size/2;h++){
            for (int w=-Lkernel_size/2;w<=Lkernel_size/2;w++){
                if (dst_y+h > 0 && dst_y+h < rows && dst_x+w > 0 && dst_x+w < cols && h*h+w*w <= Lkernel_size*Lkernel_size){
                    Lsum.x += d_Limg(dst_y+h,dst_x+w).x;
                    Lsum.y += d_Limg(dst_y+h,dst_x+w).y;
                    Lsum.z += d_Limg(dst_y+h,dst_x+w).z;
                    Lcounter ++;
                }
            }
        }
        for (int h=-Rkernel_size/2;h<=Rkernel_size/2;h++){
            for (int w=-Rkernel_size/2;w<=Rkernel_size/2;w++){
                if (dst_y+h > 0 && dst_y+h < rows && dst_x+w > 0 && dst_x+w < cols && h*h+w*w <= Rkernel_size*Rkernel_size){
                    Rsum.x +=  d_Rimg(dst_y+h,dst_x+w).x;
                    Rsum.y +=  d_Rimg(dst_y+h,dst_x+w).y;
                    Rsum.z +=  d_Rimg(dst_y+h,dst_x+w).z;
                    Rcounter ++;
                }
            }
        }

        uchar3 BlurLPixel = {(unsigned char)(Lsum.x / Lcounter), (unsigned char)(Lsum.y / Lcounter), (unsigned char)(Lsum.z / Lcounter)};
        d_result(dst_y, dst_x) = BlurLPixel;

        uchar3 BlurRPixel = {(unsigned char)(Rsum.x / Rcounter), (unsigned char)(Rsum.y / Rcounter), (unsigned char)(Rsum.z / Rcounter)};
        d_result(dst_y, dst_x + cols) = BlurRPixel;
    }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void createDenoiseCUDA(cv::cuda::GpuMat& d_Limg, cv::cuda::GpuMat& d_Rimg, cv::cuda::GpuMat& d_result, const int neighborhood, const float ratio, const int block_x, const int block_y) {
    const dim3 block(block_x, block_y);
    const dim3 grid(divUp(d_Limg.cols, block.x), divUp(d_Limg.rows, block.y));

    int rows = d_Limg.rows;
    int cols = d_Limg.cols;

    process<<<grid, block>>>(d_Limg, d_Rimg, d_result, neighborhood, ratio, rows, cols);
}