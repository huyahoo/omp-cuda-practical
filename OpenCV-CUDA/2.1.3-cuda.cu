#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <cuda_runtime.h>

using namespace std;

__global__ void calculateCovarianceMatrixKernel(const uchar3* image, float* covariance, int cols, int rows, int neighborhoodSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int halfSize = neighborhoodSize / 2;
        int xStart = max(0, x - halfSize);
        int yStart = max(0, y - halfSize);
        int xEnd = min(cols, x + halfSize);
        int yEnd = min(rows, y + halfSize);

        float3 mean = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = yStart; j < yEnd; ++j) {
            for (int i = xStart; i < xEnd; ++i) {
                uchar3 pixel = image[j * cols + i];
                mean.x += pixel.x;
                mean.y += pixel.y;
                mean.z += pixel.z;
            }
        }

        int count = (xEnd - xStart) * (yEnd - yStart);
        mean.x /= count;
        mean.y /= count;
        mean.z /= count;

        float3 cov = make_float3(0.0f, 0.0f, 0.0f);

        for (int j = yStart; j < yEnd; ++j) {
            for (int i = xStart; i < xEnd; ++i) {
                uchar3 pixel = image[j * cols + i];
                float3 diff = make_float3(pixel.x - mean.x, pixel.y - mean.y, pixel.z - mean.z);
                cov.x += diff.x * diff.x;
                cov.y += diff.y * diff.y;
                cov.z += diff.z * diff.z;
            }
        }

        covariance[(y * cols + x) * 3] = cov.x / count;
        covariance[(y * cols + x) * 3 + 1] = cov.y / count;
        covariance[(y * cols + x) * 3 + 2] = cov.z / count;
    }
}

__global__ void denoiseByCovarianceKernel(const uchar3* src, uchar3* dst, const float* covariance, int cols, int rows, int neighborhoodSize, float factorRatio) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        float determinant = covariance[(y * cols + x) * 3];
        determinant *= covariance[(y * cols + x) * 3 + 1];
        determinant *= covariance[(y * cols + x) * 3 + 2];

        int kernelSize;
        if (determinant != 0) {
            kernelSize = static_cast<int>(round(factorRatio / determinant));
            kernelSize = kernelSize % 2 == 0 ? kernelSize + 1 : kernelSize;
        } else {
            kernelSize = neighborhoodSize;
        }

        // GaussianBlur kernel size should be positive and odd
        kernelSize = max(1, kernelSize);
        kernelSize |= 1; // Ensure it's odd

        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        int count = 0;

        for (int j = y - kernelSize / 2; j <= y + kernelSize / 2; ++j) {
            for (int i = x - kernelSize / 2; i <= x + kernelSize / 2; ++i) {
                if (i >= 0 && i < cols && j >= 0 && j < rows) {
                    uchar3 pixel = src[j * cols + i];
                    sum.x += pixel.x;
                    sum.y += pixel.y;
                    sum.z += pixel.z;
                    ++count;
                }
            }
        }

        sum.x /= count;
        sum.y /= count;
        sum.z /= count;

        dst[y * cols + x] = make_uchar3(static_cast<unsigned char>(sum.x), static_cast<unsigned char>(sum.y), static_cast<unsigned char>(sum.z));
    }
}

void processCUDA(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int neighborhoodSize, double factorRatio) {
    // Define block and grid dimensions
    dim3 blockSize(32, 32);
    dim3 gridSize((src.cols + blockSize.x - 1) / blockSize.x, (src.rows + blockSize.y - 1) / blockSize.y);

    // Allocate memory for covariance matrix on device
    cv::cuda::GpuMat covarianceDev(src.rows, src.cols * 3, CV_32F);

    // Perform covariance calculation on GPU
    calculateCovarianceMatrixKernel<<<gridSize, blockSize>>>(
        reinterpret_cast<uchar3*>(const_cast<unsigned char*>(src.ptr())), 
        reinterpret_cast<float*>(covarianceDev.ptr()), 
        src.cols, src.rows, neighborhoodSize);

    // Perform denoising on GPU
    denoiseByCovarianceKernel<<<gridSize, blockSize>>>(
        reinterpret_cast<uchar3*>(const_cast<unsigned char*>(src.ptr())), 
        reinterpret_cast<uchar3*>(dst.ptr()), 
        reinterpret_cast<float*>(covarianceDev.ptr()), 
        src.cols, src.rows, neighborhoodSize, factorRatio);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <image_path> <neighborhood_size> <factor_ratio>" << endl;
        return -1;
    }

    // Read the stereo image
    cv::Mat stereo_image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Check if the image is loaded successfully
    if (stereo_image.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    // Neighborhood size for covariance matrix
    int neighborhoodSize = atoi(argv[2]);
    // Factor ratio applied to determine the Gaussian kernel size
    double factorRatio = atof(argv[3]);

    // Convert input image to uchar3
    cv::cuda::GpuMat d_stereo_image;
    d_stereo_image.upload(stereo_image);

    // Allocate memory for output image on device
    cv::cuda::GpuMat d_denoised_image(d_stereo_image.size(), d_stereo_image.type());

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 1;

    // Apply denoising
    cv::Mat denoisedImage;
    for (int it = 0; it < iter; it++) {
        processCUDA(d_stereo_image, d_denoised_image, neighborhoodSize, factorRatio);
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time difference
    chrono::duration<double> diff = end - begin;

    // Download the result back to CPU
    cv::Mat dst;
    d_denoised_image.download(dst);

    // Display performance metrics
    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // Display the original and processed images
    cv::imshow("Original Image", stereo_image);
    cv::imshow("Denoised Image", dst);
    cv::waitKey();

    return 0;
}
