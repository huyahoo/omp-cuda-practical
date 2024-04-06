#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <string>
#include <cmath>
#include <chrono>  // for high_resolution_clock

#include "helper_math.h"


using namespace std;

enum AnaglyphType {
    NORMAL = 0,
    TRUE,
    GRAY,
    COLOR,
    HALFCOLOR,
    OPTIMIZED
};

__global__ void processKernel(const cv::cuda::PtrStep<uchar3> left_image,
                              const cv::cuda::PtrStep<uchar3> right_image,
                              cv::cuda::PtrStep<uchar3> anaglyph_image,
                              int rows,
                              int cols,
                              int anaglyph_type) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        uchar3 left_pixel = left_image(y, x);
        uchar3 right_pixel = right_image(y, x);

        switch (anaglyph_type) {
            case TRUE:
                // True Anaglyphs
                anaglyph_image(y, x) = make_uchar3(
                    0.299f * right_pixel.z + 0.578f * right_pixel.y + 0.114f * right_pixel.x,
                    0,
                    0.299f * left_pixel.z + 0.578f * left_pixel.y + 0.114f * left_pixel.x
                );
                break;
            case GRAY:
                // Gray Anaglyphs
                anaglyph_image(y, x) = make_uchar3(
                    0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z,
                    0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z,
                    0.299f * left_pixel.x + 0.578f * left_pixel.y + 0.114f * left_pixel.z
                );
                break;
            case COLOR:
                // Color Anaglyphs
                anaglyph_image(y, x) = make_uchar3(
                    right_pixel.x,
                    right_pixel.y,
                    left_pixel.z
                );
                break;
            case HALFCOLOR:
                // Half Color Anaglyphs
                anaglyph_image(y, x) = make_uchar3(
                    0.299f * right_pixel.x + 0.578f * right_pixel.y + 0.114f * right_pixel.z,
                    right_pixel.y,
                    left_pixel.z
                );
                break;
            case OPTIMIZED:
                // Optimized Anaglyphs
                anaglyph_image(y, x) = make_uchar3(
                    0.7f * right_pixel.y + 0.3f * right_pixel.x,
                    right_pixel.y,
                    left_pixel.z
                );
                break;
            default:
                // No Anaglyphs
                anaglyph_image(y, x) = left_pixel;
        }
    }
}

dim3 findOptimalBlockSize(cv::cuda::GpuMat& d_left_image, cv::cuda::GpuMat& d_right_image, cv::cuda::GpuMat& d_anaglyph_image, int anaglyph_type) {
    int minBlockSize = 8;
    int maxBlockSize = 64;

    // Initialize the variables to store the optimal block size and the minimum time
    dim3 optimalBlockSize(minBlockSize, minBlockSize);
    double minTime = DBL_MAX;

    // Loop over the block sizes
    for (int blockSizeX = minBlockSize; blockSizeX <= maxBlockSize; blockSizeX += 8) {
        for (int blockSizeY = minBlockSize; blockSizeY <= maxBlockSize; blockSizeY += 8) {
            if (blockSizeX * blockSizeY > 1024) continue;

            // Calculate the grid size
            dim3 gridSize((d_left_image.cols + blockSizeX - 1) / blockSizeX, (d_left_image.rows + blockSizeY - 1) / blockSizeY);

            // Start the timer
            auto start = chrono::high_resolution_clock::now();

            // Launch the kernel
            processKernel<<<gridSize, dim3(blockSizeX, blockSizeY)>>>(d_left_image, d_right_image, d_anaglyph_image, d_left_image.rows, d_left_image.cols, anaglyph_type);

            // Synchronize device after kernel launch
            // cudaDeviceSynchronize();

            // Stop the timer
            auto end = chrono::high_resolution_clock::now();

            // Calculate the time difference
            chrono::duration<double> diff = end - start;

            // If the current time is less than the minimum time, update the optimal block size and the minimum time
            if (diff.count() < minTime) {
                optimalBlockSize = dim3(blockSizeX, blockSizeY);
                minTime = diff.count();
            }
        }
    }

    return optimalBlockSize;
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void processCUDA(const cv::cuda::GpuMat& d_left_image,
                 const cv::cuda::GpuMat& d_right_image,
                 cv::cuda::GpuMat& d_anaglyph_image,
                 int rows,
                 int cols,
                 int anaglyph_type) {
    const dim3 block(64, 32);

    const dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
    processKernel<<<grid, block>>>(d_left_image, d_right_image, d_anaglyph_image, rows, cols, anaglyph_type);
}

int main(int argc, char** argv) {
    cv::namedWindow("Original Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Processed Image", cv::WINDOW_OPENGL | cv::WINDOW_AUTOSIZE);

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <image_path> <anaglyph_type>" << endl;
        return -1;
    }

    // Read the stereo image
    cv::Mat stereo_image = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Determine the type of anaglyphs to generate
    AnaglyphType anaglyph_type = static_cast<AnaglyphType>(atoi(argv[2]));

    // Check if the image is loaded successfully
    if (stereo_image.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    if (anaglyph_type < NORMAL || anaglyph_type > OPTIMIZED) {
        cerr << "Error: Invalid anaglyph type." << endl;
        cerr << "Anaglyph types:" << endl;
        cerr << "0: None Anaglyphs" << endl;
        cerr << "1: True Anaglyphs" << endl;
        cerr << "2: Gray Anaglyphs" << endl;
        cerr << "3: Color Anaglyphs" << endl;
        cerr << "4: Half Color Anaglyphs" << endl;
        cerr << "5: Optimized Anaglyphs" << endl;
        return -1;
    }

    // Split the stereo image into left and right images
    cv::Mat left_image(stereo_image, cv::Rect(0, 0, stereo_image.cols / 2, stereo_image.rows));
    cv::Mat right_image(stereo_image, cv::Rect(stereo_image.cols / 2, 0, stereo_image.cols / 2, stereo_image.rows));

    // Create an empty anaglyph image with the same size as the left and right images
    cv::Mat anaglyph_image(left_image.size(), CV_8UC3);

    std::string anaglyph_name;
    switch (anaglyph_type) {
        case TRUE:
            anaglyph_name = "True";
            break;
        case GRAY:
            anaglyph_name = "Gray";
            break;
        case COLOR:
            anaglyph_name = "Color";
            break;
        case HALFCOLOR:
            anaglyph_name = "Half Color";
            break;
        case OPTIMIZED:
            anaglyph_name = "Optimized";
            break;
        default:
            anaglyph_name = "None";
    }

    cv::cuda::GpuMat d_left_image, d_right_image, d_anaglyph_image;

    d_left_image.upload(left_image);
    d_right_image.upload(right_image);

    // Create the GPU matrix once before the loop
    d_anaglyph_image.create(left_image.rows, left_image.cols, CV_8UC3);

    // dim3 optimalBlockSize = findOptimalBlockSize(d_left_image, d_right_image, d_anaglyph_image, anaglyph_type);
    // cerr << "Optimal block size: (" << optimalBlockSize.x << ", " << optimalBlockSize.y << ")" << endl;

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 500;

    // Perform the operation iter times
    for (int it = 0; it < iter; it++) {
        processCUDA(d_left_image, d_right_image, d_anaglyph_image, left_image.rows, left_image.cols, anaglyph_type);
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Download the result from the GPU once after the loop
    d_anaglyph_image.download(anaglyph_image);

    // Calculate the time difference
    std::chrono::duration<double> diff = end - begin;

    // Display the anaglyph image
    cv::imshow(anaglyph_name + " Anaglyph Image", anaglyph_image);

    // Display the original images
    cv::imshow("Input Image", stereo_image);

    // Save the anaglyph image
    // std::string filename = "output/2.1.1/" + anaglyph_name + "Anaglyph.jpg";
    // cv::imwrite(filename, anaglyph_image);

    // Display performance metrics
    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // Wait for a key press before closing the windows
    cv::waitKey();

    return 0;
}
