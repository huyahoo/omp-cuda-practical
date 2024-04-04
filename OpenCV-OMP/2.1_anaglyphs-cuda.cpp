#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <cuda_runtime.h>

using namespace std;

enum AnaglyphType {
    TRUE = 0,
    GRAY,
    COLOR,
    HALFCOLOR,
    OPTIMIZED
};

__global__ void computeAnaglyph(const cv::Vec3b* left_image_data, const cv::Vec3b* right_image_data,
                                cv::Vec3b* anaglyph_image_data, int rows, int cols, int anaglyph_type) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {
        cv::Vec3b left_pixel = left_image_data[i * cols + j];
        cv::Vec3b right_pixel = right_image_data[i * cols + j];
        cv::Vec3b result_pixel;

        switch (anaglyph_type) {
            case TRUE:
                // True Anaglyphs
                result_pixel[0] = 0.299 * left_pixel[2] + 0.578 * left_pixel[1] + 0.114 * left_pixel[0];
                result_pixel[1] = 0;
                result_pixel[2] = 0.299 * right_pixel[2] + 0.578 * right_pixel[1] + 0.114 * right_pixel[0];
                break;
            case GRAY:
                // Gray Anaglyphs
                result_pixel[0] = 0.299 * left_pixel[2] + 0.578 * left_pixel[1] + 0.114 * left_pixel[0];
                result_pixel[1] = 0.299 * right_pixel[2] + 0.578 * right_pixel[1] + 0.114 * right_pixel[0];
                result_pixel[2] = 0.299 * right_pixel[2] + 0.578 * right_pixel[1] + 0.114 * right_pixel[0];
                break;
            case COLOR:
                // Color Anaglyphs
                result_pixel[0] = left_pixel[2];
                result_pixel[1] = right_pixel[1];
                result_pixel[2] = right_pixel[0];
                break;
            case HALFCOLOR:
                // Half Color Anaglyphs
                result_pixel[0] = 0.299 * left_pixel[2] + 0.578 * left_pixel[1] + 0.114 * left_pixel[0];
                result_pixel[1] = right_pixel[1];
                result_pixel[2] = right_pixel[0];
                break;
            case OPTIMIZED:
                // Optimized Anaglyphs
                result_pixel[0] = 0.7 * left_pixel[1] + 0.3 * left_pixel[0];
                result_pixel[1] = right_pixel[1];
                result_pixel[2] = right_pixel[0];
                break;
        }

        anaglyph_image_data[i * cols + j] = result_pixel;
    }
}

int main( int argc, char** argv ) {
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

    if (anaglyph_type < TRUE || anaglyph_type > OPTIMIZED) {
        cerr << "Error: Invalid anaglyph type." << endl;
        cerr << "Anaglyph types:" << endl;
        cerr << "0: True Anaglyphs" << endl;
        cerr << "1: Gray Anaglyphs" << endl;
        cerr << "2: Color Anaglyphs" << endl;
        cerr << "3: Half Color Anaglyphs" << endl;
        cerr << "4: Optimized Anaglyphs" << endl;
        return -1;
    }

    // Split the stereo image into left and right images
    cv::Mat left_image(stereo_image, cv::Rect(0, 0, stereo_image.cols / 2, stereo_image.rows));
    cv::Mat right_image(stereo_image, cv::Rect(stereo_image.cols / 2, 0, stereo_image.cols / 2, stereo_image.rows));

    // Create an empty anaglyph image with the same size as the left and right images
    cv::Mat anaglyph_image(left_image.size(), CV_8UC3);

    // Allocate memory on the device
    cv::Vec3b *d_left_image, *d_right_image, *d_anaglyph_image;
    cudaMalloc(&d_left_image, left_image.rows * left_image.cols * sizeof(cv::Vec3b));
    cudaMalloc(&d_right_image, right_image.rows * right_image.cols * sizeof(cv::Vec3b));
    cudaMalloc(&d_anaglyph_image, left_image.rows * left_image.cols * sizeof(cv::Vec3b));

    // Copy input data from host to device memory
    cudaMemcpy(d_left_image, left_image.data, left_image.rows * left_image.cols * sizeof(cv::Vec3b), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right_image, right_image.data, right_image.rows * right_image.cols * sizeof(cv::Vec3b), cudaMemcpyHostToDevice);

    // Specify a reasonable block size
    dim3 blockSize(32, 32);
    // Calculate grid size to cover the whole image
    dim3 gridSize((left_image.cols + blockSize.x - 1) / blockSize.x, (left_image.rows + blockSize.y - 1) / blockSize.y);

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 500;

    // Perform the operation iter times
    for (int it = 0; it < iter; it++) {
        // Launch the kernel
        computeAnaglyph<<<gridSize, blockSize>>>(d_left_image, d_right_image, d_anaglyph_image, left_image.rows, left_image.cols, anaglyph_type);
        cudaDeviceSynchronize(); // Ensure the kernel is finished before continuing
    }

    // Stop the timer
    auto end = chrono::high_resolution_clock::now();

    // Calculate the time difference
    std::chrono::duration<double> diff = end - begin;

    // Copy output data from device to host
    cudaMemcpy(anaglyph_image.data, d_anaglyph_image, left_image.rows * left_image.cols * sizeof(cv::Vec3b), cudaMemcpyDeviceToHost);

    // Display the anaglyph image
    cv::imshow("Anaglyph Image", anaglyph_image);

    // Display performance metrics
    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // Free device memory
    cudaFree(d_left_image);
    cudaFree(d_right_image);
    cudaFree(d_anaglyph_image);

    // Wait for a key press before closing the window
    cv::waitKey();

    return 0;
}
