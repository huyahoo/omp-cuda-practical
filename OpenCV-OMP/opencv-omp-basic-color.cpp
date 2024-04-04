#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>  // for high_resolution_clock

using namespace std;

int main( int argc, char** argv )
{
    // Read the source image
    cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);

    // Create a destination image with the same size as the source image
    cv::Mat_<cv::Vec3b> destination(source.rows, source.cols);

    // Display the source image
    cv::imshow("Source Image", source);

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 500;

    // Perform the operation iter times
    for (int it = 0; it < iter; it++) {
        // Parallelize the outer loop using OpenMP
        #pragma omp parallel for
        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                for (int c = 0; c < 3; c++) {
                    // Apply the pixel-wise operation (cosine function) to each color channel
                    destination(i, j)[c] = 255.0 * cos((255 - source(i, j)[c]) / 255.0);
                }
            }
        }
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time difference
    std::chrono::duration<double> diff = end - begin;

    // Display the processed image
    cv::imshow("Processed Image", destination);

    // Display performance metrics
    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // Wait for a key press before closing the windows
    cv::waitKey();

    return 0;
}
