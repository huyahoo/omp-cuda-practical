#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <cmath>
#include <chrono>  // for high_resolution_clock
#include <omp.h> // OpenMP header

using namespace std;

enum AnaglyphType {
    TRUE = 0,
    GRAY,
    COLOR,
    HALFCOLOR,
    OPTIMIZED
};

// Function to generate a 2D Gaussian kernel
cv::Mat generateGaussianKernel(int kernel_size, double sigma) {
    cv::Mat kernel(kernel_size, kernel_size, CV_64F);

    double sum = 0.0;
    int half_kernel = kernel_size / 2;

    for (int x = -half_kernel; x <= half_kernel; x++) {
        for (int y = -half_kernel; y <= half_kernel; y++) {
            double value = exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel.at<double>(x + half_kernel, y + half_kernel) = value;
            sum += value;
        }
    }

    // Normalize the kernel
    kernel /= sum;

    return kernel;
}

// Function to apply Gaussian filter to an image
cv::Mat applyGaussianFilter(const cv::Mat& input_image, const cv::Mat& kernel) {
    cv::Mat filtered_image;
    cv::filter2D(input_image, filtered_image, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    return filtered_image;
}

int main( int argc, char** argv )
{
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

    // Display the left and right images
    // cv::imshow("Left Image", left_image);
    // cv::imshow("Right Image", right_image);

    // Create an empty anaglyph image with the same size as the left and right images
    cv::Mat anaglyph_image(left_image.size(), CV_8UC3);

    std::string anaglyph_name;

    // Start the timer
    auto begin = chrono::high_resolution_clock::now();

    // Number of iterations
    const int iter = 500;

    // Perform the operation iter times
    for (int it = 0; it < iter; it++) {
        // Parallelize the outer loop using OpenMP
        #pragma omp parallel for
        // Loop through each pixel in the left image
        for (int i = 0; i < left_image.rows; i++) {
            for (int j = 0; j < left_image.cols; j++) {
                // Get the color channels for the left and right pixels
                cv::Vec3b left_pixel = left_image.at<cv::Vec3b>(i, j);
                cv::Vec3b right_pixel = right_image.at<cv::Vec3b>(i, j);

                switch (anaglyph_type) {
                    case TRUE:
                        // True Anaglyphs
                        anaglyph_name = "True";
                        anaglyph_image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                            0.299 * right_pixel[2] + 0.578 * right_pixel[1] + 0.114 * right_pixel[0],
                            0,
                            0.299 * left_pixel[2] + 0.578 * left_pixel[1] + 0.114 * left_pixel[0]
                        );
                        break;
                    case GRAY:
                        // Gray Anaglyphs
                        anaglyph_name = "Gray";
                        anaglyph_image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                            0.299 * right_pixel[2] + 0.578 * right_pixel[1] + 0.114 * right_pixel[0],
                            0.299 * right_pixel[2] + 0.578 * right_pixel[1] + 0.114 * right_pixel[0],
                            0.299 * left_pixel[2] + 0.578 * left_pixel[1] + 0.114 * left_pixel[0]
                        );
                        break;
                    case COLOR:
                        // Color Anaglyphs
                        anaglyph_name = "Color";
                        anaglyph_image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                            right_pixel[2],
                            right_pixel[1],
                            left_pixel[0]
                        );
                        break;
                    case HALFCOLOR:
                        // Half Color Anaglyphs
                        anaglyph_name = "Half Color";
                        anaglyph_image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                            0.299* right_pixel[2] + 0.578 * right_pixel[1] + 0.114 * right_pixel[0],
                            right_pixel[1],
                            left_pixel[0]
                        );
                        break;
                    case OPTIMIZED:
                        // Optimized Anaglyphs
                        anaglyph_name = "Optimized";
                        anaglyph_image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                            0.7 * right_pixel[1] + 0.3 * right_pixel[0],
                            right_pixel[1],
                            left_pixel[0]
                        );
                        break;
                    default:
                        #pragma omp critical
                        {
                            cerr << "Error: Invalid anaglyph type." << endl;
                        }
                }
            }
        }
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time difference
    std::chrono::duration<double> diff = end - begin;

    // Display the anaglyph image
    cv::imshow(anaglyph_name + " Anaglyph Image", anaglyph_image);

    // Save the anaglyph image
    std::string filename =  "output/open-omp/" + anaglyph_name + "Anaglyph.jpg";
    cv::imwrite(filename, anaglyph_image);

    // Display performance metrics
    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    // Wait for a key press before closing the windows
    cv::waitKey();

    return 0;
}