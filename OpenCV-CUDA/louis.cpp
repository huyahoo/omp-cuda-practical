#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <chrono>

using namespace std;

void createAnaglyphCUDA ( cv::cuda::GpuMat& d_Limg, cv::cuda::GpuMat& d_Rimg, int method, cv::cuda::GpuMat& d_result, int block_x, int block_y);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <method>" << std::endl;
        return -1;
    }

    int method = std::stoi(argv[1]);

    cv::Mat image = cv::imread("garden-stereo.jpg");
    cv::Mat leftImage = image(cv::Rect(0, 0, image.cols / 2, image.rows));
    cv::Mat rightImage = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows));

    if (leftImage.empty() || rightImage.empty()) {
        std::cout << "Failed to load input images!" << std::endl;
        return -1;
    }

    cv::Mat anaglyphImage;

    cv::cuda::GpuMat d_Limg, d_Rimg, d_result;

    // for (int block_x = 1; block_x <= 32; block_x *= 2){
        // for (int block_y = 1; block_y <= 32; block_y *= 2){
            auto begin = chrono::high_resolution_clock::now();
            for (int i = 0; i < 100; i++){
                d_Limg.upload(leftImage);
                d_Rimg.upload(rightImage);
                d_result.upload(leftImage);

                createAnaglyphCUDA(d_Limg, d_Rimg, method, d_result, 32, 8);

                d_result.download(anaglyphImage);

            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-begin;
            std::cout << "Time: " << diff.count() << " s" << std::endl;
            std::cout << "Time per frame: " << diff.count()/100 << " s" << std::endl;
            std::cout << "IPS: " << 100/diff.count() << std::endl;
            // std::cout << "Block_x: " << block_x << " Block_y: " << block_y << " Time: " << diff.count() << " s" << std::endl;
        // }
    // }

    

    cv::imshow("Anaglyph Image", anaglyphImage);

    cv::waitKey();

    return 0;
}