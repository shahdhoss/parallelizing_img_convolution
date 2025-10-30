#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
using namespace std;
using namespace cv;

void image_convolution(vector<vector<int>> kernel, vector<Mat> images){
    vector<Mat> output_images(images.size());
    for (int i = 0; i < images.size(); i++)
    {
        if (images[i].empty()) {
            cout << "Error: Could not read the image file!" << endl;
            return;
        }
        if (images[i].channels() == 3)
            cvtColor(images[i], images[i], cv::COLOR_BGR2GRAY);
        output_images[i] = Mat(images[i].rows, images[i].cols, CV_8UC1);
    }

    double start_time = omp_get_wtime();
    for (int images_length = 0; images_length < images.size(); images_length++){
        for (int x = 1; x < images[images_length].rows - 1; x++){
            for (int y = 1; y < images[images_length].cols - 1; y++){
                int new_value = 0;
                for (int i = -1; i <= 1; i++){
                    for (int j = -1; j <= 1; j++){
                        int pixel = images[images_length].at<uchar>(x + i, y + j);
                        new_value += pixel * kernel[i + 1][j + 1];
                    }
                }
                output_images[images_length].at<uchar>(x, y) = static_cast<uchar>(clamp(new_value, 0, 255));
            }
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = (end_time - start_time) * 1000; // milliseconds
    cout << "Sequential execution time: " << elapsed_time << " ms" << endl;
    
    for(int i=0;i<images.size();i++){
        imwrite("../../output/sequential/"+to_string(i)+".jpg", output_images[i]);
    }
}

int main()
{
    vector<Mat> input_images;
    for(int i=1;i<=10;i++){
        input_images.push_back(imread("../../data/img/"+to_string(i)+".jpg"));
    }
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    image_convolution(kernel, input_images);
    return 0;
}
