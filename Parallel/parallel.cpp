#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <algorithm>
using namespace std;
using namespace cv;

void image_convolution(vector<vector<int>> kernel, Mat image, int stride)
{
    if (image.empty())
    {
        cout << "Error: Could not read the image file!" << endl;
        return;
    }
    if (image.channels() == 3)
    {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }
    Mat output_image(image.rows, image.cols, CV_8UC1);

    double start_time = omp_get_wtime();

#pragma omp parallel for collapse(2)
    for (int x = 1; x < image.rows - 1; x += stride)
    {
        for (int y = 1; y < image.cols - 1; y += stride)
        {
            int new_value = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    int pixel = image.at<uchar>(x + i, y + j);
                    new_value += pixel * kernel[i + 1][j + 1];
                }
            }
            // output_image.at<uchar>(x, y) = static_cast<uchar>(clamp(new_value, 0, 255));
            output_image.at<uchar>(x, y) = static_cast<uchar>(std::min(255, std::max(0, new_value)));
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = (end_time - start_time) * 1000; // milliseconds
    cout << "Parallel execution time: " << elapsed_time << " ms" << endl;

    imwrite("../img/output_par.jpg", output_image);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Stride argument missing!" << endl;
        return 1;
    }
    Mat image = imread("../img/norway.jpg");
    if (image.empty())
    {
        cout << "Error: Could not read the image file!" << endl;
        return 1;
    }
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    int stride = atoi(argv[1]);
    cout << "For stride = " << stride << endl;
    image_convolution(kernel, image, stride);
    return 0;
}
