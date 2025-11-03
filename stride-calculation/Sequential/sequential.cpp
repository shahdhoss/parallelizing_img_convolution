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
    cout << "Original Image Size: " << image.rows << " x " << image.cols << endl;
    int K = kernel.size(); // Kernel size
    if (image.channels() == 3)
        cvtColor(image, image, COLOR_BGR2GRAY);

    int out_rows = ((image.rows - K) / stride) + 1;
    int out_cols = ((image.cols - K) / stride) + 1;
    Mat output_image(out_rows, out_cols, CV_8UC1);

    double start_time = omp_get_wtime();

    for (int x = 1; x < out_rows - 1; x++)
    {
        for (int y = 1; y < out_cols - 1; y++)
        {
            // The offset is +1 because your original 3 X 3  kernel loops started at x = 1 and y = 1 (to avoid accessing pixels outside the image bounds).int x_in = x * stride + 1;
            int x_in = x * stride + 1;
            int y_in = y * stride + 1;
            int new_value = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    int pixel = image.at<uchar>(x_in + i, y_in + j);
                    new_value += pixel * kernel[i + 1][j + 1];
                }
            }
            // output_image.at<uchar>(x, y) = static_cast<uchar>(clamp(new_value, 0, 255));
            output_image.at<uchar>(x, y) = static_cast<uchar>(std::min(255, std::max(0, new_value)));
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = (end_time - start_time) * 1000; // milliseconds
    cout << "Output Image Size: " << output_image.rows << " x " << output_image.cols << endl;
    cout << "Sequential execution time: " << elapsed_time << " ms" << endl;
    imwrite("../img/output_seq.jpg", output_image);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << "Stride argument missing!" << endl;
        return 1;
    }
    Mat image = imread("../img/norway.jpg");
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    int stride = atoi(argv[1]);
    cout << "For stride = " << stride << endl;
    image_convolution(kernel, image, stride);
    return 0;
}
