#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
using namespace std;
using namespace cv;

void image_convolution(vector<vector<int>> kernel, Mat image) {
    if (image.empty()) {
        cout << "Error: Could not read the image file!" << endl;
        return;
    }
    if (image.channels() == 3) {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }
    Mat output_image(image.rows, image.cols, CV_8UC1);

    double start_time = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int x = 1; x < image.rows - 1; x++) {
        for (int y = 1; y < image.cols - 1; y++) {
            int new_value = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = image.at<uchar>(x + i, y + j);
                    new_value += pixel * kernel[i + 1][j + 1];
                }
            }
            output_image.at<uchar>(x, y) = static_cast<uchar>(clamp(new_value, 0, 255));
        }
    }

    double end_time = omp_get_wtime();
    double elapsed_time = (end_time - start_time) * 1000; //milliseconds
    cout << "Parallel execution time: " << elapsed_time << " ms" << endl;

    imwrite("../img/output_par.jpg", output_image);
}

int main() {
    Mat image = imread("../img/norway.jpg");
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    image_convolution(kernel, image);
    return 0;
}
