#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>
using namespace std;
using namespace cv;

Mat image_convolution(const vector<vector<int>> &kernel, const Mat &input_section)
{
    Mat output = input_section.clone();
    for (int x = 1; x < input_section.rows - 1; x++)
    {
        for (int y = 1; y < input_section.cols - 1; y++)
        {
            int sum = 0;
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++)
                    sum += input_section.at<uchar>(x + i, y + j) * kernel[i + 1][j + 1];
            output.at<uchar>(x, y) = saturate_cast<uchar>(sum);
        }
    }
    return output;
}

int main(int argc, char *argv[])
{
    Mat image = imread("../../data/img/" + to_string(1) + ".jpg");
    if (image.empty())
    {
        cout << "Error: Could not read the image file!" << endl;
        return 1;
    }
    if (image.channels() == 3)
        cvtColor(image, image, cv::COLOR_BGR2GRAY);
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    double start = MPI_Wtime();

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rows_per_rank = image.rows / size;
    int send_count = rows_per_rank * image.cols;
    uchar *input_ptr = image.data;
    vector<uchar> local_buffer(send_count);
    MPI_Scatter(input_ptr, send_count, MPI_UNSIGNED_CHAR, local_buffer.data(), send_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    Mat local_input(rows_per_rank, image.cols, CV_8UC1, local_buffer.data());
    Mat local_output = image_convolution(kernel, local_input);

    vector<uchar> output_buffer;
    if (rank == 0)
        output_buffer.resize(image.rows * image.cols);

    MPI_Gather(local_output.data, send_count, MPI_UNSIGNED_CHAR, output_buffer.data(), send_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        Mat final_output(image.rows, image.cols, CV_8UC1, output_buffer.data());
        imwrite("output.jpg", final_output);
        cout << "Image saved.\n";
        double end = MPI_Wtime();
        cout<< "Elapsed time = "<< end - start << " seconds"<<endl;
    }

    MPI_Finalize();
    return 0;
}