#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>
using namespace std;
using namespace cv;

Mat image_convolution(const vector<vector<int>> &kernel, const Mat &input)
{
    Mat output = input.clone();
    for (int x = 1; x < input.rows - 1; x++)
    {
        for (int y = 1; y < input.cols - 1; y++)
        {
            int sum = 0;
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++)
                    sum += input.at<uchar>(x + i, y + j) * kernel[i + 1][j + 1];
            output.at<uchar>(x, y) = saturate_cast<uchar>(sum);
        }
    }
    return output;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // No halos as every process has the full image and doesn't need to communicate
    Mat image = imread("../../data/img/1.jpg");
    if (image.empty())
    {
        if (rank == 0)
            cout << "Error: Could not read image!" << endl;
        MPI_Finalize();
        return 1;
    }

    if (image.channels() == 3)
        cvtColor(image, image, COLOR_BGR2GRAY);

    int rows = image.rows;
    int cols = image.cols;

    vector<vector<int>> kernel = {
        {0, 1, 0},
        {-1, 5, -1},
        {0, -1, 0}};

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    Mat local_output = image_convolution(kernel, image);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    cout << "Rank " << rank << " time: " << end - start << " seconds." << endl;

    if (rank == 0)
    {
        imwrite("output_weak.jpg", local_output);
        cout << "Image saved.\n";
    }

    MPI_Finalize();
    return 0;
}
