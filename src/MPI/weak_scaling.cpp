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

    Mat image;
    int cols = 0;

    if (rank == 0)
    {
        image = imread("../../data/img/1.jpg");
        if (image.empty())
        {
            cout << "Error: Could not read the image file!" << endl;
            return 1;
        }
        if (image.channels() == 3)
            cvtColor(image, image, COLOR_BGR2GRAY);
        cols = image.cols;
    }
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_rank = 110;
    int total_rows = rows_per_rank * size;
    int send_count = rows_per_rank * cols;
    int halo = 1;

    Mat full_image;
    if (rank == 0)
    {
        if (image.rows < total_rows)
        {
            full_image = Mat(total_rows, cols, CV_8UC1);
            for (int i = 0; i < size; i++)
            {
                int start_row = i * rows_per_rank;
                int copy_rows = min(rows_per_rank, image.rows);
                image.rowRange(0, copy_rows).copyTo(full_image.rowRange(start_row, start_row + copy_rows));
            }
        }
        else
        {
            full_image = image.rowRange(0, total_rows).clone();
        }
    }
    vector<uchar> local_buffer((rows_per_rank + 2 * halo) * cols);
    MPI_Scatter(
        rank == 0 ? full_image.data : nullptr,
        send_count,
        MPI_UNSIGNED_CHAR,
        local_buffer.data() + halo * cols,
        send_count,
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD);

    MPI_Request reqs[4];
    int r = 0;
    double local_start = MPI_Wtime();
    if (rank > 0)
        MPI_Irecv(local_buffer.data(), cols, MPI_UNSIGNED_CHAR, rank - 1, 0, MPI_COMM_WORLD, &reqs[r++]);

    if (rank < size - 1)
        MPI_Irecv(local_buffer.data() + (rows_per_rank + halo) * cols, cols, MPI_UNSIGNED_CHAR, rank + 1, 1, MPI_COMM_WORLD, &reqs[r++]);

    if (rank > 0)
        MPI_Isend(local_buffer.data() + halo * cols, cols, MPI_UNSIGNED_CHAR, rank - 1, 1, MPI_COMM_WORLD, &reqs[r++]);

    if (rank < size - 1)
        MPI_Isend(local_buffer.data() + (rows_per_rank - 1 + halo) * cols, cols, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD, &reqs[r++]);

    MPI_Waitall(r, reqs, MPI_STATUS_IGNORE);

    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    Mat local_input(rows_per_rank + 2 * halo, cols, CV_8UC1, local_buffer.data());
    Mat local_output = image_convolution(kernel, local_input);

    double local_end = MPI_Wtime();
    cout << "Rank " << rank << " execution time: " << local_end - local_start << " seconds\n";
    // Send back only the interior rows (exclude halos)
    vector<uchar> send_back(send_count);
    memcpy(send_back.data(), local_output.data + halo * cols, send_count);

    vector<uchar> output_buffer;
    if (rank == 0)
        output_buffer.resize(total_rows * cols);

    double start = MPI_Wtime();
    MPI_Gather(send_back.data(), send_count, MPI_UNSIGNED_CHAR,
               rank == 0 ? output_buffer.data() : nullptr,
               send_count, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0)
    {
        cout << "Weak scaling execution time: " << end - start << " seconds." << endl;
        Mat final_output(total_rows, cols, CV_8UC1, output_buffer.data());
        imwrite("output_weak_scaling.jpg", final_output);
        cout << "Output image saved.\n";
    }

    MPI_Finalize();
    return 0;
}
