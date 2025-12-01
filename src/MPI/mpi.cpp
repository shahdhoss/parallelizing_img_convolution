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
    double start = MPI_Wtime();
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image;
    if (rank == 0)
    {
        image = imread("../../data/img/1.jpg");
        if (image.empty())
        {
            cout << "Error: Could not read the image file!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (image.channels() == 3)
            cvtColor(image, image, COLOR_BGR2GRAY);
    }
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    int rows = 0, cols = 0;
    if (rank == 0)
    {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_rank = rows / size;
    int send_count = rows_per_rank * cols;
    int halo = 1;

    // Buffer has halo rows at top and bottom
    vector<uchar> local_buffer((rows_per_rank + 2 * halo) * cols);

    // Scatter WITHOUT halos → put in the interior of local_buffer
    MPI_Scatter(
        (rank == 0 ? image.data : nullptr),
        send_count,
        MPI_UNSIGNED_CHAR,
        local_buffer.data() + halo * cols,
        send_count,
        MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    int num_requests = 4;
    if (argc > 1)
        num_requests = atoi(argv[1]);
    cout << "Using " << num_requests << " MPI requests for halo exchange." << endl;
    MPI_Request reqs[num_requests];
    int r = 0;

    // Receive top halo from previous rank
    if (rank > 0)
        MPI_Irecv(local_buffer.data(), cols, MPI_UNSIGNED_CHAR,
                  rank - 1, 0, MPI_COMM_WORLD, &reqs[r++]);

    // Receive bottom halo from next rank
    if (rank < size - 1)
        MPI_Irecv(local_buffer.data() + (rows_per_rank + halo) * cols,
                  cols, MPI_UNSIGNED_CHAR,
                  rank + 1, 1, MPI_COMM_WORLD, &reqs[r++]);

    // Send first interior row to previous rank
    if (rank > 0)
        MPI_Isend(local_buffer.data() + halo * cols, cols, MPI_UNSIGNED_CHAR, rank - 1, 1, MPI_COMM_WORLD, &reqs[r++]);

    // Send last interior row to next rank
    if (rank < size - 1)
        MPI_Isend(local_buffer.data() + (rows_per_rank - 1 + halo) * cols, cols, MPI_UNSIGNED_CHAR, rank + 1, 0, MPI_COMM_WORLD, &reqs[r++]);
    MPI_Waitall(r, reqs, MPI_STATUS_IGNORE);

    Mat local_input(rows_per_rank + 2 * halo, cols, CV_8UC1, local_buffer.data());
    Mat local_output = image_convolution(kernel, local_input);

    // Prepare data to send back (exclude halos) so bright or dark stripes near boundaries doesn't appear
    vector<uchar> send_back(send_count);
    memcpy(send_back.data(), local_output.data + halo * cols, send_count);

    vector<uchar> output_buffer;
    if (rank == 0)
        output_buffer.resize(rows * cols);

    MPI_Gather(send_back.data(), send_count, MPI_UNSIGNED_CHAR, (rank == 0 ? output_buffer.data() : nullptr), send_count, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        Mat final_output(rows, cols, CV_8UC1, output_buffer.data());
        imwrite("output.jpg", final_output);
        cout << "Image saved.\n";
        double end = MPI_Wtime();
        cout<< "Elapsed time = "<< end - start << " seconds"<<endl;
    }

    MPI_Finalize();
    return 0;
}
