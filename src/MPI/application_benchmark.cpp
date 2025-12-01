#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <vector>
#include <iomanip>
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

    int iterations = 10; // Number of times to run the convolution for benchmarking
    if (argc > 1)
        iterations = atoi(argv[1]);

    // Load image
    Mat image;
    if (rank == 0)
    {
        image = imread("../../data/img/8.jpg");
        if (image.empty())
        {
            cout << "Error: Could not read the image file!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (image.channels() == 3)
            cvtColor(image, image, COLOR_BGR2GRAY);

        cout << "Image size: " << image.rows << " x " << image.cols << " pixels" << endl;
        cout << "MPI processes: " << size << endl;
        cout << "Benchmark iterations: " << iterations << endl;
    }

    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    
    // Broadcast image dimensions
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

    // Statistics collection
    double total_halo_time = 0;
    double total_compute_time = 0;
    double total_time = 0;

    // Halo exchange statistics
    int halo_msg_size = cols; // Size of one halo row in bytes

    // Run multiple iterations for accurate measurements
    for (int iter = 0; iter < iterations; iter++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        double iter_start = MPI_Wtime();

        // Buffer has halo rows at top and bottom
        vector<uchar> local_buffer((rows_per_rank + 2 * halo) * cols);

        // Scatter image data
        MPI_Scatter(
            (rank == 0 ? image.data : nullptr),
            send_count,
            MPI_UNSIGNED_CHAR,
            local_buffer.data() + halo * cols,
            send_count,
            MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);

        // === HALO EXCHANGE PHASE ===
        MPI_Request reqs[4];
        int r = 0;
        double halo_start = MPI_Wtime();

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
            MPI_Isend(local_buffer.data() + halo * cols, cols, MPI_UNSIGNED_CHAR,
                      rank - 1, 1, MPI_COMM_WORLD, &reqs[r++]);

        // Send last interior row to next rank
        if (rank < size - 1)
            MPI_Isend(local_buffer.data() + (rows_per_rank - 1 + halo) * cols,
                      cols, MPI_UNSIGNED_CHAR,
                      rank + 1, 0, MPI_COMM_WORLD, &reqs[r++]);

        MPI_Waitall(r, reqs, MPI_STATUS_IGNORE);
        double halo_end = MPI_Wtime();
        total_halo_time += (halo_end - halo_start);

        // Computations
        double compute_start = MPI_Wtime();
        Mat local_input(rows_per_rank + 2 * halo, cols, CV_8UC1, local_buffer.data());
        Mat local_output = image_convolution(kernel, local_input);
        double compute_end = MPI_Wtime();
        total_compute_time += (compute_end - compute_start);

        // Prepare data to send back (exclude halos)
        vector<uchar> send_back(send_count);
        memcpy(send_back.data(),
               local_output.data + halo * cols,
               send_count);

        vector<uchar> output_buffer;
        if (rank == 0)
            output_buffer.resize(rows * cols);

        // Gather results
        MPI_Gather(send_back.data(), send_count, MPI_UNSIGNED_CHAR,
                   (rank == 0 ? output_buffer.data() : nullptr),
                   send_count, MPI_UNSIGNED_CHAR,
                   0, MPI_COMM_WORLD);

        double iter_end = MPI_Wtime();
        total_time += (iter_end - iter_start);

        // Save output on last iteration
        if (iter == iterations - 1 && rank == 0)
        {
            Mat final_output(rows, cols, CV_8UC1, output_buffer.data());
            imwrite("output_benchmark.jpg", final_output);
        }
    }

    // Calculate averages
    double avg_halo = total_halo_time / iterations;
    double avg_compute = total_compute_time / iterations;
    double avg_total = total_time / iterations;

    // Calculate halo exchange metrics
    double halo_latency = avg_halo / 2.0; // Approximate one-way latency
    
    double halo_bandwidth = (avg_halo > 0) ? (halo_msg_size / halo_latency) / (1024.0 * 1024.0) : 0;

    // Gather statistics from all ranks
    vector<double> all_halo(size), all_compute(size), all_total(size);
    vector<double> all_latency(size), all_bandwidth(size);

    MPI_Gather(&avg_halo, 1, MPI_DOUBLE, all_halo.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&avg_compute, 1, MPI_DOUBLE, all_compute.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&avg_total, 1, MPI_DOUBLE, all_total.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&halo_latency, 1, MPI_DOUBLE, all_latency.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&halo_bandwidth, 1, MPI_DOUBLE, all_bandwidth.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "Average over " << iterations << " iterations" << endl;
        cout << fixed << setprecision(6);

        // Calculate max times (worst case across all ranks)
        double max_halo = *max_element(all_halo.begin(), all_halo.end());
        double max_compute = *max_element(all_compute.begin(), all_compute.end());
        double max_total = *max_element(all_total.begin(), all_total.end());
        double max_latency = *max_element(all_latency.begin(), all_latency.end());
        double max_bandwidth = *max_element(all_bandwidth.begin(), all_bandwidth.end());

        cout << "\nHalo Exchange Communication:" << endl;
        cout << "  Message size: " << halo_msg_size << " bytes (" << halo_msg_size / 1024.0 << " KB)" << endl;
        cout << "  Halo exchange time: " << max_halo << " s (" << max_halo * 1e6 << " µs)" << endl;
        cout << "  Latency (one-way): " << max_latency << " s (" << max_latency * 1e6 << " µs)" << endl;
        cout << "  Bandwidth: " << max_bandwidth << " MB/s" << endl;

        cout << "\nTiming Breakdown:" << endl;
        cout << "  Halo Exchange: " << max_halo << " s" << endl;
        cout << "  Computation: " << max_compute << " s" << endl;
        cout << "  Total Time: " << max_total << " s" << endl;

        double comm_percentage = (max_halo / max_total) * 100;
        double compute_percentage = (max_compute / max_total) * 100;
        
        cout << "\nPerformance Analysis:" << endl;
        cout << "  Communication Time: " << comm_percentage << "%" << endl;
        cout << "  Computation Time: " << compute_percentage << "%" << endl;

        cout << "Image saved to: output_benchmark.jpg" << endl;
    }

    MPI_Finalize();
    return 0;
}
