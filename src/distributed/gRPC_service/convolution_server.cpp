#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "imageConvolution.grpc.pb.h"


using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;
using namespace cv;
using namespace std;

class ConvolutionServiceImpl final : public imageconv::ImageConvolution::Service {
public:
    Status StreamConvolution(ServerContext* context, ServerReaderWriter<imageconv::ConvolutionResult, imageconv::ImageChunk>* stream) override { 
        imageconv::ImageChunk chunk;
        while (stream->Read(&chunk)) {
            imageconv::ConvolutionResult result;
            result.set_chunk_index(chunk.chunk_index());
            try {
                vector<vector<int>> kernel(3, vector<int>(3));
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        kernel[i][j] = chunk.kernel(i * 3 + j);
                    }
                }
                vector<uint8_t> buffer(chunk.chunk_data().begin(), chunk.chunk_data().end());
                Mat chunk_img = imdecode(buffer, IMREAD_GRAYSCALE);
                if (chunk_img.empty()) {
                    result.set_success(false);
                    result.set_error_message("Failed to decode image chunk");
                    stream->Write(result);
                    continue;
                }
                Mat output = processChunk(kernel, chunk_img, chunk.start_row(), chunk.num_rows());
                vector<uint8_t> encoded_result;
                imencode(".jpg", output, encoded_result);
                result.set_result_data(encoded_result.data(), encoded_result.size());
                result.set_success(true);
                stream->Write(result);
                cout << "Processed chunk " << chunk.chunk_index() << " of " << chunk.total_chunks() << endl;
                
            } catch (const exception& e) {
                result.set_success(false);
                result.set_error_message(e.what());
                stream->Write(result);
            }
        }
        
        return Status::OK;
    }
    
private:
    Mat processChunk(vector<vector<int>>& kernel, Mat& chunk_with_halo, int start_row, int num_rows) {
    Mat gray;
    if (chunk_with_halo.channels() == 3) {
        cvtColor(chunk_with_halo, gray, COLOR_BGR2GRAY);
    } else {
        gray = chunk_with_halo;
    }
    Mat output_image = Mat::zeros(num_rows, chunk_with_halo.cols, CV_8UC1);
    int halo_offset = (start_row > 0) ? 1 : 0;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < num_rows; y++) {
        for (int x = 0; x < chunk_with_halo.cols; x++) {
            int src_y = y + halo_offset;
            int src_x = x;
            if (src_y >= 1 && src_y < gray.rows - 1 && 
                src_x >= 1 && src_x < gray.cols - 1) {
                
                int sum = 0;
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        sum += gray.at<uchar>(src_y + i, src_x + j) * kernel[i + 1][j + 1];
                    }
                }
                output_image.at<uchar>(y, x) = saturate_cast<uchar>(sum);
            } else {
                output_image.at<uchar>(y, x) = gray.at<uchar>(src_y, src_x);
            }
        }
    }
    
    return output_image;
}
};

void RunServer() {
    string server_address("0.0.0.0:50051");
    ConvolutionServiceImpl service;
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    unique_ptr<Server> server(builder.BuildAndStart());
    cout << "Server listening on " << server_address << endl;
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}