#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include "imageConvolution.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Status;
using namespace cv;
using namespace std;

class ConvolutionClient {
public:
    ConvolutionClient(shared_ptr<Channel> channel): stub_(imageconv::ImageConvolution::NewStub(channel)) {}
    
    bool StreamImageConvolution(const string& image_path, const vector<vector<int>>& kernel, int num_chunks = 4) {
        ClientContext context;
        shared_ptr<ClientReaderWriter<imageconv::ImageChunk, imageconv::ConvolutionResult>> 
        stream(stub_->StreamConvolution(&context));
        
        Mat image = imread(image_path, IMREAD_GRAYSCALE);
        if (image.empty()) {
            cerr << "Failed to read image: " << image_path << endl;
            return false;
        }
        cout << "Image size: " << image.rows << "x" << image.cols << endl;
        int chunk_height = image.rows / num_chunks;
        vector<Mat> processed_chunks(num_chunks);
        
        thread writer([&]() {
            for (int i = 0; i < num_chunks; i++) {
                int start_row = i * chunk_height;
                int num_rows = (i == num_chunks - 1) ? 
                    (image.rows - start_row) : chunk_height;
                
                int halo_top = (i > 0) ? 1 : 0;
                int halo_bottom = (i < num_chunks - 1) ? 1 : 0;
                
                int chunk_start = max(0, start_row - halo_top);
                int chunk_end = min(image.rows, start_row + num_rows + halo_bottom);
                int chunk_rows = chunk_end - chunk_start;
                
                Mat chunk = image(Rect(0, chunk_start, image.cols, chunk_rows)).clone();
                
                vector<uint8_t> encoded_chunk;
                imencode(".jpg", chunk, encoded_chunk);
                
                imageconv::ImageChunk chunk_msg;
                chunk_msg.set_chunk_data(encoded_chunk.data(), encoded_chunk.size());
                chunk_msg.set_chunk_index(i);
                chunk_msg.set_total_chunks(num_chunks);
                chunk_msg.set_start_row(start_row);
                chunk_msg.set_num_rows(num_rows);
                chunk_msg.set_img_width(image.cols);
                chunk_msg.set_img_height(image.rows);
                chunk_msg.set_has_halo_top(halo_top > 0);
                chunk_msg.set_has_halo_bottom(halo_bottom > 0);
                
                for (int r = 0; r < 3; r++) {
                    for (int c = 0; c < 3; c++) {
                        chunk_msg.add_kernel(kernel[r][c]);
                    }
                }
                if (!stream->Write(chunk_msg)) {
                    break;
                }
                cout << "Sent chunk " << i + 1 << " of " << num_chunks << endl;
            }
            stream->WritesDone();
        });
        imageconv::ConvolutionResult result;
        while (stream->Read(&result)) {
            if (result.success()) {
                vector<uint8_t> buffer(result.result_data().begin(), result.result_data().end());
                Mat chunk_result = imdecode(buffer, IMREAD_GRAYSCALE);
                processed_chunks[result.chunk_index()] = chunk_result;
                
                cout << "Received processed chunk " << result.chunk_index() << endl;
            } else {
                cerr << "Error processing chunk " << result.chunk_index() << ": " << result.error_message() << endl;
            }
        }
        writer.join();
        Status status = stream->Finish();
        if (!status.ok()) {
            cerr << "RPC failed: " << status.error_message() << endl;
            return false;
        }
        Mat final_image = combineChunks(processed_chunks, image.cols);
        imwrite("../output/convolved_image.jpg", final_image);
        cout << "Convolution complete! Output saved." << endl;
        return true;
    }
    
private:
    unique_ptr<imageconv::ImageConvolution::Stub> stub_;
    Mat combineChunks(const vector<Mat>& chunks, int width) {
        int total_height = 0;
        for (const auto& chunk : chunks) {
            if (!chunk.empty()) {
                total_height += chunk.rows;
            }
        }
        Mat result(total_height, width, CV_8UC1);
        int current_row = 0;
        for (const auto& chunk : chunks) {
            if (!chunk.empty()) {
                chunk.copyTo(result(Rect(0, current_row, chunk.cols, chunk.rows)));
                current_row += chunk.rows;
            }
        }
        return result;
    }
};

int main(int argc, char** argv) {
    string server_address("localhost:50051");
    string image_path = "../../../data/img/0.jpg";
    if (argc > 1) {
        image_path = argv[1];
    }
    ConvolutionClient client(grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    client.StreamImageConvolution(image_path, kernel, 4);
    return 0;
}