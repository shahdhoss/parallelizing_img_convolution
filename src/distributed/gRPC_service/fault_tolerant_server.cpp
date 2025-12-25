#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <mutex>
#include "imageConvolution.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;
using namespace cv;
using namespace std;
using namespace std::chrono;

// Fault injection configuration
struct FaultConfig {
    atomic<bool> enabled{false};
    atomic<bool> crash_mode{false};
    atomic<int> artificial_delay_ms{0};
    atomic<double> drop_probability{0.0};  // 0.0 to 1.0
    atomic<int> requests_before_crash{0};  // 0 = no crash
    mutex config_mutex;
};

FaultConfig g_fault_config;

// Logger for server events
class ServerLogger {
private:
    ofstream log_file;
    mutex log_mutex;
    string server_id;
    
public:
    ServerLogger(const string& filename, const string& id) : server_id(id) {
        log_file.open(filename, ios::app);
        log_file << "\n=== Server " << server_id << " Started: " 
                 << getCurrentTimestamp() << " ===\n";
    }
    
    ~ServerLogger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }
    
    string getCurrentTimestamp() {
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        auto timer = system_clock::to_time_t(now);
        tm bt;
        #ifdef _WIN32
            localtime_s(&bt, &timer);
        #else
            localtime_r(&timer, &bt);
        #endif
        
        char buffer[100];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &bt);
        return string(buffer) + "." + to_string(ms.count());
    }
    
    void logRequest(int chunk_index) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] [" << server_id 
                 << "] Received chunk " << chunk_index << endl;
        log_file.flush();
    }
    
    void logProcessed(int chunk_index, double processing_time_ms) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] [" << server_id 
                 << "] Processed chunk " << chunk_index 
                 << " in " << processing_time_ms << "ms" << endl;
        log_file.flush();
    }
    
    void logFaultInjection(const string& fault_type) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] [" << server_id 
                 << "] FAULT INJECTED: " << fault_type << endl;
        log_file.flush();
    }
    
    void logError(const string& error_msg) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] [" << server_id 
                 << "] ERROR: " << error_msg << endl;
        log_file.flush();
    }
};

class FaultTolerantConvolutionService final : public imageconv::ImageConvolution::Service {
private:
    ServerLogger logger_;
    atomic<int> requests_processed_;
    random_device rd_;
    mt19937 gen_;
    
public:
    FaultTolerantConvolutionService(const string& server_id) 
        : logger_("server_" + server_id + ".log", server_id),
          requests_processed_(0),
          gen_(rd_()) {}
    
    Status StreamConvolution(ServerContext* context, ServerReaderWriter<imageconv::ConvolutionResult, imageconv::ImageChunk>* stream) override { 
        
        imageconv::ImageChunk chunk;
        while (stream->Read(&chunk)) {
            requests_processed_++;
            logger_.logRequest(chunk.chunk_index());
            
            // Check for crash injection
            if (g_fault_config.crash_mode && 
                g_fault_config.requests_before_crash > 0 &&
                requests_processed_ >= g_fault_config.requests_before_crash) {
                
                logger_.logFaultInjection("SERVER CRASH - Terminating");
                cout << "\n[FAULT INJECTION] Simulating server crash!" << endl;
                exit(1);  // Simulate crash
            }
            
            // Check for packet drop
            if (g_fault_config.enabled && g_fault_config.drop_probability > 0) {
                uniform_real_distribution<> dis(0.0, 1.0);
                if (dis(gen_) < g_fault_config.drop_probability) {
                    logger_.logFaultInjection("Packet dropped for chunk " + 
                                             to_string(chunk.chunk_index()));
                    cout << "[FAULT] Dropped chunk " << chunk.chunk_index() << endl;
                    continue;  // Skip processing this chunk (simulate packet loss)
                }
            }
            
            // Artificial delay injection
            if (g_fault_config.enabled && g_fault_config.artificial_delay_ms > 0) {
                int delay = g_fault_config.artificial_delay_ms;
                logger_.logFaultInjection("Artificial delay: " + to_string(delay) + "ms");
                this_thread::sleep_for(milliseconds(delay));
            }
            
            imageconv::ConvolutionResult result;
            result.set_chunk_index(chunk.chunk_index());
            
            try {
                auto start = high_resolution_clock::now();
                
                // Extract kernel
                vector<vector<int>> kernel(3, vector<int>(3));
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        kernel[i][j] = chunk.kernel(i * 3 + j);
                    }
                }
                
                // Decode image chunk
                vector<uint8_t> buffer(chunk.chunk_data().begin(), chunk.chunk_data().end());
                Mat chunk_img = imdecode(buffer, IMREAD_GRAYSCALE);
                
                if (chunk_img.empty()) {
                    result.set_success(false);
                    result.set_error_message("Failed to decode image chunk");
                    logger_.logError("Failed to decode chunk " + to_string(chunk.chunk_index()));
                    stream->Write(result);
                    continue;
                }
                
                // Process chunk
                Mat output = processChunk(kernel, chunk_img, chunk.start_row(), chunk.num_rows());
                
                // Encode result
                vector<uint8_t> encoded_result;
                imencode(".jpg", output, encoded_result);
                result.set_result_data(encoded_result.data(), encoded_result.size());
                result.set_success(true);
                
                stream->Write(result);
                
                auto end = high_resolution_clock::now();
                double processing_time = duration_cast<milliseconds>(end - start).count();
                logger_.logProcessed(chunk.chunk_index(), processing_time);
                
                cout << "[OK] Processed chunk " << chunk.chunk_index() 
                     << " of " << chunk.total_chunks() << " (" 
                     << processing_time << "ms)" << endl;
                
            } catch (const exception& e) {
                result.set_success(false);
                result.set_error_message(e.what());
                logger_.logError(string("Exception: ") + e.what());
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

// Fault injection control thread
void faultInjectionController(const string& server_id) {
    cout << "\n=== Fault Injection Controller ===" << endl;
    cout << "Commands:" << endl;
    cout << "  1 - Inject 2 second delay" << endl;
    cout << "  2 - Enable 30% packet drop" << endl;
    cout << "  3 - Schedule crash after 5 requests" << endl;
    cout << "  4 - Clear all faults" << endl;
    cout << "  q - Quit server" << endl;
    cout << "================================\n" << endl;
    
    while (true) {
        string command;
        getline(cin, command);
        
        if (command == "1") {
            g_fault_config.enabled = true;
            g_fault_config.artificial_delay_ms = 2000;
            cout << "[FAULT] Injected 2 second delay per request" << endl;
        }
        else if (command == "2") {
            g_fault_config.enabled = true;
            g_fault_config.drop_probability = 0.3;
            cout << "[FAULT] Enabled 30% packet drop rate" << endl;
        }
        else if (command == "3") {
            g_fault_config.crash_mode = true;
            g_fault_config.requests_before_crash = 5;
            cout << "[FAULT] Server will crash after 5 requests" << endl;
        }
        else if (command == "4") {
            g_fault_config.enabled = false;
            g_fault_config.crash_mode = false;
            g_fault_config.artificial_delay_ms = 0;
            g_fault_config.drop_probability = 0.0;
            g_fault_config.requests_before_crash = 0;
            cout << "[CLEAR] All faults cleared" << endl;
        }
        else if (command == "q") {
            cout << "Shutting down server..." << endl;
            exit(0);
        }
    }
}

void RunServer(const string& server_address, const string& server_id) {
    FaultTolerantConvolutionService service(server_id);
    ServerBuilder builder;
    
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    unique_ptr<Server> server(builder.BuildAndStart());
    cout << "Server " << server_id << " listening on " << server_address << endl;
    cout << "Ready to handle requests\n" << endl;
    
    // Start fault injection controller in a separate thread
    thread controller_thread(faultInjectionController, server_id);
    controller_thread.detach();
    
    server->Wait();
}

int main(int argc, char** argv) {
    string server_address = "0.0.0.0:50051";
    string server_id = "1";
    
    if (argc > 1) {
        server_address = argv[1];
    }
    if (argc > 2) {
        server_id = argv[2];
    }
    
    cout << "=== Fault-Tolerant Image Convolution Server ===" << endl;
    cout << "Server ID: " << server_id << endl;
    cout << "Address: " << server_address << endl;
    
    RunServer(server_address, server_id);
    
    return 0;
}
