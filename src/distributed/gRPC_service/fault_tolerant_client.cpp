#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <fstream>
#include <mutex>
#include <atomic>
#include "imageConvolution.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReaderWriter;
using grpc::Status;
using namespace cv;
using namespace std;
using namespace std::chrono;

// Logger class for tracking requests and failures
class RequestLogger {
private:
    ofstream log_file;
    mutex log_mutex;
    
public:
    RequestLogger(const string& filename) {
        log_file.open(filename, ios::app);
        log_file << "\n=== New Session: " << getCurrentTimestamp() << " ===\n";
    }
    
    ~RequestLogger() {
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
    
    void logRequest(int request_id, const string& server_address) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] Request #" << request_id 
                 << " sent to " << server_address << endl;
        log_file.flush();
    }
    
    void logResponse(int request_id, double latency_ms, bool success) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] Request #" << request_id 
                 << " completed - Latency: " << latency_ms << "ms - Success: " 
                 << (success ? "YES" : "NO") << endl;
        log_file.flush();
    }
    
    void logFailure(int request_id, const string& error_msg, const string& server) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] FAILURE - Request #" << request_id 
                 << " on " << server << " - Error: " << error_msg << endl;
        log_file.flush();
    }
    
    void logRetry(int request_id, int attempt, const string& new_server) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] RETRY - Request #" << request_id 
                 << " - Attempt " << attempt << " - Switching to " << new_server << endl;
        log_file.flush();
    }
    
    void logRecovery(const string& message) {
        lock_guard<mutex> lock(log_mutex);
        log_file << "[" << getCurrentTimestamp() << "] RECOVERY - " << message << endl;
        log_file.flush();
    }
};

// Fault-tolerant client with replica support and auto-retry
class FaultTolerantConvolutionClient {
private:
    vector<string> server_addresses_;
    vector<unique_ptr<imageconv::ImageConvolution::Stub>> stubs_;
    atomic<int> current_server_idx_;
    RequestLogger logger_;
    const int MAX_RETRIES = 3;
    const int TIMEOUT_MS = 5000;  // 5 second timeout
    
public:
    FaultTolerantConvolutionClient(vector<string> server_addresses) 
        : server_addresses_(server_addresses), 
          current_server_idx_(0),
          logger_("client_fault_tolerance.log") {
        
        // Create channels for all replicas
        for (const auto& addr : server_addresses_) {
            auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
            stubs_.push_back(imageconv::ImageConvolution::NewStub(channel));
            cout << "Connected to replica: " << addr << endl;
        }
    }
    
    // Try to process image with automatic failover
    bool StreamImageConvolutionWithRetry(const string& image_path, 
                                         const vector<vector<int>>& kernel, 
                                         int request_id,
                                         int num_chunks = 4) {
        
        for (int retry = 0; retry < MAX_RETRIES; retry++) {
            int server_idx = current_server_idx_ % server_addresses_.size();
            string current_server = server_addresses_[server_idx];
            
            if (retry > 0) {
                logger_.logRetry(request_id, retry + 1, current_server);
                cout << "\n[RETRY] Attempt " << (retry + 1) << " using " << current_server << endl;
            }
            
            auto start_time = high_resolution_clock::now();
            logger_.logRequest(request_id, current_server);
            
            bool success = attemptConvolution(server_idx, image_path, kernel, 
                                             request_id, num_chunks);
            
            auto end_time = high_resolution_clock::now();
            double latency = duration_cast<milliseconds>(end_time - start_time).count();
            
            if (success) {
                logger_.logResponse(request_id, latency, true);
                cout << "[SUCCESS] Request #" << request_id << " completed in " 
                     << latency << "ms" << endl;
                return true;
            } else {
                logger_.logResponse(request_id, latency, false);
                logger_.logFailure(request_id, "Stream failed", current_server);
                
                // Switch to next replica
                current_server_idx_++;
                
                // If we've tried all servers once, wait before retrying
                if ((retry + 1) % server_addresses_.size() == 0 && retry + 1 < MAX_RETRIES) {
                    cout << "[WAITING] Cooling down before next retry cycle..." << endl;
                    this_thread::sleep_for(milliseconds(1000));
                }
            }
        }
        
        cout << "[FAILED] Request #" << request_id << " failed after " 
             << MAX_RETRIES << " attempts" << endl;
        return false;
    }
    
private:
    bool attemptConvolution(int server_idx, const string& image_path, 
                           const vector<vector<int>>& kernel,
                           int request_id, int num_chunks) {
        try {
            ClientContext context;
            
            // Set deadline for the RPC
            context.set_deadline(system_clock::now() + milliseconds(TIMEOUT_MS));
            
            shared_ptr<ClientReaderWriter<imageconv::ImageChunk, imageconv::ConvolutionResult>> 
                stream(stubs_[server_idx]->StreamConvolution(&context));
            
            Mat image = imread(image_path, IMREAD_GRAYSCALE);
            if (image.empty()) {
                cerr << "[ERROR] Failed to read image: " << image_path << endl;
                return false;
            }
            
            int chunk_height = image.rows / num_chunks;
            vector<Mat> processed_chunks(num_chunks);
            atomic<int> chunks_received(0);
            
            // Writer thread
            thread writer([&]() {
                try {
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
                            cerr << "[ERROR] Failed to write chunk " << i << endl;
                            break;
                        }
                    }
                    stream->WritesDone();
                } catch (const exception& e) {
                    cerr << "[ERROR] Writer exception: " << e.what() << endl;
                }
            });
            
            // Reader thread (main thread reads)
            imageconv::ConvolutionResult result;
            while (stream->Read(&result)) {
                if (result.success()) {
                    vector<uint8_t> buffer(result.result_data().begin(), 
                                          result.result_data().end());
                    Mat chunk_result = imdecode(buffer, IMREAD_GRAYSCALE);
                    processed_chunks[result.chunk_index()] = chunk_result;
                    chunks_received++;
                } else {
                    cerr << "[ERROR] Chunk " << result.chunk_index() 
                         << " processing failed: " << result.error_message() << endl;
                }
            }
            
            writer.join();
            
            Status status = stream->Finish();
            if (!status.ok()) {
                cerr << "[ERROR] RPC failed: " << status.error_message() << endl;
                return false;
            }
            
            // Check if we got all chunks
            if (chunks_received < num_chunks) {
                cerr << "[ERROR] Only received " << chunks_received 
                     << " of " << num_chunks << " chunks" << endl;
                return false;
            }
            
            // Combine and save result
            Mat final_image = combineChunks(processed_chunks, image.cols);
            string output_path = "../output/request_" + to_string(request_id) + ".jpg";
            imwrite(output_path, final_image);
            
            return true;
            
        } catch (const exception& e) {
            cerr << "[EXCEPTION] " << e.what() << endl;
            return false;
        }
    }
    
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

// Load generator - sends continuous requests
class LoadGenerator {
private:
    FaultTolerantConvolutionClient& client_;
    string image_path_;
    vector<vector<int>> kernel_;
    int duration_seconds_;
    int delay_ms_;  // Delay between requests in milliseconds
    atomic<int> total_requests_;
    atomic<int> successful_requests_;
    atomic<int> failed_requests_;
    
public:
    LoadGenerator(FaultTolerantConvolutionClient& client, 
                  const string& image_path,
                  const vector<vector<int>>& kernel,
                  int duration_seconds,
                  int delay_ms = 500)  // Default: 500ms = ~2 req/sec
        : client_(client), 
          image_path_(image_path), 
          kernel_(kernel),
          duration_seconds_(duration_seconds),
          delay_ms_(delay_ms),
          total_requests_(0),
          successful_requests_(0),
          failed_requests_(0) {}
    
    void run() {
        double expected_rate = (delay_ms_ > 0) ? (1000.0 / delay_ms_) : 999;
        cout << "\n=== Starting Load Generator for " << duration_seconds_ << " seconds ===" << endl;
        cout << "Request rate: ~" << expected_rate << " req/sec (delay: " << delay_ms_ << "ms)" << endl;
        cout << "Press Ctrl+C to inject failures manually during the run\n" << endl;
        
        auto start_time = high_resolution_clock::now();
        auto end_time = start_time + seconds(duration_seconds_);
        
        int request_id = 1;
        while (high_resolution_clock::now() < end_time) {
            cout << "\n--- Request #" << request_id << " ---" << endl;
            total_requests_++;
            
            bool success = client_.StreamImageConvolutionWithRetry(
                image_path_, kernel_, request_id, 4);
            
            if (success) {
                successful_requests_++;
            } else {
                failed_requests_++;
            }
            
            request_id++;
            
            // Delay between requests (configurable via delay_ms_)
            if (delay_ms_ > 0) {
                this_thread::sleep_for(milliseconds(delay_ms_));
            }
        }
        
        printStatistics();
    }
    
    void printStatistics() {
        cout << "\n=== Load Generator Statistics ===" << endl;
        cout << "Total Requests: " << total_requests_ << endl;
        cout << "Successful: " << successful_requests_ << endl;
        cout << "Failed: " << failed_requests_ << endl;
        if (total_requests_ > 0) {
            cout << "Success Rate: " << (100.0 * successful_requests_ / total_requests_) << "%" << endl;
        }
        double actual_rate = total_requests_ / (double)duration_seconds_;
        cout << "Actual Throughput: " << actual_rate << " req/sec" << endl;
    }
};

int main(int argc, char** argv) {
    // Configure replica addresses
    vector<string> server_addresses = {
        "localhost:50051",  // Replica 1
        "localhost:50052"   // Replica 2
    };
    
    string image_path = "../../../data/img/0.jpg";
    int duration = 60;     // 60 seconds by default
    int delay_ms = 500;    // 500ms delay = ~2 req/sec
    
    if (argc > 1) {
        image_path = argv[1];
    }
    if (argc > 2) {
        duration = stoi(argv[2]);
    }
    if (argc > 3) {
        delay_ms = stoi(argv[3]);  // Optional: request delay in milliseconds
    }
    
    cout << "=== Fault-Tolerant Image Convolution Client ===" << endl;
    cout << "Image: " << image_path << endl;
    cout << "Server replicas: " << endl;
    for (const auto& addr : server_addresses) {
        cout << "  - " << addr << endl;
    }
    cout << "Duration: " << duration << " seconds" << endl;
    cout << "Request delay: " << delay_ms << "ms" << endl;
    
    // Create fault-tolerant client
    FaultTolerantConvolutionClient client(server_addresses);
    
    // Define sharpening kernel
    vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    
    // Run load generator with configurable rate
    LoadGenerator generator(client, image_path, kernel, duration, delay_ms);
    generator.run();
    
    return 0;
}
