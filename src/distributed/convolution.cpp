#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <fstream>
using namespace std;
using namespace cv;

Mat image_chunk_convolution(vector<vector<int>> kernel, Mat chunk_with_halo, int start_row, int num_rows) {
    Mat gray;
    if (chunk_with_halo.empty()) {
        cout << "Error: Could not read the image file!" << endl;
    }
    if (chunk_with_halo.channels() == 3){
        cvtColor(chunk_with_halo, gray, COLOR_BGR2GRAY);
    }
    else{
        gray = chunk_with_halo;
    }
    Mat output_image = Mat(num_rows, chunk_with_halo.cols, CV_8UC1);    
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < num_rows; y++) {
        for (int x = 1; x < chunk_with_halo.cols - 1; x++) {
            int sum = 0;
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++)
                    sum += chunk_with_halo.at<uchar>(y+i+1, x+j) * kernel[i+1][j+1];

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(sum);
        }
    }
    return output_image;
}

vector<uint8_t> read_file_bytes(const string& path){
    ifstream file(path, ios::binary);
    if(!file){
        throw runtime_error("Cannot open file");
    }
    file.seekg(0,ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

Mat convert_bytes_to_img(vector<uint8_t> image_bytes){
    Mat img = imdecode(image_bytes, IMREAD_COLOR);
    if(img.empty()){
        throw std::runtime_error("Failed to decode image");
    }
    return img;
}

void write_convolution_img_to_file(Mat image){
    imwrite("output/"+to_string(0)+".jpg", image);
}

int main() {
    // vector<uint8_t> image_bytes;
    // image_bytes =  read_file_bytes("../../data/img/0.jpg");
    // vector<vector<int>> kernel = {{0, 1, 0}, {-1, 5, -1}, {0, -1, 0}};
    // Mat img = convert_bytes_to_img(image_bytes);
    // Mat convolution_img = image_chunk_convolution(kernel, img);
    // write_convolution_img_to_file(convolution_img);
    // cout<<"Output image created successfully"<<endl; 
    return 0;
}
