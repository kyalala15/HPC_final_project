#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>

#define INPUT_FILE_PATH "../images/pat1000.pgm"
#define NUM_THREADS 1

const int GX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int GY[3][3] = {
    {-1, -2, -1},
    {0,  0,  0},
    {1,  2,  1}
};

void applySobel(const std::vector<int>& inputImage, std::vector<int>& outputImage, int width, int height) {

    const int TILE_SIZE = 32;  // Tune for L1/L2 cache behavior

    // #pragma omp parallel for collapse(2) schedule(dynamic)
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int ii = 1; ii < height - 1; ii += TILE_SIZE) {
        for (int jj = 1; jj < width - 1; jj += TILE_SIZE) {
            for (int i = ii; i < std::min(ii + TILE_SIZE, height - 1); ++i) {
                for (int j = jj; j < std::min(jj + TILE_SIZE, width - 1); ++j) {
                    int sumX = 0, sumY = 0;

                    for (int p = -1; p <= 1; ++p) {
                        for (int q = -1; q <= 1; ++q) {
                            int pixel = inputImage[(i + p) * width + (j + q)];
                            sumX += pixel * GX[p + 1][q + 1];
                            sumY += pixel * GY[p + 1][q + 1];
                        }
                    }

                    int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
                    outputImage[i * width + j] = magnitude;
                }
            }
        }
    }
}

void normalizeImage(std::vector<int>& image, int width, int height) {
    int maxVal = 0;
    for (int i = width + 1; i < (height - 1) * width - 1; ++i) {
        maxVal = std::max(maxVal, image[i]);
    }

    if (maxVal == 0) return;

    for (int i = width + 1; i < (height - 1) * width - 1; ++i) {
        image[i] = (image[i] * 255) / maxVal;
    }
}

bool readPGM(const std::string& filename, std::vector<int>& image, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string magicNumber;
    file >> magicNumber;
    if (magicNumber != "P5") {
        std::cerr << "Error: Unsupported file format." << std::endl;
        return false;
    }

    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore(1); // Skip newline

    std::vector<unsigned char> buffer(width * height);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    image.resize(width * height);
    for (int i = 0; i < width * height; ++i) {
        image[i] = static_cast<int>(buffer[i]);
    }

    return true;
}

bool writePGM(const std::string& filename, const std::vector<int>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    file << "P5\n" << width << " " << height << "\n255\n";

    std::vector<unsigned char> buffer(width * height);
    for (int i = 0; i < width * height; ++i) {
        buffer[i] = static_cast<unsigned char>(std::min(255, std::max(0, image[i])));
    }

    file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
    return true;
}

int main() {
    std::string inputFilename = INPUT_FILE_PATH;
    std::string outputFilename = "output.pgm";
    int width, height;

    std::cout << "# threads: " << NUM_THREADS << std::endl;

    std::vector<int> inputImage;
    if (!readPGM(inputFilename, inputImage, width, height)) {
        return 1;
    }

    std::vector<int> outputImage(width * height, 0);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    applySobel(inputImage, outputImage, width, height);

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Sobel filter execution time: " << duration.count() << " ms" << std::endl;

    normalizeImage(outputImage, width, height);

    if (!writePGM(outputFilename, outputImage, width, height)) {
        return 1;
    }

    std::cout << "Edge detection completed. Output saved to " << outputFilename << std::endl;
    return 0;
}
