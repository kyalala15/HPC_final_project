#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

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

void normalizeImage(std::vector<std::vector<int>>& image, int width, int height, int maxVal) {
    if (maxVal == 0) return;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image[i][j] = (image[i][j] * 255) / maxVal;
        }
    }
}

void applySobel(const std::vector<std::vector<int>>& inputImage,
                std::vector<std::vector<int>>& outputImage,
                int width, int height) {
    int maxGradient = 0;
    
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int sumX = 0;
            int sumY = 0;
            
            for (int p = -1; p <= 1; ++p) {
                for (int q = -1; q <= 1; ++q) {
                    int pixel = inputImage[i + p][j + q];
                    sumX += pixel * GX[p + 1][q + 1];
                    sumY += pixel * GY[p + 1][q + 1];
                }
            }
            
            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            outputImage[i][j] = magnitude;
            maxGradient = std::max(maxGradient, magnitude);
        }
    }
    
    normalizeImage(outputImage, width, height, maxGradient);
}

bool readPGM(const std::string& filename, std::vector<std::vector<int>>& image, int& width, int& height) {
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
    file.ignore(1);

    image.resize(height, std::vector<int>(width));
    for (int i = 0; i < height; ++i) {
        file.read(reinterpret_cast<char*>(image[i].data()), width);
    }

    return true;
}

bool writePGM(const std::string& filename, const std::vector<std::vector<int>>& image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    file << "P5\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < height; ++i) {
        file.write(reinterpret_cast<const char*>(image[i].data()), width);
    }

    return true;
}

int main() {
    std::string inputFilename = "../images/pat1000.pgm";
    std::string outputFilename = "output.pgm";
    int width, height;

    std::vector<std::vector<int>> inputImage;
    if (!readPGM(inputFilename, inputImage, width, height)) {
        return 1;
    }

    std::vector<std::vector<int>> outputImage(height, std::vector<int>(width, 0));
    applySobel(inputImage, outputImage, width, height);

    if (!writePGM(outputFilename, outputImage, width, height)) {
        return 1;
    }

    std::cout << "Edge detection completed. Output saved to " << outputFilename << std::endl;
    return 0;
}