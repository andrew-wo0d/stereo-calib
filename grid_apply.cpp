#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

// Function to read grid file into memory
std::vector<std::tuple<float, float, float, float>> readGridFile(const std::string& gridFilePath) {
    std::vector<std::tuple<float, float, float, float>> gridData;
    std::ifstream inFile(gridFilePath, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Error: Unable to open grid file: " << gridFilePath << std::endl;
        return gridData;
    }

    while (!inFile.eof()) {
        float originalX, originalY, mappedX, mappedY;
        inFile.read(reinterpret_cast<char*>(&originalX), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&originalY), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&mappedX), sizeof(float));
        inFile.read(reinterpret_cast<char*>(&mappedY), sizeof(float));

        if (inFile.gcount() == 0) break; // End of file reached
        gridData.emplace_back(originalX, originalY, mappedX, mappedY);
    }

    inFile.close();
    return gridData;
}

// Function to perform bilinear interpolation for pixel value
cv::Vec3b bilinearInterpolation(const cv::Mat& image, float x, float y) {
    int x0 = static_cast<int>(std::floor(x));
    int x1 = x0 + 1;
    int y0 = static_cast<int>(std::floor(y));
    int y1 = y0 + 1;

    // Ensure bounds are valid
    x0 = std::clamp(x0, 0, image.cols - 1);
    x1 = std::clamp(x1, 0, image.cols - 1);
    y0 = std::clamp(y0, 0, image.rows - 1);
    y1 = std::clamp(y1, 0, image.rows - 1);

    cv::Vec3b p00 = image.at<cv::Vec3b>(y0, x0); // Top-left
    cv::Vec3b p01 = image.at<cv::Vec3b>(y0, x1); // Top-right
    cv::Vec3b p10 = image.at<cv::Vec3b>(y1, x0); // Bottom-left
    cv::Vec3b p11 = image.at<cv::Vec3b>(y1, x1); // Bottom-right

    float alpha = x - x0; // Horizontal interpolation factor
    float beta = y - y0;  // Vertical interpolation factor

    // Interpolate horizontally and vertically
    cv::Vec3b interpolated = (1 - beta) * ((1 - alpha) * p00 + alpha * p01) +
                             beta * ((1 - alpha) * p10 + alpha * p11);

    return interpolated;
}

// Function to interpolate mapped coordinates based on grid corners
std::pair<float, float> interpolateMappedCoordinates(float x, float y, 
                                                     const std::tuple<float, float, float, float>& topLeft,
                                                     const std::tuple<float, float, float, float>& topRight,
                                                     const std::tuple<float, float, float, float>& bottomLeft,
                                                     const std::tuple<float, float, float, float>& bottomRight) {
    auto [origTLX, origTLY, mappedTLX, mappedTLY] = topLeft;
    auto [origTRX, origTRY, mappedTRX, mappedTRY] = topRight;
    auto [origBLX, origBLY, mappedBLX, mappedBLY] = bottomLeft;
    auto [origBRX, origBRY, mappedBRX, mappedBRY] = bottomRight;

    // Calculate interpolation factors
    float alpha = (x - origTLX) / (origTRX - origTLX);
    float beta = (y - origTLY) / (origBLY - origTLY);

    // Bilinear interpolation for mapped coordinates
    float mappedX = (1 - beta) * ((1 - alpha) * mappedTLX + alpha * mappedTRX) +
                    beta * ((1 - alpha) * mappedBLX + alpha * mappedBRX);

    float mappedY = (1 - beta) * ((1 - alpha) * mappedTLY + alpha * mappedTRY) +
                    beta * ((1 - alpha) * mappedBLY + alpha * mappedBRY);

    return {mappedX, mappedY};
}

// Function to apply grid correction
cv::Mat applyGridCorrection(const cv::Mat& inputImage, const std::vector<std::tuple<float, float, float, float>>& gridData, const cv::Size& gridSize) {
    cv::Mat correctedImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

    // Iterate over every pixel in the input image
    for (int y = 0; y < inputImage.rows; ++y) {
        for (int x = 0; x < inputImage.cols; ++x) {
            // Calculate the grid cell coordinates
            int gridX = static_cast<int>(std::floor(static_cast<float>(x) / (inputImage.cols / gridSize.width)));
            int gridY = static_cast<int>(std::floor(static_cast<float>(y) / (inputImage.rows / gridSize.height)));

            gridX = std::clamp(gridX, 0, gridSize.width - 2); // Prevent accessing out-of-bound
            gridY = std::clamp(gridY, 0, gridSize.height - 2);

            // Get the four corners of the current grid cell
            const auto& topLeft = gridData[gridY * gridSize.width + gridX];
            const auto& topRight = gridData[gridY * gridSize.width + gridX + 1];
            const auto& bottomLeft = gridData[(gridY + 1) * gridSize.width + gridX];
            const auto& bottomRight = gridData[(gridY + 1) * gridSize.width + gridX + 1];

            // Interpolate the mapped coordinates for the current pixel
            auto [mappedX, mappedY] = interpolateMappedCoordinates(x, y, topLeft, topRight, bottomLeft, bottomRight);

            // Perform bilinear interpolation to fetch the pixel value
            cv::Vec3b pixelValue = bilinearInterpolation(inputImage, mappedX, mappedY);

            // Assign the interpolated value to the corrected image
            correctedImage.at<cv::Vec3b>(y, x) = pixelValue;
        }
    }

    return correctedImage;
}

// Visualize original and corrected images
void visualizeCorrection(const cv::Mat& originalImage, const cv::Mat& correctedImage, const std::string& windowName) {
    cv::Mat combined;
    cv::hconcat(originalImage, correctedImage, combined); // Combine images side by side
    cv::imshow(windowName, combined);
    cv::waitKey(0); // Show for a brief moment
}

// Process all images in the specified directory
void processImages(const std::string& inputDir, const std::string& gridFile, const std::string& outputDir, const cv::Size& gridSize) {
    // Load grid file
    std::vector<std::tuple<float, float, float, float>> grid = readGridFile(gridFile);
    // Ensure grid files were successfully loaded
    if (grid.empty()) {
        std::cerr << "Error: Grid files are empty or not loaded correctly." << std::endl;
        return;
    }

    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // Iterate through all images in the input directory
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) {
                std::cerr << "Error: Unable to read image: " << entry.path() << std::endl;
                continue;
            }
            cv::Mat correctedImage = applyGridCorrection(image, grid, gridSize);

            // Save corrected images
            std::string outputPathLeft = outputDir + "/" + entry.path().filename().string();
            cv::imwrite(outputPathLeft, correctedImage);
            // Visualize corrections
            visualizeCorrection(image, correctedImage, "Camera Correction");
        }
    }
}

int main() {
    // Load configuration from file
    std::string configFilePath = "configs.yaml";
    cv::FileStorage config(configFilePath, cv::FileStorage::READ);
    if (!config.isOpened()) {
        std::cerr << "Error: Unable to open configuration file: " << configFilePath << std::endl;
        return -1;
    }
    // Read directories from the configuration file
    std::string grid_left, grid_right, leftDir, rightDir, saveDir, gridSizeFile;
    config["leftGrid"] >> grid_left;
    config["rightGrid"] >> grid_right;
    config["leftCameraDir"] >> leftDir;
    config["rightCameraDir"] >> rightDir;
    config["saveDir"] >> saveDir;
    config["gridSizeFile"] >> gridSizeFile;
    config.release();

    std::cout << "Loaded configuration:" << std::endl;
    std::cout << "  leftCameraDir: " << leftDir << std::endl;
    std::cout << "  rightCameraDir: " << rightDir << std::endl;
    std::cout << "  saveDir: " << saveDir << std::endl;
    std::cout << "  gridSizeFile: " << gridSizeFile << std::endl;

    // Load configuration from file
    std::string gridFilePath = saveDir + "/" + gridSizeFile;
    cv::FileStorage fs(gridFilePath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open grid size file: " << gridFilePath << std::endl;
        return -1;
    }
    // Read grid size from the params file
    cv::Size gridSize;
    fs["gridWidth"] >> gridSize.width;
    fs["gridHeight"] >> gridSize.height;
    fs.release();

    std::cout << "  gridSize: " << gridSize.width << "x" << gridSize.height << std::endl;

    std::string gridFileLeft = saveDir + "/" + grid_left;    // Path to left camera grid file
    std::string gridFileRight = saveDir + "/" + grid_right;  // Path to right camera grid file
    std::string outputDirLeft = saveDir + "/" + "corrected_images_left";    // Output directory for corrected images
    std::string outputDirRight = saveDir + "/" + "corrected_images_right";    // Output directory for corrected images

    // Process all images using the grid files
    processImages(leftDir, gridFileLeft, outputDirLeft, gridSize);
    processImages(rightDir, gridFileRight, outputDirRight, gridSize);

    return 0;
}
