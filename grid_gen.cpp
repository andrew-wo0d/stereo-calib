#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

// Function to dynamically calculate grid size based on image dimensions
cv::Size calculateGridSize(const cv::Size& imageSize, int gridBaseSize) {
    int rows = imageSize.height / gridBaseSize; // Calculate grid rows based on image height
    int cols = imageSize.width / gridBaseSize; // Calculate grid columns based on image width
    return cv::Size(cols, rows);
}

void saveGridSize(const std::string& filePath, const std::string& gridWidthKey, const int gridWidth, const std::string& gridHeightKey, const int gridHeight) {
    // Open the FileStorage for writing
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open file for writing: " << filePath << std::endl;
        return;
    }

    // Process grid width & height key-value pairs
    fs << gridWidthKey << gridWidth;
    fs << gridHeightKey << gridHeight;

    // Release the FileStorage
    fs.release();
    std::cout << "Grid size saved to file: " << filePath << std::endl;
}

// Function to generate the grid and save it to a binary file
void generateGrid(const cv::Size& imageSize, const cv::Size& gridSize, 
                  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, 
                  const cv::Mat& R, const cv::Mat& P, const std::string& outputFile, bool visualize = false) {
    int rows = gridSize.height + 1; // Include boundary
    int cols = gridSize.width + 1; // Include boundary

    // Generate grid points
    std::vector<cv::Point2f> gridPoints;
    float stepX = static_cast<float>(imageSize.width) / gridSize.width;
    float stepY = static_cast<float>(imageSize.height) / gridSize.height;

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            float px = std::min(x * stepX, static_cast<float>(imageSize.width));
            float py = std::min(y * stepY, static_cast<float>(imageSize.height));
            gridPoints.emplace_back(px, py);
        }
    }

    // Undistort points
    std::vector<cv::Point2f> undistortedPoints;
    cv::undistortPoints(gridPoints, undistortedPoints, cameraMatrix, distCoeffs, R, P);

    // Save grid to binary file
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << outputFile << std::endl;
        return;
    }

    for (size_t i = 0; i < gridPoints.size(); ++i) {
        float originalX = gridPoints[i].x;
        float originalY = gridPoints[i].y;
        float mappedX = undistortedPoints[i].x;
        float mappedY = undistortedPoints[i].y;

        outFile.write(reinterpret_cast<const char*>(&originalX), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&originalY), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&mappedX), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&mappedY), sizeof(float));
    }
    outFile.close();
    std::cout << "Grid parameters saved to: " << outputFile << std::endl;

    // Visualize grid if required
    if (visualize) {
        cv::Mat visualization(imageSize, CV_8UC3, cv::Scalar(255, 255, 255)); // White canvas
        for (const auto& pt : undistortedPoints) {
            cv::circle(visualization, pt, 3, cv::Scalar(0, 0, 255), -1); // Draw red points
        }
        cv::imshow("Grid Visualization", visualization);
        cv::waitKey(0); // Wait for key press to close visualization
    }
}

int main() {
    // Load image & grid size from config file
    std::string configFilePath = "configs.yaml";
    cv::FileStorage config = cv::FileStorage(configFilePath, cv::FileStorage::READ);
    if (!config.isOpened()) {
        std::cerr << "Error: Unable to open config file: " << configFilePath << std::endl;
        return -1;
    }
    int gridBaseSize;
    std::string grid_left, grid_right, saveDir, calibParamsFile, gridSizeFile;
    cv::Size imageSize;
    config["picWidth"] >> imageSize.width;
    config["picHeight"] >> imageSize.height;
    config["gridBaseSize"] >> gridBaseSize;
    config["leftGrid"] >> grid_left;
    config["rightGrid"] >> grid_right;
    config["saveDir"] >> saveDir;
    config["calibParamsFile"] >> calibParamsFile;
    config["gridSizeFile"] >> gridSizeFile;
    config.release();

    // Load calibration parameters from file
    std::string calibFilePath = saveDir + "/" + calibParamsFile;
    cv::FileStorage fs(calibFilePath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open calibration file: " << calibFilePath << std::endl;
        return -1;
    }
    cv::Mat cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2;
    cv::Mat R1, R2, P1, P2;
    fs["cameraMatrix1"] >> cameraMatrix1;
    fs["distCoeffs1"] >> distCoeffs1;
    fs["cameraMatrix2"] >> cameraMatrix2;
    fs["distCoeffs2"] >> distCoeffs2;
    fs["R1"] >> R1;
    fs["P1"] >> P1;
    fs["R2"] >> R2;
    fs["P2"] >> P2;
    fs.release();

    // Dynamically calculate grid size
    cv::Size gridSize = calculateGridSize(imageSize, gridBaseSize);
    std::cout << "Calculated grid size: " << gridSize.width << "x" << gridSize.height << std::endl;
    std::string gridFilePath = saveDir + "/" + gridSizeFile;
    saveGridSize(gridFilePath,
        "gridWidth", gridSize.width + 1,
        "gridHeight", gridSize.height + 1);

    // Generate grid for left camera
    std::string outputFileLeft = saveDir + "/" + grid_left;
    generateGrid(imageSize, gridSize, cameraMatrix1, distCoeffs1, R1, P1, outputFileLeft, true);

    // Generate grid for right camera
    std::string outputFileRight = saveDir + "/" + grid_right;
    generateGrid(imageSize, gridSize, cameraMatrix2, distCoeffs2, R2, P2, outputFileRight, true);

    return 0;
}
