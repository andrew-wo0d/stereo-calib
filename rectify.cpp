#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace fs = std::filesystem;

// Compute Euler angles from rotation matrix
cv::Vec3d computeEulerAngles(const cv::Mat& R) {
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;

    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }

    return cv::Vec3d(x * (180.0 / M_PI), y * (180.0 / M_PI), z * (180.0 / M_PI));
}

// Recursive function for writing key-value pairs
template <typename T>
void writeParameter(cv::FileStorage& fs, const std::string& key, const T& value) {
    fs << key << value;
}

// Base function: variadic template recursion termination
void saveCalibrationParameters(cv::FileStorage& fs) {
    // Nothing more to process, just return
}

// Variadic template version of saveCalibrationParameters
template <typename T, typename... Args>
void saveCalibrationParameters(cv::FileStorage& fs, const std::string& key, const T& value, const Args&... args) {
    // Write the current key-value pair
    writeParameter(fs, key, value);

    // Recursively process the remaining parameters
    saveCalibrationParameters(fs, args...);
}

// Top-level function for managing FileStorage
template <typename... Args>
void saveCalibrationParameters(const std::string& filePath, const Args&... args) {
    // Open the FileStorage for writing
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Error: Unable to open file for writing: " << filePath << std::endl;
        return;
    }

    // Process all the key-value pairs
    saveCalibrationParameters(fs, args...);

    // Release the FileStorage
    fs.release();
    std::cout << "Calibration parameters saved to file: " << filePath << std::endl;
}

// Generate monocular calibration parameters
void generateMonocularParameters(const std::string& inputDir, const cv::Size boardSize, const float squareSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs) {
    std::vector<std::vector<cv::Point2f>> imagePoints;
    std::vector<std::vector<cv::Point3f>> objectPoints;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) continue;

            std::vector<cv::Point2f> corners;
            bool found = cv::findChessboardCorners(image, boardSize, corners);

            if (found) {
                cv::Mat gray;
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
                cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
                imagePoints.push_back(corners);

                std::vector<cv::Point3f> obj;
                for (int i = 0; i < boardSize.height; ++i) {
                    for (int j = 0; j < boardSize.width; ++j) {
                        obj.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
                    }
                }
                objectPoints.push_back(obj);
            }
        }
    }

    cv::calibrateCamera(objectPoints, imagePoints, boardSize, cameraMatrix, distCoeffs, cv::noArray(), cv::noArray());
    std::cout << "Monocular calibration completed. Camera matrix:\n" << cameraMatrix << "\nDistortion coefficients:\n" << distCoeffs << std::endl;
}

void monocularCorrection(const std::string& inputDir, const std::string& outputDir,
    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs) {
    // Create the output directory if it does not exist
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // Iterate through the input directory to process images
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
        // Read the input image
        cv::Mat image = cv::imread(entry.path().string());
            if (image.empty()) continue;

            // Perform undistortion
            cv::Mat undistorted;
            cv::undistort(image, undistorted, cameraMatrix, distCoeffs);

            // Save the corrected image to the output directory
            std::string outputPath = outputDir + "/" + entry.path().filename().string();
            cv::imwrite(outputPath, undistorted);

            std::cout << "Corrected image saved to: " << outputPath << std::endl;
        }
    }

    std::cout << "Monocular correction completed. All corrected images are saved to: " << outputDir << std::endl;
}


// Generate stereo calibration parameters
void generateStereoParameters(const std::string& inputDirLeft, const std::string& inputDirRight,
    const cv::Size boardSize, const float squareSize, const cv::Size imageSize,
    const cv::Mat& cameraMatrix1, const cv::Mat& distCoeffs1,
    const cv::Mat& cameraMatrix2, const cv::Mat& distCoeffs2,
    cv::Mat& R, cv::Mat& T, cv::Mat& E, cv::Mat& F) {
    std::vector<std::vector<cv::Point2f>> imagePointsLeft, imagePointsRight;
    std::vector<std::vector<cv::Point3f>> objectPoints;

    auto rightIter = fs::directory_iterator(inputDirRight); // Iterator for right images
    for (const auto& entryLeft : fs::directory_iterator(inputDirLeft)) {
        if (entryLeft.path().extension() == ".jpg" || entryLeft.path().extension() == ".png") {
            if (rightIter == fs::directory_iterator{}) break; // No corresponding right image

            cv::Mat leftImage = cv::imread(entryLeft.path().string());
            cv::Mat rightImage = cv::imread(rightIter->path().string());

            if (leftImage.empty() || rightImage.empty()) continue;

            std::vector<cv::Point2f> cornersLeft, cornersRight;
            bool foundLeft = cv::findChessboardCorners(leftImage, boardSize, cornersLeft);
            bool foundRight = cv::findChessboardCorners(rightImage, boardSize, cornersRight);

            if (foundLeft && foundRight) {
                cv::Mat grayLeft, grayRight;
                cv::cvtColor(leftImage, grayLeft, cv::COLOR_BGR2GRAY);
                cv::cvtColor(rightImage, grayRight, cv::COLOR_BGR2GRAY);

                cv::cornerSubPix(grayLeft, cornersLeft, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));
                cv::cornerSubPix(grayRight, cornersRight, cv::Size(11, 11), cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.001));

                imagePointsLeft.push_back(cornersLeft);
                imagePointsRight.push_back(cornersRight);

                std::vector<cv::Point3f> obj;
                for (int i = 0; i < boardSize.height; ++i) {
                    for (int j = 0; j < boardSize.width; ++j) {
                        obj.push_back(cv::Point3f(j * squareSize, i * squareSize, 0));
                    }
                }
                objectPoints.push_back(obj);
            }

            ++rightIter; // Advance the right image iterator
        }
    }

    // Call stereoCalibrate with the correct parameter list
    cv::stereoCalibrate(
        objectPoints, imagePointsLeft, imagePointsRight,
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
        imageSize, R, T, E, F, // Essential and Fundamental Matrices
        cv::CALIB_FIX_INTRINSIC,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1e-5)
    );

    std::cout << "Stereo calibration completed.\nRotation matrix R:\n" << R << "\nTranslation vector T:\n" << T << "\n";
}

void stereoCorrection(const std::string& inputDirLeft, const std::string& inputDirRight, const std::string& outputDir, const cv::Size imageSize,
    const cv::Mat& cameraMatrix1, const cv::Mat& distCoeffs1,
    const cv::Mat& cameraMatrix2, const cv::Mat& distCoeffs2,
    const cv::Mat& R, const cv::Mat& T,
    cv::Mat& R1, cv::Mat& R2,
    cv::Mat& P1, cv::Mat& P2, cv::Mat& Q) {
    // Create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // Rectification matrices and projection matrices
    cv::stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q);

    // Generate rectification maps
    cv::Mat mapx1, mapy1, mapx2, mapy2;
    cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_32FC1, mapx1, mapy1);
    cv::initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_32FC1, mapx2, mapy2);

    std::cout << "Stereo rectification completed:" << std::endl;

    // Process and remap images
    auto rightIter = fs::directory_iterator(inputDirRight); // Iterator for right images
    for (const auto& entryLeft : fs::directory_iterator(inputDirLeft)) {
        if (entryLeft.path().extension() == ".jpg" || entryLeft.path().extension() == ".png") {
            if (rightIter == fs::directory_iterator{}) break; // No corresponding right image

            // Read images
            cv::Mat leftImage = cv::imread(entryLeft.path().string());
            cv::Mat rightImage = cv::imread(rightIter->path().string());
            if (leftImage.empty() || rightImage.empty()) continue;

            // Remap images
            cv::Mat rectifiedLeft, rectifiedRight;
            cv::remap(leftImage, rectifiedLeft, mapx1, mapy1, cv::INTER_LINEAR);
            cv::remap(rightImage, rectifiedRight, mapx2, mapy2, cv::INTER_LINEAR);

            // Save rectified images
            std::string outputPathLeft = outputDir + "/rectified_left_" + entryLeft.path().filename().string();
            std::string outputPathRight = outputDir + "/rectified_right_" + rightIter->path().filename().string();

            cv::imwrite(outputPathLeft, rectifiedLeft);
            cv::imwrite(outputPathRight, rectifiedRight);

            std::cout << "Saved rectified images: " << outputPathLeft << ", " << outputPathRight << std::endl;

            ++rightIter; // Advance the right image iterator
        }
    }

    std::cout << "All stereo images have been rectified and saved to: " << outputDir << std::endl;
}


int main() {
    // Load configuration from file
    std::string configFilePath = "configs.yaml";
    cv::FileStorage config(configFilePath, cv::FileStorage::READ);
    if (!config.isOpened()) {
        std::cerr << "Error: Unable to open configuration file: " << configFilePath << std::endl;
        return -1;
    }
    // Read chessboard dimensions and square size from the configuration file
    cv::Size boardSize;
    cv::Size imageSize;
    float squareSize;
    config["boardWidth"] >> boardSize.width;
    config["boardHeight"] >> boardSize.height;
    config["squareSize"] >> squareSize;
    config["picWidth"] >> imageSize.width;
    config["picHeight"] >> imageSize.height;
    // Read directories from the configuration file
    std::string leftDir, rightDir, saveDir, calibParamsFile;
    config["leftCameraDir"] >> leftDir;
    config["rightCameraDir"] >> rightDir;
    config["saveDir"] >> saveDir;
    config["calibParamsFile"] >> calibParamsFile;
    config.release();
    if (boardSize.width <= 0 || boardSize.height <= 0 || squareSize <= 0) {
        std::cerr << "Invalid configuration values in " << configFilePath << std::endl;
        return -1;
    }
    if (!fs::exists(saveDir)) {
        fs::create_directory(saveDir);
    }
    std::cout << "Loaded configuration:" << std::endl;
    std::cout << "  boardSize: " << boardSize.width << "x" << boardSize.height << std::endl;
    std::cout << "  squareSize: " << squareSize << " mm" << std::endl;
    std::cout << "  imageSize: " << imageSize.width << "x" << imageSize.height << std::endl;
    std::cout << "  leftCameraDir: " << leftDir << std::endl;
    std::cout << "  rightCameraDir: " << rightDir << std::endl;
    std::cout << "  saveDir: " << saveDir << std::endl;
    std::cout << "  calibParamsFile: " << calibParamsFile << std::endl;

    cv::Mat cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2;
    cv::Mat R, T, E, F;
    cv::Mat R1, R2, P1, P2, Q;

    // Step 1: Generate and save monocular calibration parameters
    std::cout << "Generating monocular calibration parameters for the left camera..." << std::endl;
    generateMonocularParameters(leftDir, boardSize, squareSize, cameraMatrix1, distCoeffs1);

    std::cout << "Generating monocular calibration parameters for the right camera..." << std::endl;
    generateMonocularParameters(rightDir, boardSize, squareSize, cameraMatrix2, distCoeffs2);

    // Step 2: Generate stereo calibration parameters
    std::cout << "Generating stereo calibration parameters..." << std::endl;
    generateStereoParameters(leftDir, rightDir, boardSize, squareSize, imageSize, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F);

    // Step 3: Perform monocular correction
    std::cout << "Performing monocular correction for the left camera..." << std::endl;
    monocularCorrection(leftDir, saveDir + "/calibrated_left", cameraMatrix1, distCoeffs1);

    std::cout << "Performing monocular correction for the right camera..." << std::endl;
    monocularCorrection(rightDir, saveDir + "/calibrated_right", cameraMatrix2, distCoeffs2);

    // Step 4: Perform stereo correction
    std::cout << "Performing stereo correction..." << std::endl;
    stereoCorrection(leftDir, rightDir, saveDir + "/stereo_calibrated", imageSize, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, R1, R2, P1, P2, Q);

    // Output Euler angles of R1 and R2
    cv::Vec3d eulerR1 = computeEulerAngles(R1);
    cv::Vec3d eulerR2 = computeEulerAngles(R2);

    // Save the parameters
    saveCalibrationParameters(saveDir + "/" + calibParamsFile,
        "cameraMatrix1", cameraMatrix1,
        "distCoeffs1", distCoeffs1,
        "cameraMatrix2", cameraMatrix2,
        "distCoeffs2", distCoeffs2,
        "R", R,
        "T", T,
        "E", E,
        "F", F,
        "R1", R1,
        "R2", R2,
        "P1", P1,
        "P2", P2,
        "Q", Q,
        "Euler1", eulerR1,
        "Euler2", eulerR2);

    return 0;
}
