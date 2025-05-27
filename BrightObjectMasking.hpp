#ifndef BRIGHTOBJECTMASKING_H_
#define BRIGHTOBJECTMASKING_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>

class BrightObjectMasking
{
public:
    BrightObjectMasking();
    BrightObjectMasking(
        uint32_t sigma,
        uint32_t erosionKernalSize,
        uint32_t dialationKernalSize);

    void setSigma(uint32_t sigma);
    void setErosionKernalSize(uint32_t erosionKernalSize);
    void setDilationKernalSize(uint32_t dilationKernalSize);
    void enableBrightMaskImageSaving(std::filesystem::path filepath);
    void disableBrightMaskImageSaving();
    void mask(cv::Mat &image);
    void getMask(cv::Mat &image, cv::Mat &mask);

private:
    uint32_t sigma;
    uint32_t erosionKernalSize;
    uint32_t dilationKernalSize;

    cv::Mat erosionKernel;
    cv::Mat dialationKernel;

    bool saveImages;
    std::filesystem::path filepath;

    void validateFilepath();
};

#endif