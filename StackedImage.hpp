#ifndef STACKEDIMAGE_H_
#define STACKEDIMAGE_H_

#include <opencv2/opencv.hpp>
#include <atomic>
#include <functional>
#include <vector>

class StackedImage
{
public:
    StackedImage();

    enum MODE
    {
        IMAGECOUNT = 0,
        ACCUMULATEDEXPOSURE
    };
    void setStackMode(StackedImage::MODE mode);
    void setStackAccumulatedExposure(uint64_t accumulatedExposure);
    void setStackNumberOfImages(uint64_t imageCount);

    void setNewStackCallback(
        std::function<void(cv::Mat &, double, double, uint64_t)> callback);

    void add(cv::Mat &image, double exposure, double gain);
    void add(cv::Mat &image, double exposure);
    bool getStack(
        cv::Mat &buffer,
        double &brightnessFactor,
        double &accumulatedExposure,
        uint64_t &numberOfImages);

    bool getStack(
        cv::Mat &buffer);

    void reset(void);

private:
    std::function<void(
        cv::Mat &image,
        double brightnessFactor,
        double accumulatedExposure,
        uint64_t numberOfImages)>
        newStackCallback;

    cv::Mat setStack;
    double setAccumulatedExposure = 0.0;
    double setBrightnessFactor = 0.0;
    uint64_t setStackNumberOfImagesCount = 0;

    cv::Mat stackInProgress;
    uint64_t configuredStackAccumulatedExposure = 0;
    uint64_t configuredStackNumberOfImages = 0;

    bool isStackSet;

    double accumulatedExposure = 0.0;
    double brightnessFactor = 0.0;
    uint64_t numberOfImages = 0;

    StackedImage::MODE mode = StackedImage::MODE::IMAGECOUNT;
};

#endif