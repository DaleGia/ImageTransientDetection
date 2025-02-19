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

    void setStackAccumulatedExposure(uint64_t accumulatedExposureUs);
    void setNewStackCallback(
        std::function<void(cv::Mat &, uint64_t)> callback);

    void add(cv::Mat &image, uint64_t exposureUs);

    bool getStack(
        cv::Mat &buffer,
        uint64_t &accumulatedExposureUs);

    bool getStack(
        cv::Mat &buffer);

    void reset(void);

private:
    cv::Mat stack;
    cv::Mat stackInProgress;
    uint64_t stackAccumulatedExposureUs;

    bool isStackSet;

    std::function<void(
        cv::Mat &image,
        uint64_t accumulatedExposureUs)>
        newStackCallback;

    uint64_t accumulatedExposureUs;
    uint64_t numberOfImages;
};

#endif