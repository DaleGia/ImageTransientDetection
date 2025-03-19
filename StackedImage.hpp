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

    void setStackAccumulatedExposure(uint64_t accumulatedExposure);
    void setNewStackCallback(
        std::function<void(cv::Mat &, double)> callback);

    void add(cv::Mat &image, double exposure, double gain);

    bool getStack(
        cv::Mat &buffer,
        double &brightnessFactor);

    bool getStack(
        cv::Mat &buffer);

    void reset(void);

private:
    cv::Mat stack;
    cv::Mat stackInProgress;
    uint64_t stackAccumulatedExposure;

    bool isStackSet;

    std::function<void(
        cv::Mat &image,
        double brightnessFactor)>
        newStackCallback;

    double accumulatedExposure;
    double brightnessFactor;
    uint64_t numberOfImages;
};

#endif