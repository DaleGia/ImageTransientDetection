#ifndef IMAGETRANSIENTDETECTION_H_
#define IMAGETRANSIENTDETECTION_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <atomic>

class ImageTransientDetection
{
public:
    ImageTransientDetection();
    void setDebugMode(bool debug);
    void setSigma(uint8_t sigma);
    void setMinimumSize(uint32_t size);
    void setMaximumSize(uint32_t size);
    uint32_t detect(cv::Mat &frameA, cv::Mat &frameB, cv::Rect &detectionBox, double &maxPixelValue);
    struct Stats
    {
        double absdiffMin;
        double absdiffMax;
        double absdiffMean;
        double absdiffStdDev;
        double sigmaThreshold;
        double numberOfContours;
        double numberOfValidContours;
        double largestContour;
    };
    ImageTransientDetection::Stats getLastImageStats() const;

private:
    double absdiffMin;
    double absdiffMax;
    double absdiffMean;
    double absdiffStdDev;
    double sigmaThreshold;
    double numberOfContours;
    double numberOfValidContours;
    double largestContour;
    uint32_t sigma;
    uint32_t minimumSize;
    uint32_t maximumSize;

    bool debug = false;
};

#endif