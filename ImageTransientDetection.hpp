#ifndef IMAGETRANSIENTDETECTION_H_
#define IMAGETRANSIENTDETECTION_H_

/**
 * @file ImageTransientDetection.hpp
 * Copyright (c) 2023 Dale Giancono All rights reserved.
 *
 * @brief
 * TODO add me
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <atomic>

class ImageTransientDetection
{
public:
    ImageTransientDetection() {};

    void setSigma(uint8_t sigma);
    void setMinimumSize(uint32_t size);
    void setMaximumSize(uint32_t size);

    uint32_t detect(
        cv::Mat& frameA,
        cv::Mat& frameB,
        cv::Rect& detectionBox,
        double& maxPixelValue);

    struct LastImageStats
    {
        double absdiffMin;
        double absdiffMax;
        double absdiffMean;
        double absdiffStdDev;;
        double sigmaThreshold;
        double numberOfContours;
        double numberOfValidContours;
    };

    LastImageStats getLastImageStats() const;

private:
    std::atomic<double> absdiffMin;
    std::atomic<double> absdiffMax;
    std::atomic<double> absdiffMean;
    std::atomic<double> absdiffStdDev;
    std::atomic<double> sigmaThreshold;
    std::atomic<double> numberOfContours;
    std::atomic<double> numberOfValidContours;
    std::atomic<uint32_t> sigma = 5;
    std::atomic<uint32_t> minimumSize = 1;
    std::atomic<uint32_t> maximumSize = 4294967295;
};

/**
 * @brief
 * Set the sigma value for transient detection.
 *
 * @param sigma
 * The sigma value used to calculate the threshold for detecting transients.
 * A higher sigma results in a higher threshold, making detection more selective.
 */

void ImageTransientDetection::setSigma(uint8_t sigma)
{
    this->sigma = sigma;
}

/**
 * @brief
 * Set the minimum size of a transient object in terms of the number of pixels.
 *
 * @param size
 * The minimum size of a transient object. Any contours with fewer pixels than
 * this will be rejected.
 */
void ImageTransientDetection::setMinimumSize(uint32_t size)
{
    this->minimumSize = size;
}

/**
 * @brief
 * Set the maximum size of a transient object in terms of the number of pixels.
 *
 * @param size
 * The maximum size of a transient object in terms of the number of pixels.
 */
void ImageTransientDetection::setMaximumSize(uint32_t size)
{
    this->maximumSize = size;
}


/**
 * @brief
 * Detect transient objects in a pair of images.
 *
 * This function subtracts the two input images, calculates the absolute difference,
 * and then applies a threshold to the result. It then finds contours in the
 * thresholded image, and tests each contour to see if it is a valid transient
 * object based on its area.
 *
 * @param frameA
 * The first image in the pair.
 * @param frameB
 * The second image in the pair.
 * @param detectionBox
 * The bounding box of the transient object(s) (if any) in the pair of images.
 * @param maxPixelValue
 * The maximum pixel value of any transient object in the pair of images.
 *
 * @return
 * The number of valid transient objects detected in the pair of images.
 */
uint32_t ImageTransientDetection::detect(
    cv::Mat& frameA,
    cv::Mat& frameB,
    cv::Rect& detectionBox,
    double& maxPixelValue)
{
    cv::Mat absDiffFrame;
    cv::Mat thresholdFrame;

    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> validContours;
    std::vector<cv::Vec4i> hierarchy;

    double min;
    double max;
    cv::Scalar mean;
    cv::Scalar std;
    double threshold;

    absDiffFrame = cv::Mat::zeros(frameA.size(), frameA.type());

    /* diff the two frames */
    try
    {
        cv::absdiff(frameA, frameB, absDiffFrame);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Image Transient Detection absdiff error: " << e.what() << '\n';
    }

    try
    {
        cv::minMaxLoc(absDiffFrame, &min, &max);
        cv::meanStdDev(absDiffFrame, mean, std);

        threshold = mean[0] + (this->sigma * std[0]);
        cv::threshold(
            absDiffFrame,
            thresholdFrame,
            threshold,
            255,
            cv::THRESH_BINARY);
        thresholdFrame.convertTo(thresholdFrame, CV_8U);

    }
    catch (const std::exception& e)
    {
        std::cerr << "Image Transient Detection thresholding error: " << e.what() << '\n';
    }

    try
    {
        cv::findContours(
            thresholdFrame,
            contours,
            hierarchy,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);
    }
    catch (const std::exception& e)
    {
        std::cerr << "findContours error: " << e.what() << '\n';
    }

    this->numberOfContours = contours.size();
    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();
    maxPixelValue = std::numeric_limits<double>::min();

    for (auto& contour : contours)
    {
        double contourArea = cv::contourArea(contour);
        if (contourArea >= this->minimumSize && contourArea <= this->maximumSize)
        {
            validContours.push_back(contour);

            for (const auto& point : contour)
            {
                minX = std::min(minX, point.x);
                minY = std::min(minY, point.y);
                maxX = std::max(maxX, point.x);
                maxY = std::max(maxY, point.y);

                /* This compares the maximum pixel value of this contour to
                   the other contours */
                cv::Rect boundingRect = cv::boundingRect(contour);
                for (int y = boundingRect.y; y < boundingRect.y + boundingRect.height; y++)
                {
                    for (int x = boundingRect.x; x < boundingRect.x + boundingRect.width; x++)
                    {
                        if (frameA.at<double>(y, x) > maxPixelValue)
                        {
                            maxPixelValue = frameA.at<double>(y, x);
                        }
                    }
                }
            }
        }
        if (!validContours.empty())
        {
            detectionBox = cv::Rect(minX, minY, maxX - minX, maxY - minY);
        }

        this->absdiffMin.store(min, std::memory_order_relaxed);
        this->absdiffMax.store(max, std::memory_order_relaxed);
        this->absdiffMean.store(mean[0], std::memory_order_relaxed);
        this->absdiffStdDev.store(std[0], std::memory_order_relaxed);
        this->sigmaThreshold.store(threshold, std::memory_order_relaxed);
        this->numberOfContours.store(contours.size(), std::memory_order_relaxed);
        this->numberOfValidContours.store(validContours.size(), std::memory_order_relaxed);

        return validContours.size();
    }
}

/**
 * @brief Gets the statistics of the last image that was processed.
 *
 * @return The statistics of the last image that was processed.
 */
ImageTransientDetection::LastImageStats ImageTransientDetection::getLastImageStats() const
{
    ImageTransientDetection::LastImageStats stats;
    stats.absdiffMin = this->absdiffMin;
    stats.absdiffMax = this->absdiffMax;
    stats.absdiffMean = this->absdiffMean;
    stats.absdiffStdDev = this->absdiffStdDev;
    stats.sigmaThreshold = this->sigmaThreshold;
    stats.numberOfContours = this->numberOfContours;
    stats.numberOfValidContours = this->numberOfValidContours;
    return stats;
}


#endif
