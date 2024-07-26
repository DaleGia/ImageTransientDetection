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
#include "ImagePreviewWindow.hpp"

class ImageTransientDetection
{
public:
    ImageTransientDetection() {};

    void setThreshold(uint32_t threshold);
    void setMinimumSize(uint32_t size);
    void setMaximumSize(uint32_t size);

    bool detect(
        cv::Mat &frameA,
        cv::Mat &frameB,
        cv::Rect &detectionBox,
        cv::Point &detectionCentroid,
        uint32_t &detectionSize);

    void getLastDiffedFrame(cv::Mat &buffer);
    void getLastThresholdedFrame(cv::Mat &buffer);
    uint32_t getLastLargestContour();

private:
    cv::Mat lastDiffedFrame;
    cv::Mat lastThresholdedFrame;
    uint32_t lastLargestContour = 0;

private:
    cv::Mat lastDiffedFrame;
    cv::Mat lastThresholdedFrame;

    volatile uint32_t threshold = 2;
    volatile uint32_t minimumSize = 1;
    volatile uint32_t maximumSize = 4294967295;
};

void ImageTransientDetection::setThreshold(uint32_t threshold)
{
    this->threshold = threshold;
}

void ImageTransientDetection::setMinimumSize(uint32_t size)
{
    this->minimumSize = size;
}

void ImageTransientDetection::setMaximumSize(uint32_t size)
{
    this->maximumSize = size;
}

/**
 * @brief
 * Diffs two images, applies a threshold then detects if any blobs of pixels exist
 * between the minimum and maxiumum defined sizes. YOU MUST NORMALISE IMAGES
 * BEFORE CALLING THIS FUCTION.
 *
 * @param frameA
 * @param frameB
 * @param detectionBox
 * @param detectionCentroid
 * @param detectionSize
 * @return true
 * @return false
 */
bool ImageTransientDetection::detect(
    cv::Mat &frameA,
    cv::Mat &frameB,
    cv::Rect &detectionBox,
    cv::Point &detectionCentroid,
    uint32_t &detectionSize)
{
    // detectTime.start();
    // imageProcessTime.start();
    cv::Mat maskedFrameA;
    cv::Mat maskedFrameB;
    bool validDetectionSet;
    double minValue;
    double maxValue;
    cv::Scalar average;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Moments> moments;

    validDetectionSet = false;

    this->lastDiffedFrame = cv::Mat::zeros(frameA.size(), frameA.type());

    /* diff the two frames */
    try
    {
        cv::absdiff(frameA, frameB, this->lastDiffedFrame);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Image Transient Detection absdiff error: " << e.what() << '\n';
    }

    try
    {
        cv::threshold(
            this->lastDiffedFrame,
            this->lastThresholdedFrame,
            this->threshold,
            255,
            cv::THRESH_BINARY);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Image Transient Detection thresholding error: " << e.what() << '\n';
    }

    try
    {
        cv::Mat tempthresh;
        this->lastThresholdedFrame.convertTo(tempthresh, CV_8U, 1 / 255.0, 0);
        cv::findContours(
            tempthresh,
            contours,
            hierarchy,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);
    }
    catch (const std::exception &e)
    {
        std::cerr << "findContours error: " << e.what() << '\n';
    }

    this->lastLargestContour = 0;
    int largestContourIndex = 0;
    for (int i = 1; i < contours.size(); i++)
    {
        if (this->lastLargestContour < cv::contourArea(contours[i]))
        {
            largestContourIndex = i;
            this->lastLargestContour = cv::contourArea(contours[i]);
        }
    }

    if (this->lastLargestContour < this->minimumSize)
    {
        // Not a valid detection. Too small
    }
    else if (this->lastLargestContour > this->maximumSize)
    {
        // Not a valid detection. Too big
    }
    else
    {
        validDetectionSet = true;
    }

    if (true == validDetectionSet)
    {
        try
        {
            cv::threshold(
                this->lastDiffedFrame,
                this->lastThresholdedFrame,
                this->threshold,
                255,
                cv::THRESH_BINARY);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Image Transient Detection thresholding error: " << e.what() << '\n';
        }
        /* For the detection bounding box */
        detectionBox = cv::boundingRect(contours[largestContourIndex]);
        cv::Moments M = cv::moments(contours[largestContourIndex]);
        detectionCentroid = cv::Point(M.m10 / M.m00, M.m01 / M.m00);

        /* For the detection size */
        float size;
        cv::Point2f center;

        cv::minEnclosingCircle(contours[largestContourIndex], center, size);

        detectionSize = (uint32_t)(size * 2.0);
    }

    return validDetectionSet;
}

void ImageTransientDetection::getLastDiffedFrame(cv::Mat &buffer)
{
    buffer = this->lastDiffedFrame.clone();
}

void ImageTransientDetection::getLastThresholdedFrame(cv::Mat &buffer)
{
    buffer = this->lastThresholdedFrame.clone();
}

uint32_t ImageTransientDetection::getLastLargestContour()
{
    return this->lastLargestContour;
}