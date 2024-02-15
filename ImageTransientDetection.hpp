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
        ImageTransientDetection(){};

        void setThreshold(uint32_t threshold);
        void setMinimumSize(uint32_t size);
        void setMaximumSize(uint32_t size);
        void setDialationSize(uint32_t size);

        bool detect(
            cv::Mat &frameA,
            cv::Mat &frameB,
            cv::Rect &detectionBox,
            cv::Point &detectionCentroid,    
            uint32_t &detectionSize);

        void getLastDiffedFrame(cv::Mat &buffer);
        void getLastThresholdedFrame(cv::Mat &buffer);
    private:
        cv::Mat lastDiffedFrame;
        cv::Mat lastThresholdedFrame;
        cv::Mat dialationKernal;

        volatile uint32_t threshold;
        volatile uint32_t minimumSize;
        volatile uint32_t maximumSize;
        volatile uint32_t dialationSize;
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

void ImageTransientDetection::setDialationSize(uint32_t dialationSize)
{
    this->dialationSize = dialationSize;
    this->dialationKernal = 
        cv::getStructuringElement(
            cv::MORPH_ELLIPSE, 
            cv::Size(2 * this->dialationSize + 1, 2 * this->dialationSize + 1));
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

    bool validDetectionSet;
    double minValue; 
    double maxValue;
    int num_labels;
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;

    validDetectionSet = false;

    if(true == this->lastDiffedFrame.empty())
    {
        this->lastDiffedFrame = cv::Mat::zeros(frameA.size(), frameA.type());
    }

    /* diff the two frames */
    cv::absdiff(frameA, frameB, this->lastDiffedFrame);
    cv::minMaxLoc(
        this->lastDiffedFrame, 
        &minValue, 
        &maxValue);

    cv::threshold(
        this->lastDiffedFrame, 
        this->lastThresholdedFrame, 
        this->threshold, 
        maxValue,
        cv::THRESH_BINARY);

    // cv::dilate(
    //     this->lastThresholdedFrame, 
    //     this->lastThresholdedFrame, 
    //     this->dialationKernal);

    this->lastThresholdedFrame.convertTo(this->lastThresholdedFrame, CV_8U);
    
    num_labels = 
        cv::connectedComponentsWithStats(
            this->lastThresholdedFrame, 
            labels, 
            stats, 
            centroids, 
            4, 
            CV_32S);

    for(int i = 1; i < num_labels; ++i) 
    {
        double x, y;
        cv::Point w, h;            

        if(stats.at<int>(i, cv::CC_STAT_AREA) < this->minimumSize) 
        {
            // Not a valid detection. Too small
        }
        else if(stats.at<int>(i, cv::CC_STAT_AREA) > this->maximumSize) 
        {
            // Not a valid detection. Too big
        }
        else
        {
            /* For the detection bounding box */
            detectionBox = cv::Rect(
                stats.at<int>(i, cv::CC_STAT_LEFT), 
                stats.at<int>(i, cv::CC_STAT_TOP),
                stats.at<int>(i, cv::CC_STAT_WIDTH),
                stats.at<int>(i, cv::CC_STAT_HEIGHT));

            /* For the detection centroid */
            double x = centroids.at<double>(i, 0);
            double y = centroids.at<double>(i, 1);
            int roundedX = static_cast<int>(std::round(x));
            int roundedY = static_cast<int>(std::round(y));
            detectionCentroid = cv::Point(roundedX, roundedY);
            
            /* For the detection size */
            detectionSize = stats.at<int>(i, cv::CC_STAT_AREA);

            validDetectionSet = true;
        }
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