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
#include "Benchmark.hpp"



class ImageTransientDetection
{
    public:
        ImageTransientDetection()
        {
            this->numberOfFramesInAverageFrame = 50;
            this->threshold = 1000;
            this->minimumSize = 1;
            this->maximumSize = 1000;
            this->newAverageFrameCallback = NULL;
            this->detectionEnabled = true;
        }

        ImageTransientDetection(
            uint16_t threshold,
            uint32_t minimumSize,
            uint32_t maximumSize,
            uint32_t averageFrameNumber = 1,
            std::function<void(cv::Mat)> averageFrameCallback = nullptr);
        
        void setNumberOfAverageFrames(uint32_t frames);
        void setThreshold(uint32_t threshold);
        void setMinimumSize(uint32_t size);
        void setMaximumSize(uint32_t size);
        void setNewAverageFrameCallback(std::function<void(cv::Mat)> callback);

        void disableDetection(void);
        void enableDetection(void);
        bool isDetectionEnabled(void);

        bool detect(
            cv::Mat &frame,
            cv::Mat &detectionFrame,
            cv::Rect &detectionBox,
            uint32_t &detectionSize);

        void getLastRawFrame(cv::Mat &buffer);
        void getLastDiffedFrame(cv::Mat &buffer);
        void getLastThresholdedFrame(cv::Mat &buffer);
        void getAverageFrame(cv::Mat &buffer);
    private:
        cv::Mat lastRawFrame;

        cv::Mat lastDiffedFrame;

        cv::Mat lastThresholdedFrame;

        cv::Mat averageFrame;

        bool averageFrameSet = false;
        cv::Mat nextAverageFrame;
        uint32_t nextAverageFrameCount;
        

        volatile uint32_t threshold;
        volatile uint32_t minimumSize;
        volatile uint32_t maximumSize;
        volatile uint32_t numberOfFramesInAverageFrame;
        std::function<void(cv::Mat)> newAverageFrameCallback;

        volatile bool detectionEnabled;

        // Benchmark detectTime;
        // Benchmark imageProcessTime;
        // Benchmark transientDetectTime;
};

ImageTransientDetection::ImageTransientDetection(
    uint16_t threshold,
    uint32_t minimumSize,
    uint32_t maximumSize,
    uint32_t averageFrameNumber,
    std::function<void(cv::Mat)> averageFrameCallback)
{
    this->numberOfFramesInAverageFrame = averageFrameNumber;
    this->threshold = threshold;
    this->minimumSize = minimumSize;
    this->maximumSize = maximumSize;
    this->newAverageFrameCallback = averageFrameCallback;
}

void ImageTransientDetection::setNumberOfAverageFrames(uint32_t frames)
{
    this->numberOfFramesInAverageFrame = frames;
}

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

void ImageTransientDetection::setNewAverageFrameCallback(
    std::function<void(cv::Mat)> callback)
{
    this->newAverageFrameCallback = callback;
}

void ImageTransientDetection::disableDetection(void)
{
    detectionEnabled = false;
}

void ImageTransientDetection::enableDetection(void)
{
    detectionEnabled = true;
}

bool ImageTransientDetection::isDetectionEnabled(void)
{
    return detectionEnabled;
}

bool ImageTransientDetection::detect(
    cv::Mat &frame,
    cv::Mat &detectionFrame,
    cv::Rect &detectionBox,
    uint32_t &detectionSize)
{
    // detectTime.start();
    // imageProcessTime.start();

    bool validDetectionSet = false;

    if(this->nextAverageFrame.empty())
    {
        this->nextAverageFrame = 
                cv::Mat::zeros(frame.size(), frame.type());
    }

    this->lastRawFrame = frame.clone();

    cv::add(
            this->nextAverageFrame, 
            frame, 
            this->nextAverageFrame);
    this->nextAverageFrame /= 2;
    this->nextAverageFrameCount++;
    if(this->nextAverageFrameCount >= numberOfFramesInAverageFrame)
    {
        this->nextAverageFrameCount = 0;

        this->averageFrame = this->nextAverageFrame.clone();

        this->nextAverageFrame = 
            cv::Mat::zeros(frame.size(), frame.type());        

        if(nullptr != this->newAverageFrameCallback)
        {
            this->newAverageFrameCallback(this->averageFrame.clone());
        }    
    }

    // imageProcessTime.stop();
    // transientDetectTime.start();
    if(!this->averageFrame.empty())
    {
        /* This is the actual detection */

        double minValue; 
        double maxValue;
        cv::subtract(frame, this->averageFrame, this->lastDiffedFrame);

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

        this->lastThresholdedFrame.convertTo(this->lastThresholdedFrame, CV_8U);
        
        int num_labels;
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
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
            else if(true == validDetectionSet)
            {
                // std::cout << "Multiple detections in single image found" << std::endl;
                // std::cout << "Try adjusting number of frames in average " <<
                // "frame, " << "threshold, minimum size, or maximum size" << std::endl;
                // validDetectionSet = false;
                break;
            }
            else
            {
                validDetectionSet = true;

                detectionBox = cv::Rect(
                    stats.at<int>(i, cv::CC_STAT_LEFT), 
                    stats.at<int>(i, cv::CC_STAT_TOP),
                    stats.at<int>(i, cv::CC_STAT_WIDTH),
                    stats.at<int>(i, cv::CC_STAT_HEIGHT));
                detectionSize = stats.at<int>(i, cv::CC_STAT_AREA);
                detectionFrame = frame(detectionBox).clone();
            }
        }
    }
    // transientDetectTime.stop();
    // detectTime.stop();
    // imageProcessTime.print("image processing");
    // transientDetectTime.print("transient detection");
    // detectTime.print("function");
    // std::cout << std::endl;
    return validDetectionSet;
}

void ImageTransientDetection::getLastRawFrame(cv::Mat &buffer)
{
    buffer = this->lastRawFrame.clone();
}

void ImageTransientDetection::getLastDiffedFrame(cv::Mat &buffer)
{
    buffer = this->lastDiffedFrame.clone();
}

void ImageTransientDetection::getLastThresholdedFrame(cv::Mat &buffer)
{
    buffer = this->lastThresholdedFrame.clone();
}

void ImageTransientDetection::getAverageFrame(cv::Mat &buffer)
{
    buffer = this->averageFrame.clone();
}