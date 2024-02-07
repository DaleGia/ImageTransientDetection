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
            this->numberOfFramesInStack = 1;
            this->threshold = 1000;
            this->minimumSize = 1;
            this->maximumSize = 1000;
            this->detectionEnabled = true;
        }

        ImageTransientDetection(
            uint16_t threshold,
            uint32_t minimumSize,
            uint32_t maximumSize,
            uint32_t stackedFrameNumber = 1);
        
        void setNumberOfFramesInStack(uint32_t frames);
        void setNumberOfStrackedDetectionFrames(uint32_t frames);
        void setThreshold(uint32_t threshold);
        void setMinimumSize(uint32_t size);
        void setMaximumSize(uint32_t size);

        void disableDetection(void);
        void enableDetection(void);
        bool isDetectionEnabled(void);

        bool detect(
            cv::Mat &frame,
            cv::Mat &detectionFrame,
            cv::Rect &detectionBox,
            uint32_t &detectionSize);

        void getLastRawFrame(cv::Mat &buffer);
        void getLastStackedFrame(cv::Mat &buffer);
        void getLastDiffedFrame(cv::Mat &buffer);
        void getLastThresholdedFrame(cv::Mat &buffer);
    private:
        cv::Mat lastRawFrame;

        cv::Mat lastDiffedStackedFrame;

        cv::Mat lastThresholdedStackedFrame;

        cv::Mat stackedFrameA;
        cv::Mat stackedFrameB;

        bool stackedFramesSet = false;
        

        volatile uint32_t threshold;
        volatile uint32_t minimumSize;
        volatile uint32_t maximumSize;
        volatile uint32_t numberOfFramesInStack;
        volatile bool detectionEnabled;

        uint32_t stackInProgressCount = 0;
        uint32_t stackCompleteCount = 0;

        // Benchmark detectTime;
        // Benchmark imageProcessTime;
        // Benchmark transientDetectTime;
};

ImageTransientDetection::ImageTransientDetection(
    uint16_t threshold,
    uint32_t minimumSize,
    uint32_t maximumSize,
    uint32_t stackedFrameNumber)
{
    this->numberOfFramesInStack = stackedFrameNumber;
    this->threshold = threshold;
    this->minimumSize = minimumSize;
    this->maximumSize = maximumSize;    
}

void ImageTransientDetection::setNumberOfFramesInStack(uint32_t frames)
{
    this->numberOfFramesInStack = frames;
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

void ImageTransientDetection::disableDetection(void)
{
    this->detectionEnabled = false;
}

void ImageTransientDetection::enableDetection(void)
{
    this->detectionEnabled = true;
}

bool ImageTransientDetection::isDetectionEnabled(void)
{
    return this->detectionEnabled;
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

    if(this->stackedFrameA.empty())
    {
        this->stackedFrameA = 
                cv::Mat::zeros(frame.size(), CV_32F);
    }

    if(this->stackedFrameB.empty())
    {
        this->stackedFrameB = 
                cv::Mat::zeros(frame.size(), CV_32F);
    }
    
    frame.convertTo(this->lastRawFrame, CV_32F);
    cv::accumulate(frame, this->stackedFrameB);
    this->stackInProgressCount++;
    if(this->stackInProgressCount >= this->numberOfFramesInStack)
    {
        /* This means we are ready to do the diffing here*/
    
        /* Now that the diffing has been done we can get ready to build the
         next stack*/
        this->stackInProgressCount = 0;
        this->stackedFrameA = this->stackedFrameB.clone();
        this->stackedFrameB = 
            cv::Mat::zeros(frame.size(), CV_32F);
        stackCompleteCount++;
        if(1 < stackCompleteCount)   
        {
            this->stackedFramesSet = true;      
        } 
    }

    // imageProcessTime.stop();
    // transientDetectTime.start();
    if(true == this->stackedFramesSet)
    {
        /* This is the actual detection */

        double minValue; 
        double maxValue;
        
        /* subtract the accumulated frame from the raw frame */
        cv::absdiff(this->stackedFrameA, this->stackedFrameB, this->lastDiffedStackedFrame);
        
        cv::minMaxLoc(
            this->lastDiffedStackedFrame, 
            &minValue, 
            &maxValue);

        cv::threshold(
            this->lastDiffedStackedFrame, 
            this->lastThresholdedStackedFrame, 
            this->threshold, 
            maxValue,
            cv::THRESH_BINARY);

        this->lastThresholdedStackedFrame.convertTo(this->lastThresholdedStackedFrame, CV_8U);
        
        int num_labels;
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        num_labels = 
            cv::connectedComponentsWithStats(
                this->lastThresholdedStackedFrame, 
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
    buffer = this->lastDiffedStackedFrame.clone();
}

void ImageTransientDetection::getLastThresholdedFrame(cv::Mat &buffer)
{
    buffer = this->lastThresholdedStackedFrame.clone();
}

void ImageTransientDetection::getLastStackedFrame(cv::Mat &buffer)
{
    buffer = this->stackedFrameA.clone();
}