/**
 * @file ImageTransientDetectionCUDA.hpp
 * Copyright (c) 2023 Dale Giancono All rights reserved.
 * 
 * @brief
 * TODO add me
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include "Benchmark.hpp"



class ImageTransientDetectionCUDA
{
    public:
        ImageTransientDetectionCUDA()
        {
        }

        ImageTransientDetectionCUDA(
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
        cv::cuda::GpuMat lastRawFrame;
        cv::cuda::GpuMat lastDiffedFrame;
        cv::cuda::GpuMat lastThresholdedFrame;

        cv::cuda::GpuMat averageFrame;
        bool averageFrameSet = false;
        cv::cuda::GpuMat nextAverageFrame;
        uint32_t nextAverageFrameCount;
        

        volatile uint32_t threshold = 1000;
        volatile uint32_t minimumSize = 1;
        volatile uint32_t maximumSize = 100;
        volatile uint32_t numberOfFramesInAverageFrame = 1;
        std::function<void(cv::Mat)> newAverageFrameCallback;

        volatile bool detectionEnabled = true;

        // Benchmark detectTime;
        // Benchmark imageProcessTime;
        // Benchmark transientDetectTime;
        // Benchmark upload;
        // Benchmark add;
        // Benchmark divide;
};

ImageTransientDetectionCUDA::ImageTransientDetectionCUDA(
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

void ImageTransientDetectionCUDA::setNumberOfAverageFrames(uint32_t frames)
{
    this->numberOfFramesInAverageFrame = frames;
}

void ImageTransientDetectionCUDA::setThreshold(uint32_t threshold)
{
    this->threshold = threshold;
}

void ImageTransientDetectionCUDA::setMinimumSize(uint32_t size)
{
    this->minimumSize = size;
}

void ImageTransientDetectionCUDA::setMaximumSize(uint32_t size)
{
    this->maximumSize = size;
}

void ImageTransientDetectionCUDA::setNewAverageFrameCallback(
    std::function<void(cv::Mat)> callback)
{
    this->newAverageFrameCallback = callback;
}

void ImageTransientDetectionCUDA::disableDetection(void)
{
    detectionEnabled = false;
}

void ImageTransientDetectionCUDA::enableDetection(void)
{
    detectionEnabled = true;
}

bool ImageTransientDetectionCUDA::isDetectionEnabled(void)
{
    return detectionEnabled;
}

bool ImageTransientDetectionCUDA::detect(
    cv::Mat &frame,
    cv::Mat &detectionFrame,
    cv::Rect &detectionBox,
    uint32_t &detectionSize)
{


    // detectTime.start();
    // imageProcessTime.start();
    double minValue = 0; 
    double maxValue = 0;
    bool validDetectionSet = false;

    if(this->nextAverageFrame.empty())
    {
        this->nextAverageFrame.create(frame.size(), frame.type());
        this->nextAverageFrame.setTo(cv::Scalar::all(0));

    }
    // upload.start();
    this->lastRawFrame.upload(frame);
    // upload.stop();
    // add.start();
    cv::cuda::add(
            this->lastRawFrame,
            this->nextAverageFrame, 
            this->nextAverageFrame);
    // add.stop();
    // divide.start();
    cv::cuda::divide(this->nextAverageFrame, 2, this->nextAverageFrame);
    // divide.stop();
    this->nextAverageFrameCount++;
    if(this->nextAverageFrameCount >= numberOfFramesInAverageFrame)
    {
        this->nextAverageFrameCount = 0;
        this->nextAverageFrame.copyTo(this->averageFrame);
        this->nextAverageFrame.setTo(cv::Scalar::all(0));
        if(nullptr != this->newAverageFrameCallback)
        {
            cv::Mat image;
            this->averageFrame.download(image);
            this->newAverageFrameCallback(image);
        }    
    }

    // imageProcessTime.stop();
    // transientDetectTime.start();
    if(!this->averageFrame.empty())
    {

        cv::cuda::subtract(this->lastRawFrame, this->averageFrame, this->lastDiffedFrame);

        cv::cuda::minMaxLoc(
            this->lastDiffedFrame, 
            &minValue, 
            &maxValue,
            NULL,
            NULL);
                    
        cv::cuda::threshold(
            this->lastDiffedFrame, 
            this->lastThresholdedFrame, 
            this->threshold, 
            maxValue,
            cv::THRESH_BINARY);

        /* If detection is not enabled, just return false now...*/
        if(false == this->detectionEnabled)
        {
            return false;
        }

        this->lastThresholdedFrame.convertTo(this->lastThresholdedFrame, CV_8U);

        cv::Mat threshold;
        this->lastThresholdedFrame.download(threshold);
        
        int num_labels;
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        num_labels = 
            cv::connectedComponentsWithStats(
                threshold, 
                labels, 
                stats, 
                centroids, 
                8, 
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
                std::cout << "Multiple detections in single image found" << std::endl;
                std::cout << "Try adjusting number of frames in average " <<
                "frame, " << "threshold, minimum size, or maximum size" << std::endl;
            }
            else
            {
                validDetectionSet = true;

                int centerX = stats.at<int>(i, cv::CC_STAT_LEFT) + stats.at<int>(i, cv::CC_STAT_WIDTH) / 2;
                int centerY = stats.at<int>(i, cv::CC_STAT_TOP) + stats.at<int>(i, cv::CC_STAT_HEIGHT) / 2;
                int newX = centerX - 100 / 2;
                int newY = centerY - 100 / 2;

                detectionBox = cv::Rect(
                    newX, 
                    newY,
                    100,
                    100);
                detectionSize = stats.at<int>(i, cv::CC_STAT_AREA);
                this->lastRawFrame.download(detectionFrame);
                detectionFrame = detectionFrame(detectionBox);
            }
        }
    }
    // transientDetectTime.stop();
    // detectTime.stop();
    // upload.print("upload");
    // add.print("add");
    // divide.print("divide");
    // imageProcessTime.print("image processing");
    // transientDetectTime.print("transient detection");
    // detectTime.print("function");
    // std::cout << std::endl;
    return validDetectionSet;
}

void ImageTransientDetectionCUDA::getLastRawFrame(cv::Mat &buffer)
{
    cv::Mat image;
    this->lastRawFrame.download(image);
    buffer = image.clone();
}

void ImageTransientDetectionCUDA::getLastDiffedFrame(cv::Mat &buffer)
{
    cv::Mat image;
    this->lastDiffedFrame.download(image);
    buffer = image.clone();
}

void ImageTransientDetectionCUDA::getLastThresholdedFrame(cv::Mat &buffer)
{
    cv::Mat image;
    this->lastThresholdedFrame.download(image);
    buffer = image.clone();
}

void ImageTransientDetectionCUDA::getAverageFrame(cv::Mat &buffer)
{
    cv::Mat image;
    this->averageFrame.download(image);
    buffer = image.clone();
}