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

class ImageTransientDetection
{
    public:
        ImageTransientDetection()
        {
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
            cv::Rect &detectionBox);

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
        

        volatile uint32_t threshold = 1000;
        volatile uint32_t minimumSize = 1;
        volatile uint32_t maximumSize = 100;
        volatile uint32_t numberOfFramesInAverageFrame = 1;
        std::function<void(cv::Mat)> newAverageFrameCallback;

        volatile bool detectionEnabled = true;
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
    cv::Rect &detectionBox)
{
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
        this->averageFrame = 
            cv::Mat::zeros(frame.size(), frame.type()); 
        this->averageFrame = this->nextAverageFrame.clone();
        this->nextAverageFrame = 
            cv::Mat::zeros(frame.size(), frame.type());        

        if(nullptr != this->newAverageFrameCallback)
        {
            this->newAverageFrameCallback(this->averageFrame.clone());
        }    
    }

    if(!this->averageFrame.empty())
    {
        /* This is the actual detection */
        cv::Mat subtractedFrame;
        cv::Mat thresholdFrame;

        double minValue; 
        double maxValue;
        
        cv::subtract(frame, this->averageFrame, subtractedFrame);
        this->lastDiffedFrame = subtractedFrame.clone();

        cv::minMaxLoc(
            subtractedFrame, 
            &minValue, 
            &maxValue);
        cv::threshold(
            subtractedFrame, 
            thresholdFrame, 
            this->threshold, 
            maxValue,
            cv::THRESH_BINARY);
        this->lastThresholdedFrame = thresholdFrame.clone();

        /* If detection is not enabled, just return false now...*/
        if(false == this->detectionEnabled)
        {
            return false;
        }

        thresholdFrame.convertTo(thresholdFrame, CV_8U);

        
        int num_labels;
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        num_labels = 
            cv::connectedComponentsWithStats(
                thresholdFrame, 
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
                detectionFrame = frame(detectionBox);
            }
        }
    }

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