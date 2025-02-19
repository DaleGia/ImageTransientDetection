#include "ImageTransientDetection.hpp"

ImageTransientDetection::ImageTransientDetection()
    : sigma(5),
      minimumSize(1),
      maximumSize(4294967295)
{
}

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
 * The maximum size of a transient object. Any contours with more pixels than
 * this will be rejected.
 */
void ImageTransientDetection::setMaximumSize(uint32_t size)
{
    this->maximumSize = size;
}

/**
 * @brief Detect transient objects in two images.
 *
 * This function computes the absolute difference between two input frames,
 * applies a threshold based on a calculated mean and standard deviation, and
 * then finds contours in the thresholded image. It filters these contours
 * based on their area to identify valid transient objects. The function updates
 * the detection box with the bounding box of the detected transients and
 * returns the number of valid transient objects detected.
 *
 * @param frameA The first image frame.
 * @param frameB The second image frame.
 * @param detectionBox A rectangle defining the area of the detected transient object(s).
 * @param maxPixelValue The maximum pixel value found in the transient objects.
 *
 * @return The number of valid transient objects detected.
 */

uint32_t ImageTransientDetection::detect(
    cv::Mat &frameA,
    cv::Mat &frameB,
    cv::Rect &detectionBox,
    double &maxPixelValue)
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
    catch (const std::exception &e)
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
    catch (const std::exception &e)
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
    catch (const std::exception &e)
    {
        std::cerr << "findContours error: " << e.what() << '\n';
    }

    this->numberOfContours = contours.size();
    int minX = std::numeric_limits<int>::max();
    int minY = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::min();
    int maxY = std::numeric_limits<int>::min();
    maxPixelValue = std::numeric_limits<double>::min();

    for (auto &contour : contours)
    {
        double contourArea = cv::contourArea(contour);
        if (contourArea >= this->minimumSize && contourArea <= this->maximumSize)
        {
            validContours.push_back(contour);

            for (const auto &point : contour)
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

        this->absdiffMin = min;
        this->absdiffMax = max;
        this->absdiffMean = mean[0];
        this->absdiffStdDev = std[0];
        this->sigmaThreshold = threshold;
        this->numberOfContours = contours.size();
        this->numberOfValidContours = validContours.size();
        return validContours.size();
    }

    return 0;
}

/**
 * @brief Gets the statistics of the last image that was processed.
 *
 * @return The statistics of the last image that was processed.
 */
ImageTransientDetection::Stats ImageTransientDetection::getLastImageStats() const
{
    ImageTransientDetection::Stats stats;
    stats.absdiffMin = this->absdiffMin;
    stats.absdiffMax = this->absdiffMax;
    stats.absdiffMean = this->absdiffMean;
    stats.absdiffStdDev = this->absdiffStdDev;
    stats.sigmaThreshold = this->sigmaThreshold;
    stats.numberOfContours = this->numberOfContours;
    stats.numberOfValidContours = this->numberOfValidContours;
    return stats;
}