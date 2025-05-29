#include "StackedImage.hpp"

/**
 * @brief Constructor for StackedImage.
 *
 * This constructor initialises the accumulated exposure us count, and the
 * stack image.
 */
StackedImage::StackedImage()
{
}

/**
 * @brief Sets the target accumulated exposure us for the stack
 *
 * This method sets the target accumulated exposure us for the stack. The stack
 * will be updated when the accumulated exposure us of the stack exceeds this
 * value.
 *
 * @param accumulatedExposure The target accumulated exposure us for the stack
 */
void StackedImage::setStackAccumulatedExposure(
    uint64_t accumulatedExposure)
{
    this->configuredStackAccumulatedExposure = accumulatedExposure;
    this->mode = StackedImage::MODE::ACCUMULATEDEXPOSURE;
}

void StackedImage::setStackNumberOfImages(uint64_t imageCount)
{
    this->configuredStackNumberOfImages = imageCount;
    this->mode = StackedImage::MODE::IMAGECOUNT;
}

void StackedImage::setStackMode(StackedImage::MODE mode)
{
    this->mode = mode;
}

/**
 * @brief Sets the callback function for when a new stack is complete.
 *
 * This method assigns a callback function that will be executed when
 * a new stacked image buffer is complete. The callback function receives
 * the stacked image and the accumulated exposure in microseconds.
 *
 * @param callback A std::function that takes a reference to a cv::Mat
 * and a uint64_t representing the accumulated exposure time. This function
 * will be called when a new stacked image is ready.
 */

void StackedImage::setNewStackCallback(
    std::function<void(cv::Mat &, double, double, uint64_t)> callback)
{
    this->newStackCallback = callback;
}

/**
 * @brief Adds an image to the current stack in progress.
 *
 * This method accumulates the given image and its exposure time into
 * the stack in progress. Once the accumulated exposure time reaches or
 * exceeds the target stack accumulated exposure time, the current stack
 * in progress is finalized, and a new stack is started. If a callback
 * function is set, it is called with the completed stack and its total
 * exposure time.
 *
 * @param image The image to be added to the stack.
 * @param exposureUs The exposure time of the image in microseconds.
 */

void StackedImage::add(
    cv::Mat &image,
    double exposure,
    double brightness)
{
    bool newStackFlag = false;

    if (this->stackInProgress.empty())
    {
        this->stackInProgress =
            cv::Mat::zeros(image.size(), CV_64F);
    }

    cv::accumulate(image, this->stackInProgress);

    this->accumulatedExposure += exposure;
    this->brightnessFactor += brightness;

    this->numberOfImages++;

    if (StackedImage::MODE::ACCUMULATEDEXPOSURE == this->mode)
    {
        if (this->accumulatedExposure >=
            this->configuredStackAccumulatedExposure)
        {
            newStackFlag = true;
        }
    }
    else if (StackedImage::MODE::IMAGECOUNT == this->mode)

    {
        if (this->numberOfImages >= this->configuredStackNumberOfImages)
        {
            newStackFlag = true;
        }
    }

    if (true == newStackFlag)
    {
        this->setStack = this->stackInProgress.clone();
        this->setBrightnessFactor = this->brightnessFactor;
        this->setAccumulatedExposure = this->accumulatedExposure;
        this->setStackNumberOfImagesCount = this->numberOfImages;
        this->isStackSet = true;

        this->accumulatedExposure = 0;
        this->brightnessFactor = 0;
        this->numberOfImages = 0;

        this->stackInProgress =
            cv::Mat::zeros(image.size(), CV_64F);

        if (nullptr != this->newStackCallback)
        {
            this->newStackCallback(
                this->setStack,
                this->setBrightnessFactor,
                this->setAccumulatedExposure,
                this->setStackNumberOfImagesCount);
        }
    }

    return;
}

void StackedImage::add(
    cv::Mat &image,
    double exposure)
{
    this->add(image, exposure, 1.0);
    return;
}

/**
 * @brief
 * Gets the current stack and its accumulated exposure in microseconds.
 *
 * This method copies the current stack into the given buffer and sets the
 * given accumulated exposure time to the current accumulated exposure time.
 *
 * @param buffer A reference to a cv::Mat to store the current stack.
 * @param accumulatedExposure A reference to a uint64_t to store the
 * accumulated exposure time of the current stack in microseconds.
 *
 * @returns
 * true if the stack is available and false otherwise.
 */
bool StackedImage::getStack(
    cv::Mat &buffer,
    double &brightnessFactor,
    double &accumulatedExposure,
    uint64_t &numberOfImages)
{
    if (false == this->isStackSet)
    {
        return false;
    }
    else
    {
        buffer = this->setStack.clone();
        brightnessFactor = this->setBrightnessFactor;
        accumulatedExposure = this->setAccumulatedExposure;
        numberOfImages = this->setStackNumberOfImagesCount;
    }
    return true;
}

/**
 * @brief Gets the current stack
 *
 * This method copies the current stack into the given buffer.
 *
 * @param buffer A reference to a cv::Mat to store the current stack.
 *
 * @returns
 * true if the stack is available and false otherwise.
 */
bool StackedImage::getStack(cv::Mat &buffer)
{
    if (false == this->isStackSet)
    {
        return false;
    }
    else
    {
        buffer = this->setStack.clone();
    }

    return true;
}

/**
 * @brief Resets the stacked image state.
 *
 * This method releases the current stack and stack in progress,
 * sets the stack status to not set, and resets the accumulated
 * exposure time to zero.
 */

void StackedImage::reset(void)
{
    this->setStack.release();
    this->stackInProgress.release();
    this->isStackSet = false;
    this->accumulatedExposure = 0;
    this->brightnessFactor = 0;
    this->numberOfImages = 0;
    this->setAccumulatedExposure = 0;
    this->setBrightnessFactor = 0;
    this->setStackNumberOfImagesCount = 0;
}