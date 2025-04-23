#include "BrightObjectMasking.hpp"

/**
 * @brief
 * Default constructor for BrightObjectMasking.
 *
 * @details
 * The default constructor for BrightObjectMasking initialises the object with
 * a standard deviation multiplier of 3, and erosion and dilation kernels of
 * size 1.
 */
BrightObjectMasking::BrightObjectMasking() : sigma(3), erosionKernalSize(1), dilationKernalSize(1)
{
    this->erosionKernel = cv::Mat::ones(this->erosionKernalSize, this->erosionKernalSize, CV_8UC1);
    this->dialationKernel = cv::Mat::ones(this->dilationKernalSize, this->dilationKernalSize, CV_8UC1);
}

/**
 * @brief
 *
 * Constructor for BrightObjectMasking.
 *
 * @param sigma
 * The standard deviation multiplier to be used when calculating the threshold
 * value for the binary image.
 *
 * @param erosionKernalSize
 * The size of the kernel to be used for erosion.
 *
 * @param dialationKernalSize
 * The size of the kernel to be used for dilation.
 *
 * @details
 * The constructor for BrightObjectMasking takes three parameters. The first parameter
 * is the standard deviation multiplier to be used when calculating the threshold
 * value for the binary image. The second parameter is the size of the kernel to be
 * used for erosion. The third parameter is the size of the kernel to be used for
 * dilation. The constructor then initialises the erosion and dilation kernels
 * using cv::Mat::ones.
 */
BrightObjectMasking::BrightObjectMasking(uint32_t sigma, uint32_t erosionKernalSize, uint32_t dialationKernalSize)
    : sigma(sigma), erosionKernalSize(erosionKernalSize), dilationKernalSize(dialationKernalSize)
{
    this->erosionKernel = cv::Mat::ones(this->erosionKernalSize, this->erosionKernalSize, CV_8UC1);
    this->dialationKernel = cv::Mat::ones(this->dilationKernalSize, this->dilationKernalSize, CV_8UC1);
}

/**
 * @brief
 * Set the standard deviation multiplier to be used when calculating
 * the threshold for transient detection.
 *
 * @param sigma
 * The standard deviation multiplier to be used when calculating the
 * threshold for transient detection.
 *
 * @details
 * The standard deviation multiplier is used to calculate the threshold
 * for transient detection. A higher value will result in a lower
 * threshold, and a lower value will result in a higher threshold.
 */
void BrightObjectMasking::setSigma(uint32_t sigma)
{
    this->sigma = sigma;
}

/**
 * @brief
 * Set the size of the erosion kernel.
 *
 * @param erosionKernalSize
 * The size of the erosion kernel to be used.
 *
 * @details
 * The erosion kernel is used to erode the image before thresholding.
 * A larger kernel size will result in a larger area being eroded.
 */
void BrightObjectMasking::setErosionKernalSize(uint32_t erosionKernalSize)
{
    this->erosionKernalSize = erosionKernalSize;
    this->erosionKernel = cv::Mat::ones(this->erosionKernalSize, this->erosionKernalSize, CV_8UC1);
}

/**
 * @brief
 * Set the dilation kernel size.
 *
 * @param dilationKernalSize
 * The size of the dilation kernel to use when masking bright objects.
 *
 * @details
 * The dilation kernel size determines the amount of dilation that is
 * applied to the binary image after thresholding. A larger kernel will
 * result in more dilation, while a smaller kernel will result in less.
 */
void BrightObjectMasking::setDilationKernalSize(uint32_t dilationKernalSize)
{
    this->dilationKernalSize = dilationKernalSize;
    this->dialationKernel = cv::Mat::ones(this->dilationKernalSize, this->dilationKernalSize, CV_8UC1);
}

/**
 * @brief
 * Mask bright objects in the given image.
 *
 * @details
 * This function applies a mask to the given image to remove bright objects.
 * The mask is calculated by thresholding the image at a value of 1 standard
 * deviation above the mean, and then eroding and dilating the thresholded image
 * to remove small bright spots and make sure all the bright spots are fully
 * covered. The mask is then applied to the image by performing a bitwise AND
 * operation.
 *
 * @param image
 * The image to be masked.
 */
void BrightObjectMasking::mask(cv::Mat &image)
{
    cv::Mat mask;
    cv::Scalar mean;
    cv::Scalar std;

    /* Calculate the sigma */
    cv::meanStdDev(image, mean, std);
    int threshold = mean[0] + (std[0] * this->sigma);

    cv::threshold(image, mask, threshold, 255, CV_8U);

    /* Erode it to remove small bright spots*/
    cv::erode(mask, mask, this->erosionKernel);

    /* Dialate it to make sure all the bright spots are
        fully covered */
    cv::dilate(mask, mask, this->dialationKernel);

    /* Now invert the image to mast these bright spots */
    cv::bitwise_not(mask, mask);

    /* Apply the mask*/
    cv::bitwise_and(image, image, image, mask);
}

/**
 * @brief
 * Generate a mask for the given image to remove bright objects.
 *
 * @param image
 * The image to be processed.
 *
 * @return
 * The mask for the given image.
 *
 * @details
 * This function applies a mask to the given image to remove bright objects.
 * The mask is calculated by thresholding the image at a value of 1 standard
 * deviation above the mean, and then eroding and dilating the thresholded image
 * to remove small bright spots and make sure all the bright spots are fully
 * covered. The mask is then inverted to mask out the bright spots.
 */
cv::Mat BrightObjectMasking::getMask(cv::Mat &image)
{
    cv::Mat mask;
    cv::Mat final;
    cv::Scalar mean;
    cv::Scalar std;

    /* Calculate the sigma */
    cv::meanStdDev(image, mean, std);
    int threshold = mean[0] + (std[0] * this->sigma);

    /* Threshold the image */
    cv::threshold(image, mask, threshold, 255, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8U);
    /* Erode it to remove small bright spots*/
    cv::erode(mask, mask, this->erosionKernel);

    /* Dialate it to make sure all the bright spots are
        fully covered */
    cv::dilate(mask, mask, this->dialationKernel);

    /* Now invert the image to mast these bright spots */
    cv::bitwise_not(mask, mask);

    return mask.clone();
}
