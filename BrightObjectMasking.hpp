#ifndef BRIGHTOBJECTMASKING_H_
#define BRIGHTOBJECTMASKING_H_

/**
 * @file BrightObjectMasking.hpp
 * Copyright (c) 2023 Dale Giancono All rights reserved.
 *
 * @brief
 * TODO add me
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>

class BrightObjectMasking
{
public:
    BrightObjectMasking();
    BrightObjectMasking(
        uint32_t sigma,
        uint32_t erosionKernalSize,
        uint32_t dialationKernalSize) :
        sigma(sigma),
        erosionKernalSize(erosionKernalSize),
        dilationKernalSize(dilationKernalSize)
    {
        this->erosionKernel = cv::Mat::ones(
            this->erosionKernalSize,
            this->erosionKernalSize,
            CV_8UC1);

        this->dialationKernel = cv::Mat::ones(
            this->dilationKernalSize,
            this->dilationKernalSize,
            CV_8UC1);
    }

    void setSigma(uint32_t sigma);
    void setErosionKernalSize(uint32_t erosionKernalSize);
    void setDilationKernalSize(uint32_t dilationKernalSize);
    void enableBrightMaskImageSaving(std::filesystem::path filepath);

    void disableBrightMaskImageSaving();
    void mask(cv::Mat& image);
    cv::Mat getMask(cv::Mat& image);


private:
    uint32_t sigma = 3;
    uint32_t erosionKernalSize = 1;
    uint32_t dilationKernalSize = 1;

    cv::Mat erosionKernel;
    cv::Mat dialationKernel;

    bool saveImages = false;

    std::filesystem::path filepath;

    void validateFilepath()
    {
        if (!std::filesystem::is_directory(filepath))
        {
            throw std::invalid_argument("BrightObjectMasking filepath is not a valid directory path");
        }
    }
};

BrightObjectMasking::BrightObjectMasking()
{

};

void BrightObjectMasking::enableBrightMaskImageSaving(std::filesystem::path filepath)
{
    this->filepath.clear();
    this->filepath = filepath;
    if (true == std::filesystem::is_directory(filepath))
    {
        this->filepath.clear();
        this->filepath = filepath;
        this->saveImages = true;
    }
    else
    {
        std::cerr << "BrightObjectMasking filepath is not a valid directory path" << std::endl;
    }
}

void BrightObjectMasking::disableBrightMaskImageSaving()
{
    this->filepath.clear();
    this->saveImages = false;
}
void BrightObjectMasking::setSigma(uint32_t sigma)
{
    this->sigma = sigma;
}

void BrightObjectMasking::setErosionKernalSize(uint32_t erosionKernalSize)
{
    this->erosionKernalSize = erosionKernalSize;
}

void BrightObjectMasking::setDilationKernalSize(uint32_t dilationKernalSize)
{
    this->dilationKernalSize = dilationKernalSize;
}

void BrightObjectMasking::mask(cv::Mat& image)
{
    cv::Mat mask;
    cv::Scalar mean;
    cv::Scalar std;

    /* Calculate the sigma */
    cv::meanStdDev(image, mean, std);
    int threshold = mean[0] + (std[0] * this->sigma);

    /* Threshold the image */
    cv::threshold(
        image,
        mask,
        threshold,
        255,
        CV_8U);

    /* Erode it to remove small bright spots*/
    cv::erode(mask, mask, this->erosionKernel);

    /* Dialate it to make sure all the bright spots are
        fully covered
    */
    cv::dilate(mask, mask, this->dialationKernel);

    /* Now invert the image to mast these bright spots */
    cv::bitwise_not(mask, mask);

    cv::bitwise_and(image, image, image, mask);
}


cv::Mat BrightObjectMasking::getMask(cv::Mat& image)
{
    cv::Mat mask;
    cv::Scalar mean;
    cv::Scalar std;

    /* Calculate the sigma */
    cv::meanStdDev(image, mean, std);
    int threshold = mean[0] + (std[0] * this->sigma);

    /* Threshold the image */
    cv::threshold(
        image,
        mask,
        threshold,
        255,
        CV_8U);

    /* Erode it to remove small bright spots*/
    cv::erode(mask, mask, this->erosionKernel);

    /* Dialate it to make sure all the bright spots are
        fully covered
    */
    cv::dilate(mask, mask, this->dialationKernel);

    /* Now invert the image to mast these bright spots */
    cv::bitwise_not(mask, mask);

    return mask;
}

#endif
