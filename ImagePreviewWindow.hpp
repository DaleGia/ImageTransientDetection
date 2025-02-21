#ifndef IMAGEPREVIEWWINDOW_H
#define IMAGEPREVIEWWINDOW_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

class ImagePreviewWindow
{
public:
    ImagePreviewWindow(std::string windowName) : name(windowName) {
                                                 };

    void setSize(uint16_t width, uint16_t height)
    {
        cv::setWindowProperty(this->name, cv::WindowPropertyFlags::WND_PROP_ASPECT_RATIO, width / height);
        cv::resizeWindow(this->name, width, height);
    }

    void setImage(cv::Mat &image)
    {
        /* Convert the image to a strech RGBA image */
        double minVal, maxVal;
        bool isColour;
        int srcType;

        if (image.empty())
        {
            return;
        }

        this->currentImage = image.clone();

        srcType = image.type();
        // Check for common color and grayscale formats:
        isColour =
            image.type() ==
                CV_8UC3 ||
            srcType == CV_16UC3 || srcType == CV_32FC3;

        cv::minMaxLoc(image, &minVal, &maxVal);

        if (true == isColour)
        {
            image.convertTo(
                this->currentImage,
                CV_8UC3);
        }
        else
        {
            image.convertTo(
                this->currentImage,
                CV_8UC1);
        }

        cv::imshow(this->name, currentImage);
        cv::waitKey(1);
    };

    void setImageStreched(cv::Mat &image, double percent)
    {
        /* Convert the image to a strech RGBA image */
        double minVal;
        double maxVal;
        double statsMin;
        double statsMax;
        cv::Scalar std;
        cv::Scalar mean;
        bool isColour;
        int srcType;

        if (image.empty())
        {
            return;
        }

        currentImage = image.clone();

        srcType = image.type();
        // Check for common color and grayscale formats:
        isColour =
            image.type() ==
                CV_8UC3 ||
            srcType == CV_16UC3 || srcType == CV_32FC3;

        cv::minMaxLoc(image, &statsMin, &statsMax);
        cv::meanStdDev(image, mean, std);

        maxVal = statsMax * percent;
        double newMaxVal = maxVal + (maxVal - statsMin);

        if (true == isColour && CV_8UC3)
        {
            image.convertTo(
                this->currentImage,
                CV_8UC3,
                255.0 / (newMaxVal - minVal), -minVal * 255.0 / (newMaxVal - minVal));
        }
        else if (true == isColour && CV_16UC3)
        {
            image.convertTo(
                this->currentImage,
                CV_8UC3,
                65525.0 / (newMaxVal - minVal), -minVal * 65525.0 / (newMaxVal - minVal));
        }
        else if (image.type() == CV_8UC1)
        {
            image.convertTo(
                this->currentImage,
                CV_8UC1,
                255.0 / (newMaxVal - minVal), -minVal * 255.0 / (newMaxVal - minVal));
        }
        else if (image.type() == CV_16UC1)
        {
            image.convertTo(
                this->currentImage,
                CV_8UC1,
                4095.0 / (newMaxVal - minVal), -minVal * 4095.0 / (newMaxVal - minVal));
        }
        else
        {
            cv::normalize(image, this->currentImage, 0, 255, cv::NORM_MINMAX);
            this->currentImage.convertTo(
                this->currentImage,
                CV_8UC3);
        }
        std::string title = name + " Min: " + std::to_string(static_cast<int>(statsMin)) + " Max: " + std::to_string(static_cast<int>(statsMax)) + " Mean: " + std::to_string(static_cast<int>(mean[0])) + " STD: " + std::to_string(static_cast<int>(std[0]));

        cv::setWindowTitle(this->name, title);
        cv::imshow(this->name, this->currentImage);
        cv::waitKey(1);
    }

private:
    cv::Mat currentImage;
    std::string name;
};

#endif // IMAGEPREVIEWWINDOW_H
