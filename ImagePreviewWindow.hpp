#ifndef IMAGEPREVIEWWINDOW_H
#define IMAGEPREVIEWWINDOW_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

class ImagePreviewWindow
{
public:
    ImagePreviewWindow(std::string windowName) : name(windowName), statName(windowName + "_stats"), zoomName(windowName + "_zoom")
    {
        cv::namedWindow(name, cv::WINDOW_NORMAL || cv::WindowFlags::WINDOW_KEEPRATIO);
        cv::setMouseCallback(name, ImagePreviewWindow::onMouseWheel, this);
    };

    ImagePreviewWindow(std::string windowName, uint32_t width, uint32_t height) : name(windowName), statName(windowName + "_stats"), zoomName(windowName + "_zoom"), width(width), height(height)
    {
        cv::namedWindow(name, cv::WINDOW_NORMAL || cv::WindowFlags::WINDOW_KEEPRATIO);
        cv::setMouseCallback(name, ImagePreviewWindow::onMouseWheel, this);
        this->setSize(width, height);
    };

    void setSize(uint16_t width, uint16_t height)
    {
        cv::resizeWindow(this->name, width, height);
        this->width = width;
        this->height = height;
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
        this->currentImageMutex.lock();
        currentImage = image.clone();
        this->currentImageMutex.unlock();

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
                image,
                CV_8UC3);
        }
        else
        {
            image.convertTo(
                image,
                CV_8UC1);
        }
        cv::namedWindow(name, cv::WINDOW_NORMAL || cv::WindowFlags::WINDOW_KEEPRATIO);
        cv::imshow(this->name, image);
        if (this->width != 0 && this->height != 0)
        {
            cv::setWindowProperty(this->name, cv::WindowPropertyFlags::WND_PROP_ASPECT_RATIO, this->width / this->height);
            cv::resizeWindow(this->name, this->width, this->height);
        }

        if (true == this->zoomEnabled)
        {
            this->showZoomed();
        }

        if (true == this->statEnabled)
        {
            this->showZoomedStats();
        }

        cv::waitKey(50);
    };

    void setImageStreched(cv::Mat &image, double maxpercent, double minpercent)
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
        this->currentImageMutex.lock();
        currentImage = image.clone();
        this->currentImageMutex.unlock();

        cv::minMaxLoc(image, &statsMin, &statsMax);
        cv::meanStdDev(image, mean, std);

        maxVal = statsMax * maxpercent;
        minVal = statsMin + minpercent;
        double newMaxVal = maxVal + (maxVal - statsMin);
        double newMinVal = minVal + (minVal - statsMin);
        // image.convertTo(
        //     image,
        //     CV_8U,
        //     255.0 / (newMaxVal - statsMin),
        //     -minVal * 255.0 / (newMaxVal - statsMin));

        image.convertTo(
            image,
            CV_8U,
            255.0 / (newMaxVal - statsMin),
            -newMinVal * 255.0 / (newMaxVal - statsMin));

        std::string title = name + " Min: " + std::to_string(static_cast<int>(statsMin)) + " Max: " + std::to_string(static_cast<int>(statsMax)) + " Mean: " + std::to_string(static_cast<int>(mean[0])) + " STD: " + std::to_string(static_cast<int>(std[0]));

        cv::namedWindow(name, cv::WINDOW_NORMAL || cv::WindowFlags::WINDOW_KEEPRATIO);
        cv::setWindowTitle(this->name, title);
        cv::imshow(this->name, image);

        if (this->width != 0 && this->height != 0)
        {
            cv::setWindowProperty(this->name, cv::WindowPropertyFlags::WND_PROP_ASPECT_RATIO, this->width / this->height);
            cv::resizeWindow(this->name, this->width, this->height);
        }

        if (true == this->zoomEnabled)
        {
            this->showZoomed();
        }

        if (true == this->statEnabled)
        {
            this->showZoomedStats();
        }

        cv::waitKey(50);
    }

private:
    cv::Mat currentImage;
    std::mutex currentImageMutex;
    std::string name;

    bool zoomEnabled = false;
    bool statEnabled = false;
    int xzoom = 0;
    int yzoom = 0;
    std::string zoomName;
    std::string statName;
    uint32_t width = 0;
    uint32_t height = 0;
    void showZoomed(void)
    {
        cv::Scalar std;
        cv::Scalar mean;
        double statsMin;
        double statsMax;
        double minVal;
        double maxVal;

        cv::Mat image;
        // Calculate the top-left corner of the ROI, ensuring it's within the image boundaries
        int x1 = std::max(0, this->xzoom - 50);
        int y1 = std::max(0, this->yzoom - 50);

        this->currentImageMutex.lock();
        image = this->currentImage.clone();
        this->currentImageMutex.unlock();

        cv::minMaxLoc(image, &statsMin, &statsMax);
        cv::meanStdDev(image, mean, std);

        maxVal = statsMax;
        double newMaxVal = maxVal + (maxVal - statsMin);
        image.convertTo(
            image,
            CV_8U,
            255.0 / (newMaxVal - statsMin),
            -minVal * 255.0 / (newMaxVal - statsMin));

        // Calculate the bottom-right corner, ensuring it's within the image boundaries
        int x2 = std::min(image.cols - 1, this->xzoom + 50);
        int y2 = std::min(image.rows - 1, this->yzoom + 50);

        // Create a rectangle for the ROI
        cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

        // Extract the ROI from the image
        cv::Mat cropped_img = image(roi).clone();
        cv::resize(cropped_img, cropped_img, cv::Size(400, 400));
        cv::imshow(this->zoomName, cropped_img);
    };

    void showZoomedStats(void)
    {
        cv::Mat image;

        cv::Scalar std;
        cv::Scalar mean;
        double statsMin;
        double statsMax;

        // Calculate the top-left corner of the ROI, ensuring it's within the image boundaries
        int x1 = std::max(0, this->xzoom - 25);
        int y1 = std::max(0, this->yzoom - 25);

        this->currentImageMutex.lock();
        image = this->currentImage.clone();
        this->currentImageMutex.unlock();
        // Calculate the bottom-right corner, ensuring it's within the image boundaries
        int x2 = std::min(image.cols - 1, this->xzoom + 25);
        int y2 = std::min(image.rows - 1, this->yzoom + 25);

        // Create a rectangle for the ROI
        cv::Rect roi(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

        // Extract the ROI from the image
        cv::Mat cropped_img = image(roi).clone();
        cv::meanStdDev(cropped_img, mean, std);
        cv::minMaxLoc(cropped_img, &statsMin, &statsMax);

        cropped_img.convertTo(
            cropped_img,
            CV_8U,
            255.0 / (statsMax - statsMin), -statsMin * 255.0 / (statsMax - statsMin));

        cv::resize(cropped_img, cropped_img, cv::Size(400, 400), 0, 0, cv::INTER_NEAREST);
        std::string title = "Min: " + std::to_string((int)statsMin) + " Max: " + std::to_string((int)statsMax) + " Mean: " + std::to_string((int)mean[0]) + " STD: " + std::to_string((int)std[0]);
        cv::setWindowTitle(this->statName, title);
        cv::imshow(this->statName, cropped_img);
    }

    static void onMouseWheel(int event, int x, int y, int flags, void *userdata)
    {
        ImagePreviewWindow *window = (ImagePreviewWindow *)userdata;

        if (event == cv::EVENT_MOUSEMOVE)
        {
            window->xzoom = x;
            window->yzoom = y;
        }

        else if (event == cv::EVENT_LBUTTONDOWN)
        {
            if (false == window->zoomEnabled)
            {
                window->zoomEnabled = true;
            }
            else
            {
                window->zoomEnabled = false;
                try
                {
                    cv::destroyWindow(window->zoomName);
                }
                catch (std::exception &e)
                {
                }
            }
        }
        else if (event == cv::EVENT_RBUTTONDOWN)
        {
            if (false == window->statEnabled)
            {
                window->statEnabled = true;
            }
            else
            {
                window->statEnabled = false;
                try
                {
                    cv::destroyWindow(window->statName);
                }
                catch (std::exception &e)
                {
                }
            }
        }

        if (true == window->zoomEnabled)
        {
            window->showZoomed();
        }

        if (true == window->statEnabled)
        {
            window->showZoomedStats();
        }
    };
};

#endif // IMAGEPREVIEWWINDOW_H
