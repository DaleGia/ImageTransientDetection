#ifndef IMAGEPREVIEWWINDOW_H
#define IMAGEPREVIEWWINDOW_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

class ImagePreviewWindow 
{
    public:
        ImagePreviewWindow(std::string windowName) : name(windowName)
        {
        };

        void setImage(cv::Mat &image)
        {
            /* Convert the image to a strech RGBA image */
            double minVal, maxVal;
            bool isColour;
            int srcType;

            if(image.empty())
            {
                return;
            }
            
            srcType = image.type();
            // Check for common color and grayscale formats:
            isColour = 
                image.type() == 
                    CV_8UC3 || srcType == CV_16UC3 || srcType == CV_32FC3;

            
            cv::minMaxLoc(image, &minVal, &maxVal);

            if (true == isColour) 
            {
                image.convertTo(
                    currentImage, 
                    CV_8UC3);          
            } 
            else 
            {
                image.convertTo(
                    currentImage, 
                    CV_8UC1);     
            }

            cv::imshow(name ,currentImage);
            cv::waitKey(1);
        };

        void setImageStreched(cv::Mat &image)
        {
            /* Convert the image to a strech RGBA image */
            double minVal, maxVal;
            bool isColour;
            int srcType;

            if(image.empty())
            {
                return;
            }
            
            srcType = image.type();
            // Check for common color and grayscale formats:
            isColour = 
                image.type() == 
                    CV_8UC3 || srcType == CV_16UC3 || srcType == CV_32FC3;

            
            cv::minMaxLoc(image, &minVal, &maxVal);

            if (true == isColour) 
            {
                image.convertTo(
                    currentImage, 
                    CV_8UC3, 
                    255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));          
            } 
            else 
            {
                image.convertTo(
                    currentImage, 
                    CV_8UC1, 
                    255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));     
            }

            cv::imshow(name ,currentImage);
            cv::waitKey(1);
        }

    private:
        cv::Mat currentImage;
        std::string name;
};

#endif // IMAGEPREVIEWWINDOW_H
