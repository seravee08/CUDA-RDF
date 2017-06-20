#include <preprocess/rgbImage.hpp>

namespace preprocess{
	void RGBImage::setRGB(const cv::Mat_<cv::Vec3i>& rgb){
		rgb_ = rgb;

    }

    cv::Mat_<cv::Vec3i> RGBImage::getRGB() const{
    	return rgb_;
    }
}