#include <rdf/rgbImage.hpp>

namespace rdf{
	void RGBImage::setRGB(const cv::Mat_<cv::Vec3i>& rgb){
		rgb_ = rgb;

    }

    cv::Mat_<cv::Vec3i> RGBImage::getRGB(){
    	return rgb_;
    }
}