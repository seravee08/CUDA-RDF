#ifndef RGBIMAGE_HPP
#define RGBIMAGE_HPP

#include <preprocess/image.hpp>

namespace preprocess{
	class RGBImage: public Image{
	public:
		RGBImage(){}
		~RGBImage(){}
		void setRGB(const cv::Mat_<cv::Vec3i>& rgb);
		cv::Mat_<cv::Vec3i> getRGB() const;
	private:
		cv::Mat_<cv::Vec3i> rgb_;
	};

	typedef boost::shared_ptr<RGBImage> RGBImagePtr;
}


#endif