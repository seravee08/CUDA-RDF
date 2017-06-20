#ifndef RGBIMAGE_HPP
#define RGBIMAGE_HPP

#include <rdf/image.hpp>

namespace rdf{
	class RGBImage: public Image{
	public:
		RGBImage(){}
		~RGBImage(){}
		void setRGB(const cv::Mat_<cv::Vec3i>& rgb);
		cv::Mat_<cv::Vec3i> getRGB();
	private:
		cv::Mat_<cv::Vec3i> rgb_;
	};

	typedef boost::shared_ptr<RGBImage> RGBImagePtr;
} // namespace rdf


#endif