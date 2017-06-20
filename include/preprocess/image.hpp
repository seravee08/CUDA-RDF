#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

namespace preprocess{
	class Image{
	public:
		Image(){}
		~Image(){}
		void setCenter(const cv::Vec2i& center);
		cv::Vec2i getCenter();
		void setFileNames(const std::string& id, const std::string& ts);
		void getFileNames(std::string& id, std::string& ts);
	protected:
		cv::Vec2i center_;
		std::string id_;
		std::string ts_;
	};

	typedef boost::shared_ptr<Image> ImagePtr;
}


#endif