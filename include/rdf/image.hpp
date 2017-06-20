#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

namespace rdf{
	class Image{
	public:
		Image(){radius_ = 60;}
		~Image(){}
		void setRadius(int radius);
		int getRadius();
		void setCenter(const cv::Vec2i& center);
		cv::Vec2i getCenter();
		void setFileNames(const std::string& id, const std::string& ts);
		void getFileNames(std::string& id, std::string& ts);
	protected:
		cv::Vec2i center_;
		int radius_;
		std::string id_;
		std::string ts_;
	};

	typedef boost::shared_ptr<Image> ImagePtr;
} // namespace rdf


#endif