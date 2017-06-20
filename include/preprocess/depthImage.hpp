#ifndef DEPTHIMAGE_HPP
#define DEPTHIMAGE_HPP

#include <preprocess/image.hpp>

namespace preprocess{
	class DepthImage: public Image{
	public:
		DepthImage(){}
		~DepthImage(){}
		void setDepth(const cv::Mat_<float>& depth);
		cv::Mat_<float> getDepth() const;
	private:
		cv::Mat_<float> depth_;

	};

	typedef boost::shared_ptr<DepthImage> DepthImagePtr;
}


#endif