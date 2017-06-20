#include <preprocess/depthImage.hpp>

namespace preprocess{
	void DepthImage::setDepth(const cv::Mat_<float>& depth){
		depth_ = depth;
	}

	cv::Mat_<float> DepthImage::getDepth() const{
		return depth_;
	}
}