#include <rdf/depthImage.hpp>

namespace rdf{
	void DepthImage::setDepth(const cv::Mat_<float>& depth){
		depth_ = depth;
	}

	cv::Mat_<float> DepthImage::getDepth(){
		return depth_;
	}
}