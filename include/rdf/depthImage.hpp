#ifndef DEPTHIMAGE_HPP
#define DEPTHIMAGE_HPP

#include <rdf/image.hpp>

namespace rdf{
	class DepthImage: public Image{
	public:
		DepthImage(){}
		~DepthImage(){}
		void setDepth(const cv::Mat_<float>& depth);
		cv::Mat_<float> getDepth();
	private:
		cv::Mat_<float> depth_;

	};

	typedef boost::shared_ptr<DepthImage> DepthImagePtr;
} // namespace rdf


#endif