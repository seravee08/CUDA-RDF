#include <rdf/sample.hpp>

namespace rdf {

	Sample::Sample(int x, int y, int idx):label_(-1) {
		coor_ = cv::Point2i(x,y);
		idx_  = idx;
	}

	Sample::Sample(cv::Point2i coor):idx_(-1), label_(-1) {
		coor_ = coor;
	}

	void Sample::setCoor(int x, int y) {
		coor_ = cv::Point2i(x,y);
	}

	void Sample::setCoor(const cv::Point2i& coor){
		coor_ = coor;
	}

	cv::Point2i Sample::getCoor() const {
		return coor_;
	}

	void Sample::setIdx(int idx){
		idx_ = idx;
	}

	int Sample::getIdx() const {
		return idx_;
	}

	void Sample::setLabel(int label) {
		label_ = label;
	}

	int Sample::getLabel() const {
		return label_;
	}

	void Sample::setDepth(const cv::Mat_<float>& depth) {
		depth_ = depth;
	}

	cv::Mat_<float> Sample::getDepth() const {
		return depth_;
	}

	float Sample::getDepth(int row, int col) const {
		return depth_(row,col);
	}

	void Sample::setDepthID(int id) {
		depth_id = id;
	}

	int Sample::getDepthID() const {
		return depth_id;
	}

	void Sample::setRGB(const cv::Mat_<cv::Vec3i>& rgb){
		rgb_ = rgb;
	}

	cv::Mat_<cv::Vec3i> Sample::getRGB() const {
		return rgb_;
	}

	cv::Vec3i Sample::getRGB(int row, int col) const {
		return rgb_(row,col);
	}

} // namespace rdf
