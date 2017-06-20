#include <preprocess/anno.hpp>


namespace preprocess{
	void Anno::setAnno(const std::vector<cv::Vec3f>& joints){
		joints_.resize(joints.size());
		for(unsigned int idx = 0; idx < joints.size(); ++idx){
			cv::Mat(joints[idx]).copyTo(joints_[idx]);
		}
	}

	std::vector<cv::Vec3f> Anno::getAnno(){
		return joints_;
	}

	void Anno::setFileNames(const std::string& id, const std::string& ts){
		id_ = id;
		ts_ = ts;
	}

	void Anno::getFileNames(std::string& id, std::string& ts){
		id = id_;
		ts = ts_;
	}
}