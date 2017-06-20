#ifndef ANNO_HPP
#define ANNO_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

namespace preprocess{
	class Anno{
	public:
		Anno(){numJoints = 20; joints_.resize(numJoints);}
		~Anno(){}
		void setAnno(const std::vector<cv::Vec3f>& joints);
		std::vector<cv::Vec3f> getAnno();
		void setFileNames(const std::string& id, const std::string& ts);
		void getFileNames(std::string& id, std::string& ts);
	private:
		std::vector<cv::Vec3f> joints_;
		int numJoints;
		std::string id_;
		std::string ts_;
	};

	typedef boost::shared_ptr<Anno> AnnoPtr;
}


#endif