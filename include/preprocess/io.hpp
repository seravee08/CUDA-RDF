#ifndef IO_H
#define IO_H

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

namespace preprocess {

	class IO{

	public:
		IO(){}
		~IO(){}
		cv::Mat_<float> readDepth(boost::filesystem::path& p);
		cv::Mat_<float> readRawDepth(boost::filesystem::path& p);
		cv::Mat_<cv::Vec3i> readRGB(boost::filesystem::path& p);
		std::vector<cv::Vec3f> readAnno(boost::filesystem::path& p);
	    void getIdTs(boost::filesystem::path depth_path, std::string& id, std::string& ts);
	    void getIdTsCh(boost::filesystem::path depth_path, std::string& id, std::string& ts, std::string& ch);
	    boost::filesystem::path annoPath(const boost::filesystem::path& depth_path);
	    boost::filesystem::path annoRawPath(const boost::filesystem::path& depth_path);
	    boost::filesystem::path rgbPath(const boost::filesystem::path& depth_path);
	    boost::filesystem::path rgbRawPath(const boost::filesystem::path& depth_path);
	    void writeDepth(boost::filesystem::path p, const cv::Mat_<float>& depth);
	    void writeRGB(boost::filesystem::path p, const cv::Mat_<cv::Vec3i>& rgb);
	    void writeAnno(boost::filesystem::path p, const std::vector<cv::Vec3f>& anno);

	private:
	};

}// namespace utils

#endif