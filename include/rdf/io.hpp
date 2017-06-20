#ifndef IO_H
#define IO_H

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>

#include <opencv2/opencv.hpp>

#include <vector>

#include <google/protobuf/message.h>
#include <proto/rdf.pb.h>

namespace rdf {

	class IO{

	public:
		IO(){}
		~IO(){}
		cv::Mat_<float> readDepth(boost::filesystem::path& p);
		cv::Mat_<cv::Vec3i> readRGB(boost::filesystem::path& p);
	    void getIdTs(boost::filesystem::path depth_map_path, std::string& id, std::string& ts);
	    boost::filesystem::path annoPath(const boost::filesystem::path& depth_path);
	    boost::filesystem::path rgbPath(const boost::filesystem::path& depth_path);
	    void writeDepth(boost::filesystem::path p, const cv::Mat_<float>& depth);
	    void writeRGB(boost::filesystem::path p, const cv::Mat_<cv::Vec3i>& rgb);

	private:
	};

	typedef boost::shared_ptr<IO> IOPtr;

}// namespace rdf

#endif