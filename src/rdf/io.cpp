#include <rdf/io.hpp>
#include <iostream>
#include <rdf/depthImage.hpp>
#include <rdf/rgbImage.hpp>


#define bg_value 1.0
namespace rdf{


void IO::getIdTs(boost::filesystem::path depth_path, std::string& id, std::string& ts) {
    std::string depth_filename = depth_path.filename().string();
    
    boost::regex expression("(\\d+)_(\\d+)_depth.*");
    boost::smatch what;
    if(boost::regex_match(depth_filename, what, expression, boost::match_extra)) {
        id = what[1].str();
        ts = what[2].str();
    }
}


boost::filesystem::path IO::rgbPath(const boost::filesystem::path& depth_path){
  std::string id, ts;
  getIdTs(depth_path, id, ts);
  
  boost::format fmt("%s_%s_rgb.png");
  return depth_path.parent_path() / (fmt % id % ts).str();
}


cv::Mat_<cv::Vec3i> IO::readRGB(boost::filesystem::path& p){
	cv::Mat_<cv::Vec3i> pngC3 = cv::imread(p.string(), 1);
	return pngC3;
}

cv::Mat_<float> IO::readDepth(boost::filesystem::path& p){
	cv::Mat_<cv::Vec3f> exrC3 = cv::imread(p.string(), -1);
	std::vector<cv::Mat_<float> > exrChannels;
	cv::split(exrC3, exrChannels);
	cv::Mat_<float> depth = exrChannels[0];

	//postprocess
	for(int row = 0; row < depth.rows; ++row){
		for(int col = 0; col < depth.cols; ++col){
			float d = depth(row, col);
			if(d < 0 || d > bg_value)
				d = bg_value;
			depth(row, col) = d;
		}
	}
	return depth;
}

void IO::writeDepth(boost::filesystem::path p, const cv::Mat_<float>& depth){
	std::string cp = p.string();
	imwrite(cp, depth);
}


void IO::writeRGB(boost::filesystem::path p, const cv::Mat_<cv::Vec3i>& rgb){
	std::string cp = p.string();
	imwrite(cp, rgb);
}

}

