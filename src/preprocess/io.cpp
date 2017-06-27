#include <preprocess/io.hpp>
#include <iostream>
#include <fstream>
#include <preprocess/depthImage.hpp>
#include <preprocess/rgbImage.hpp>

#define bg_value 1.0
#define numJoints 20
namespace preprocess{


	void IO::getIdTs(boost::filesystem::path depth_path, std::string& id, std::string& ts) {
		std::string depth_filename = depth_path.filename().string();

		boost::regex expression("(\\d+)_(\\d+)_depth.*");
		boost::smatch what;
		if (boost::regex_match(depth_filename, what, expression, boost::match_extra)) {
			id = what[1].str();
			ts = what[2].str();
		}
	}

	void IO::getIdTsCh(boost::filesystem::path depth_path, std::string& id, std::string& ts, std::string& ch) {
		std::string depth_filename = depth_path.filename().string();

		boost::regex expression("(\\d+)_(\\d+)_depth_(\\d+).exr");
		boost::smatch what;
		if (boost::regex_match(depth_filename, what, expression, boost::match_extra)) {
			id = what[1].str();
			ts = what[2].str();
			ch = what[3].str();
		}
	}

	boost::filesystem::path IO::annoPath(const boost::filesystem::path& depth_path){
		std::string id, ts;
		getIdTs(depth_path, id, ts);

		boost::format fmt("%s_%s_anno.txt");
		boost::filesystem::path parent = depth_path.parent_path().parent_path();
		parent /= "label";
		return parent / (fmt % id % ts).str();
	}

	boost::filesystem::path IO::annoRawPath(const boost::filesystem::path& depth_path){
		std::string id, ts;
		getIdTs(depth_path, id, ts);

		boost::format fmt("%s_%s_anno_blender.txt");
		return depth_path.parent_path() / (fmt % id % ts).str();
	}

	boost::filesystem::path IO::rgbPath(const boost::filesystem::path& depth_path){
		std::string id, ts;
		getIdTs(depth_path, id, ts);

		boost::format fmt("%s_%s_rgb.png");
		boost::filesystem::path parent = depth_path.parent_path().parent_path();
		parent /= "mask";
		return parent / (fmt % id % ts).str();
	}

	boost::filesystem::path IO::rgbRawPath(const boost::filesystem::path& depth_path){
		std::string id, ts, ch;
		getIdTsCh(depth_path, id, ts, ch);

		boost::format fmt("%s_%s_depth_%s.png");
		return depth_path.parent_path() / (fmt % id % ts % ch).str();
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
		for (int row = 0; row < depth.rows; ++row){
			for (int col = 0; col < depth.cols; ++col){
				float d = depth(row, col);
				if (d < 0 || d > bg_value)
					d = bg_value;
				depth(row, col) = d;
			}
		}
		return depth;
	}

	cv::Mat_<float> IO::readRawDepth(boost::filesystem::path& p){
		cv::Mat_<cv::Vec3f> exrC3 = cv::imread(p.string(), -1);
		std::vector<cv::Mat_<float> > exrChannels;
		cv::split(exrC3, exrChannels);
		cv::Mat_<float> depth = exrChannels[0];

		return depth;
	}

	cv::Mat_<float> IO::readWrongDepth(boost::filesystem::path& p){
		cv::Mat_<float> depth = cv::imread(p.string(), -1);

		return depth;
	}

	std::vector<cv::Vec2f> IO::readAnno(boost::filesystem::path& p){
		std::vector<cv::Vec2f> joints(numJoints);
		std::ifstream fin(p.c_str());

		for (unsigned int joint_idx = 0; joint_idx < joints.size(); ++joint_idx) {
			fin >> joints[joint_idx][0];
			fin >> joints[joint_idx][1];
		}

		fin.close();

		return joints;
	}

	void IO::writeDepth(boost::filesystem::path p, const cv::Mat_<float>& depth){
		std::string cp = p.string();

		//merge to 3 channels to suppress error
		cv::Mat_<cv::Vec3f> depth_3channels;
		std::vector<cv::Mat_<float> > channels;
		channels.push_back(depth);
		channels.push_back(depth);
		channels.push_back(depth);
		cv::merge(channels, depth_3channels);

		imwrite(cp, depth_3channels);
	}


	void IO::writeRGB(boost::filesystem::path p, const cv::Mat_<cv::Vec3i>& rgb){
		std::string cp = p.string();
		imwrite(cp, rgb);
	}


	void IO::writeAnno(boost::filesystem::path p, const std::vector<cv::Vec2f>& anno){
		std::ofstream fout(p.c_str());

		for (unsigned int joint_idx = 0; joint_idx < anno.size(); ++joint_idx){
			for (int idx = 0; idx < 2; ++idx){
				fout << anno[joint_idx][idx];
				fout << " ";
			}
		}

		fout.close();
	}

}