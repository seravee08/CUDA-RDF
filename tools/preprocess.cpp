#include <preprocess/anno.hpp>
#include <preprocess/depthImage.hpp>
#include <preprocess/rgbImage.hpp>
#include <preprocess/io.hpp>
#include <preprocess/math.hpp>
#include <util/fileutil.h>

#include <iostream>
#include <vector>
#include <map>

#include <opencv2/opencv.hpp>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

int num;
int radius;
int out_dimension;
std::vector<preprocess::DepthImage> depth_;
std::vector<preprocess::RGBImage> rgb_;
std::vector<preprocess::Anno> anno_;

typedef int (*BrewFunction)();
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap brew_map;

#define RegisterBrewFunction(func) \
class __Register_##func { \
  public:  \
    __Register_##func() { \
    	brew_map[#func] = &func; \
    } \
}; \
__Register_##func register_##func; \

static BrewFunction GetBrewFunction(const std::string& name) {
	if(brew_map.count(name)) {
		return brew_map[name];
	}
	else {
		std::cout << "Available preprocess actions:";
		for(BrewMap::iterator it = brew_map.begin(); it != brew_map.end(); it++) {
			std::cout << "\t" << it->first;
		}
		std::cout << std::endl << "Unknown action: " << name << std::endl;
		return NULL;
	}
}

void readData(std::string type) {

	std::string p = "../data";
	boost::filesystem::path path(p);
	preprocess::IO io_;

	std::vector<boost::filesystem::path> all_depth_paths;
	std::vector<boost::filesystem::path> all_depth_paths_after;
	std::vector<boost::filesystem::path> all_rgb_paths_after;
	std::vector<boost::filesystem::path> all_anno_paths_after;

	addPaths(path, ".*.exr", all_depth_paths);
	int num_assumed = all_depth_paths.size();
	num = num_assumed;

	//decide the actual num of images;
	for(int idx = 0; idx < num_assumed; idx++) {
		boost::filesystem::path depth_path = all_depth_paths[idx];
		boost::filesystem::path rgb_path;
		boost::filesystem::path anno_path;

		if(type == "raw") {
			rgb_path = io_.rgbRawPath(depth_path);
		    if(!boost::filesystem::exists(rgb_path)) {
			    num--;
			    continue;
		    }

		    anno_path = io_.annoRawPath(depth_path);
		    if(!boost::filesystem::exists(anno_path)) {
			    num--;
			    continue;
		    }
		}
		else if(type == "preprocessed") {
			rgb_path = io_.rgbPath(depth_path);
		    if(!boost::filesystem::exists(rgb_path)) {
			    num--;
			    continue;
		    }

		    anno_path = io_.annoPath(depth_path);
		    if(!boost::filesystem::exists(anno_path)) {
			    num--;
			    continue;
		    }
		}
		else {
			std::cerr << "Invalid type!" << std::endl;
		}

		all_depth_paths_after.push_back(depth_path);
		all_rgb_paths_after.push_back(rgb_path);
		all_anno_paths_after.push_back(anno_path);
	}

	if(num < num_assumed) {
		std::cout << "rgb images or annotations are missing!" << std::endl;
	}

	//read data
	depth_.resize(num);
	rgb_.resize(num);
	anno_.resize(num);

	for(int idx = 0; idx < num; idx++) {
		//read data
		boost::filesystem::path depth_path = all_depth_paths_after[idx];
		std::string id, ts;
		io_.getIdTs(depth_path, id, ts);
		cv::Mat_<float> depth;
		if(type == "raw") {
			depth = io_.readRawDepth(depth_path);
		}
		else if(type == "preprocessed") {
			depth = io_.readDepth(depth_path);
		}
		else {
			std::cerr << "Invalid type!" << std::endl;
		}
		depth_[idx].setDepth(depth);
		depth_[idx].setFileNames(id, ts);

		boost::filesystem::path rgb_path = all_rgb_paths_after[idx];
		cv::Mat_<cv::Vec3i> rgb = io_.readRGB(rgb_path);
		rgb_[idx].setRGB(rgb);
		rgb_[idx].setFileNames(id, ts);

		boost::filesystem::path anno_path = all_anno_paths_after[idx];
		std::vector<cv::Vec3f> anno = io_.readAnno(anno_path);
		anno_[idx].setAnno(anno);
		anno_[idx].setFileNames(id, ts);
	}
}

void writeData() {
	boost::filesystem::path out_path("../new_data");
	if(!boost::filesystem::exists(out_path)){
		boost::filesystem::create_directory(out_path);
	}

	preprocess::IO io_;

	//output
	for(int idx = 0; idx < num; idx++) {
		std::string id, ts;
	    depth_[idx].getFileNames(id, ts);

		boost::format fmt_depth("%s_%s_depth.exr");
	    boost::filesystem::path out_path_depth = out_path / (fmt_depth % id % ts).str();
	    io_.writeDepth(out_path_depth, depth_[idx].getDepth());

	    boost::format fmt_rgb("%s_%s_rgb.png");
	    boost::filesystem::path out_path_rgb = out_path / (fmt_rgb % id % ts).str();
	    io_.writeRGB(out_path_rgb, rgb_[idx].getRGB());

	    boost::format fmt_anno("%s_%s_anno.txt");
	    boost::filesystem::path out_path_anno = out_path / (fmt_anno % id % ts).str();
	    io_.writeAnno(out_path_anno, anno_[idx].getAnno());
	}
}


int scale() {
	readData("preprocessed");

	//store all the median depth values
	std::vector<float> medians;
	medians.resize(num);

	for(int idx = 0; idx < num; idx++) {
		float median_dep = preprocess::Math::findHandMedianCoor(depth_[idx], rgb_[idx]);
		medians[idx] = median_dep;
	}

	std::sort(medians.begin(), medians.end());
	float median = medians[num / 2];

	for(int idx = 0; idx < num; idx++) {
		float ratio = preprocess::Math::findHandMedianCoor(depth_[idx], rgb_[idx]) / median;
		preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], ratio);
	}

	writeData();
	return 0;
}
RegisterBrewFunction(scale);


int crop() {
	readData("preprocessed");

	//crop
	for(int idx = 0; idx < num; idx++) {
		preprocess::Math::crop(depth_[idx], rgb_[idx], anno_[idx], radius);
		preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
	}

	writeData();
	return 0;
}
RegisterBrewFunction(crop);

int crop_test_1() {
	readData("preprocessed");
	int rad = radius;
	for(int idx = 0; idx < num; idx++) {
		int temp = preprocess::Math::crop_test_1(depth_[idx], rgb_[idx], anno_[idx], radius);
		rad = std::min(rad, temp);
	}

	std::cout << "The min radius is " << rad << std::endl;
	return 0;
}
RegisterBrewFunction(crop_test_1);

int crop_test_2() {
	readData("preprocessed");
	int sum = 0;
	for(int idx = 0; idx < num; idx++) {
		sum += preprocess::Math::crop_test_2(depth_[idx], rgb_[idx], anno_[idx], radius);
	}

	std::cout << "There are " << sum << " invalid crops!" << std::endl;
	return 0;
}
RegisterBrewFunction(crop_test_2);

int pad() {
	readData("preprocessed");
	for(int idx = 0; idx < num; idx++) {
		preprocess::Math::pad(depth_[idx], rgb_[idx], radius);
	}
	writeData();
	return 0;
}
RegisterBrewFunction(pad);

int filter() {
	readData("preprocessed");

	for(int idx = 0; idx < num; idx++) {
		preprocess::Math::high_filter(depth_[idx]);
		preprocess::Math::normalizeAll(depth_[idx]);
	}

	writeData();
	return 0;
}
RegisterBrewFunction(filter)


int rescale() {
	readData("preprocessed");

	for(int idx = 0; idx < num; idx++) {
		preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], out_dimension);
	}

	writeData();
	return 0;
}
RegisterBrewFunction(rescale)


int rename() {
	readData("raw");
	writeData();
	return 0;
}
RegisterBrewFunction(rename)


int main(int argc, char** argv){
	namespace po = boost::program_options;
	po::options_description desc("options");
	desc.add_options()
		("scale", "scale the images to make them depth invariant.")
		("crop", po::value<int>(&radius), "crop the images centered at the mass of ROI, with the given radius.")
		("filter", "add high-pass filter.")
		("rescale", po::value<int>(&out_dimension), "rescale the images to the given size.")
		("rename", "rename all images for later processing")
		("crop_test_1", po::value<int>(&radius), "crop test 1")
		("crop_test_2", po::value<int>(&radius), "crop test 2")
		("pad", po::value<int>(&radius), "pad the images");

	po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(argc > 1) {
    	if(vm.count("scale")) {
    		std::cout << "Scale!" << std::endl;
    		return GetBrewFunction(std::string("scale"))();
    	}
    	else if(vm.count("crop")) {
    		std::cout << "Crop with radius " << radius << "!" << std::endl;
    		return GetBrewFunction(std::string("crop"))();
    	}
    	else if(vm.count("filter")) {
    		std::cout << "high-pass filter!" << std::endl;
    		return GetBrewFunction(std::string("filter"))();
    	}
    	else if(vm.count("rescale")) {
    		std::cout << "resacle the images to " << out_dimension << " by " << out_dimension << "!" << std::endl;
    		return GetBrewFunction(std::string("rescale"))();
    	}
    	else if(vm.count("rename")) {
    		std::cout << "rename all images!" << std::endl;
    		return GetBrewFunction(std::string("rename"))();
    	}
    	else if(vm.count("crop_test_1")) {
    		std::cout << "crop test 1!" << std::endl;
    		return GetBrewFunction(std::string("crop_test_1"))();
    	}
    	else if(vm.count("crop_test_2")) {
    		std::cout << "crop test 2!" << std::endl;
    		return GetBrewFunction(std::string("crop_test_2"))();
    	}
    	else if(vm.count("pad")) {
    		std::cout << "pad!" << std::endl;
    		return GetBrewFunction(std::string("pad"))();
    	}
    	else {
    		std::cout << "Illegal option!" << std::endl;
    	}
    }
    else {
    	std::cout << "Should have options!" << std::endl;
    }

    return 0;
}
