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
int outSize;
int idx = 0;
std::vector<preprocess::DepthImage> depth_;
std::vector<preprocess::RGBImage> rgb_;
std::vector<preprocess::Anno> anno_;
#define batchSize 3000
int left;

typedef int(*BrewFunction)();
typedef std::map<std::string, BrewFunction> BrewMap;
BrewMap brew_map;

std::vector<boost::filesystem::path> all_depth_paths;
std::vector<boost::filesystem::path> all_depth_paths_after;
std::vector<boost::filesystem::path> all_rgb_paths_after;
std::vector<boost::filesystem::path> all_anno_paths_after;


#define RegisterBrewFunction(func) \
class __Register_##func { \
  public:  \
    __Register_##func() { \
    	brew_map[#func] = &func; \
    } \
}; \
__Register_##func register_##func; \

static BrewFunction GetBrewFunction(const std::string& name) {
	if (brew_map.count(name)) {
		return brew_map[name];
	}
	else {
		std::cout << "Available preprocess actions:";
		for (BrewMap::iterator it = brew_map.begin(); it != brew_map.end(); it++) {
			std::cout << "\t" << it->first;
		}
		std::cout << std::endl << "Unknown action: " << name << std::endl;
		return NULL;
	}
}

void readPath(std::string type) {
	std::string p = "../data";
	if (type == "preprocessed") {
		p += "/image";
	}
	boost::filesystem::path path(p);
	preprocess::IO io_;

	addPaths(path, ".*.exr", all_depth_paths);
	int num_assumed = all_depth_paths.size();
	num = num_assumed;

	//decide the actual num of images;
	for (int idx = 0; idx < num_assumed; idx++) {
		boost::filesystem::path depth_path = all_depth_paths[idx];
		boost::filesystem::path rgb_path;
		boost::filesystem::path anno_path;

		if (type == "raw") {
			rgb_path = io_.rgbRawPath(depth_path);
			if (!boost::filesystem::exists(rgb_path)) {
				num--;
				continue;
			}

			anno_path = io_.annoRawPath(depth_path);
			if (!boost::filesystem::exists(anno_path)) {
				num--;
				continue;
			}
		}
		else if (type == "preprocessed") {
			rgb_path = io_.rgbPath(depth_path);
			if (!boost::filesystem::exists(rgb_path)) {
				num--;
				continue;
			}

			anno_path = io_.annoPath(depth_path);
			if (!boost::filesystem::exists(anno_path)) {
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

	if (num < num_assumed) {
		std::cout << "rgb images or annotations are missing!" << std::endl;
	}
}

void readData(std::string type, const int& start, const bool& last) {
	//read data
	depth_.resize(batchSize);
	rgb_.resize(batchSize);
	anno_.resize(batchSize);

	preprocess::IO io_;
	int batch_size = !last ? batchSize : left;

	for (int idx = 0; idx < batch_size; idx++) {
		//read data
		boost::filesystem::path depth_path = all_depth_paths_after[idx + start];
		std::string id, ts;
		io_.getIdTs(depth_path, id, ts);
		cv::Mat_<float> depth;
		if (type == "raw") {
			depth = io_.readRawDepth(depth_path);
		}
		else if (type == "preprocessed") {
			depth = io_.readRawDepth(depth_path);
		}
		else {
			std::cerr << "Invalid type!" << std::endl;
		}
		depth_[idx].setDepth(depth);
		depth_[idx].setFileNames(id, ts);

		boost::filesystem::path rgb_path = all_rgb_paths_after[idx + start];
		cv::Mat_<cv::Vec3i> rgb = io_.readRGB(rgb_path);
		rgb_[idx].setRGB(rgb);
		rgb_[idx].setFileNames(id, ts);

		boost::filesystem::path anno_path = all_anno_paths_after[idx + start];
		std::vector<cv::Vec2f> anno = io_.readAnno(anno_path);
		anno_[idx].setAnno(anno);
		anno_[idx].setFileNames(id, ts);
	}
}

void readWrongData(std::string type, const int& start, const bool& last) {
	//read data
	depth_.resize(batchSize);
	rgb_.resize(batchSize);
	anno_.resize(batchSize);

	preprocess::IO io_;
	int batch_size = !last ? batchSize : left;

	for (int idx = 0; idx < batch_size; idx++) {
		//read data
		boost::filesystem::path depth_path = all_depth_paths_after[idx + start];
		std::string id, ts;
		io_.getIdTs(depth_path, id, ts);
		cv::Mat_<float> depth;
		if (type == "raw") {
			depth = io_.readWrongDepth(depth_path);
		}
		else if (type == "preprocessed") {
			depth = io_.readWrongDepth(depth_path);
		}
		else {
			std::cerr << "Invalid type!" << std::endl;
		}
		depth_[idx].setDepth(depth);
		depth_[idx].setFileNames(id, ts);

		boost::filesystem::path rgb_path = all_rgb_paths_after[idx + start];
		cv::Mat_<cv::Vec3i> rgb = io_.readRGB(rgb_path);
		rgb_[idx].setRGB(rgb);
		rgb_[idx].setFileNames(id, ts);

		boost::filesystem::path anno_path = all_anno_paths_after[idx + start];
		std::vector<cv::Vec2f> anno = io_.readAnno(anno_path);
		anno_[idx].setAnno(anno);
		anno_[idx].setFileNames(id, ts);
	}
}

void freeData() {
	depth_.clear();
	rgb_.clear();
	anno_.clear();
}


// void readRGB() {
// 	std::string p = "../data";
// 	boost::filesystem::path path(p);
// 	preprocess::IO io_;

// 	std::vector<boost::filesystem::path> all_rgb_paths;

// 	addPaths(path, ".*.png", all_rgb_paths);
// 	num = all_rgb_paths.size();

// 	//read data
// 	rgb_.resize(num);

// 	for(int idx = 0; idx < num; idx++) {
// 		//read data
// 		boost::filesystem::path rgb_path = all_rgb_paths[idx];
// 		std::string id, ts;
// 		io_.getIdTs(rgb_path, id, ts);
// 		cv::Mat_<cv::Vec3i> rgb;

// 		rgb = io_.readRGB(rgb_path);
// 		rgb_[idx].setRGB(rgb);
// 		rgb_[idx].setFileNames(id, ts);
// 	}
// }


void writeData(const bool& last) {
	boost::filesystem::path out_path("../new_data");
	if (!boost::filesystem::exists(out_path)){
		boost::filesystem::create_directory(out_path);
	}
	boost::filesystem::path out_path_image("../new_data/image");
	if (!boost::filesystem::exists(out_path_image)){
		boost::filesystem::create_directory(out_path_image);
	}
	boost::filesystem::path out_path_mask("../new_data/mask");
	if (!boost::filesystem::exists(out_path_mask)){
		boost::filesystem::create_directory(out_path_mask);
	}
	boost::filesystem::path out_path_label("../new_data/label");
	if (!boost::filesystem::exists(out_path_label)){
		boost::filesystem::create_directory(out_path_label);
	}

	preprocess::IO io_;

	int batch_size = !last ? batchSize : left;

	//output
	for (int idx = 0; idx < batch_size; idx++) {
		std::string id, ts;
		depth_[idx].getFileNames(id, ts);

		boost::format fmt_depth("%s_%s_depth.exr");
		boost::filesystem::path out_path_depth = out_path_image / (fmt_depth % id % ts).str();
		io_.writeDepth(out_path_depth, depth_[idx].getDepth());

		boost::format fmt_rgb("%s_%s_rgb.png");
		boost::filesystem::path out_path_rgb = out_path_mask / (fmt_rgb % id % ts).str();
		io_.writeRGB(out_path_rgb, rgb_[idx].getRGB());

		boost::format fmt_anno("%s_%s_anno.txt");
		boost::filesystem::path out_path_anno = out_path_label / (fmt_anno % id % ts).str();
		io_.writeAnno(out_path_anno, anno_[idx].getAnno());

		//    boost::format fmt_jpg("%s_%s.jpg");
		//    boost::filesystem::path folder("/home/zhi/HandTracking/RDF/filter/");
		//    if(!boost::filesystem::exists(folder)){
		// 	boost::filesystem::create_directory(folder);
		// }
		//   	boost::filesystem::path jpg_path = folder / (fmt_jpg % id % ts).str();
		//   	cv::Mat_<float> dd = depth_[idx].getDepth();
		//   	cv::normalize(dd, dd, 0, 255, cv::NORM_MINMAX);
		//   	imwrite(jpg_path.c_str(), dd);
	}
}

void writeSingleData(const int& idx) {
	boost::filesystem::path out_path("../new_data");
	if (!boost::filesystem::exists(out_path)){
		boost::filesystem::create_directory(out_path);
	}
	boost::filesystem::path out_path_image("../new_data/image");
	if (!boost::filesystem::exists(out_path_image)){
		boost::filesystem::create_directory(out_path_image);
	}
	boost::filesystem::path out_path_mask("../new_data/mask");
	if (!boost::filesystem::exists(out_path_mask)){
		boost::filesystem::create_directory(out_path_mask);
	}
	boost::filesystem::path out_path_label("../new_data/label");
	if (!boost::filesystem::exists(out_path_label)){
		boost::filesystem::create_directory(out_path_label);
	}
	//output
	std::string id, ts;
	depth_[idx].getFileNames(id, ts);
	preprocess::IO io_;

	boost::format fmt_depth("%s_%s_depth.exr");
	boost::filesystem::path out_path_depth = out_path_image / (fmt_depth % id % ts).str();
	io_.writeDepth(out_path_depth, depth_[idx].getDepth());

	boost::format fmt_rgb("%s_%s_rgb.png");
	boost::filesystem::path out_path_rgb = out_path_mask / (fmt_rgb % id % ts).str();
	io_.writeRGB(out_path_rgb, rgb_[idx].getRGB());

	boost::format fmt_anno("%s_%s_anno.txt");
	boost::filesystem::path out_path_anno = out_path_label / (fmt_anno % id % ts).str();
	io_.writeAnno(out_path_anno, anno_[idx].getAnno());
}


// int merge() {
// 	readRGB();
// 	int row = rgb_[0].getRGB().rows;
// 	int col = rgb_[0].getRGB().cols;
// 	cv::Mat_<int> image(row, col, 0);

// 	for(int idx = 0; idx < num; idx++) {
// 		cv::Mat_<cv::Vec3i> rgb = rgb_[idx].getRGB();
// 		for(int i = 0; i < row; i++) {
// 			for(int j = 0; j < col; j++) {
// 				if(!(rgb(i, j)[0] == 0 && rgb(i, j)[1] == 0 && rgb(i, j)[2] == 255)) {
// 					rgb(i, j)[2] = 0;
// 				}
// 			}
// 		}

// 		std::vector<cv::Mat_<int> > rgbChannels;
// 	    cv::split(rgb, rgbChannels);
// 	    cv::Mat_<int> r = rgbChannels[2];

// 	    preprocess::Math::merge(r, image);
// 	}

// 	for(int i = 0; i < row; i++) {
// 		for(int j = 0; j < col; j++) {
// 			image(i, j) /= num;
// 		}
// 	}

// 	boost::filesystem::path out_file("../mean.png");
// 	cv::imwrite(out_file.c_str(), image);
// 	return 0;
// }
// RegisterBrewFunction(merge);


int scale() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for (int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		//store all the mean depth values
		std::vector<float> means;
		means.resize(batchSize);

		for (int idx = 0; idx < batchSize; idx++) {
			float mean_dep = preprocess::Math::findHandMeanDep(depth_[idx], rgb_[idx]);
			means[idx] = mean_dep;
		}

		std::sort(means.begin(), means.end());
		//float median = means[batchSize / 2];
		float median = means[batchSize / 6];

		for (int idx = 0; idx < batchSize; idx++) {
			float ratio = preprocess::Math::findHandMeanDep(depth_[idx], rgb_[idx]) / median;
			preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], ratio);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		readData("preprocessed", count * batchSize, last);

		//store all the mean depth values
		std::vector<float> means;
		means.resize(left);

		for (int idx = 0; idx < left; idx++) {
			float mean_dep = preprocess::Math::findHandMeanDep(depth_[idx], rgb_[idx]);
			means[idx] = mean_dep;
		}

		std::sort(means.begin(), means.end());
		//float median = means[left / 2];
		float median = means[left / 6];

		for (int idx = 0; idx < left; idx++) {
			float ratio = preprocess::Math::findHandMeanDep(depth_[idx], rgb_[idx]) / median;
			preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], ratio);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}
	return 0;
}
RegisterBrewFunction(scale);


// int crop() {
// 	readData("preprocessed");

// 	//crop
// 	for(int idx = 0; idx < num; idx++) {
// 		preprocess::Math::crop(depth_[idx], rgb_[idx], anno_[idx], radius);
// 		preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
// 	}

// 	writeData();
// 	return 0;
// }
// RegisterBrewFunction(crop);


// int crop_test_1() {
// 	readData("preprocessed");
// 	int rad = radius;
// 	for(int idx = 0; idx < num; idx++) {
// 		int temp = preprocess::Math::crop_test_1(depth_[idx], rgb_[idx], anno_[idx], radius);
// 		rad = std::min(rad, temp);
// 	}

// 	std::cout << "The min radius is " << rad << std::endl;
// 	return 0;
// }
// RegisterBrewFunction(crop_test_1);


// int crop_test_2() {
// 	readData("preprocessed");
// 	int sum = 0;
// 	for(int idx = 0; idx < num; idx++) {
// 		sum += preprocess::Math::crop_test_2(depth_[idx], rgb_[idx], anno_[idx], radius);
// 	}

// 	std::cout << "There are " << sum << " invalid crops!" << std::endl;
// 	return 0;
// }
// RegisterBrewFunction(crop_test_2);


int pad_crop() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for (int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for (int idx = 0; idx < batchSize; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::pad_crop(depth_[idx], rgb_[idx], anno_[idx], radius);
			if (legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		readData("preprocessed", count * batchSize, last);

		for (int idx = 0; idx < left; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::pad_crop(depth_[idx], rgb_[idx], anno_[idx], radius);
			if (legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(pad_crop);


int findCandidates() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for (int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for (int idx = 0; idx < batchSize; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::findCandidates(depth_[idx], rgb_[idx], anno_[idx], outSize);
			if (legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		readData("preprocessed", count * batchSize, last);

		for (int idx = 0; idx < left; idx++) {
			bool qualified = preprocess::Math::isQualified(rgb_[idx]);
			bool legal = qualified && preprocess::Math::findCandidates(depth_[idx], rgb_[idx], anno_[idx], outSize);
			if (legal) {
				preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
				writeSingleData(idx);
			}
		}
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(findCandidates);


int filter() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for (int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for (int idx = 0; idx < batchSize; idx++) {
			//preprocess::Math::normalizeAll(depth_[idx]);
			preprocess::Math::high_filter(depth_[idx]);
			preprocess::Math::normalizeMinusOneToOne(depth_[idx]);
			//preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
			//preprocess::Math::normalizeAll(depth_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		readData("preprocessed", count * batchSize, last);

		for (int idx = 0; idx < left; idx++) {
			//preprocess::Math::normalizeAll(depth_[idx]);
			preprocess::Math::high_filter(depth_[idx]);
			preprocess::Math::normalizeMinusOneToOne(depth_[idx]);
			//preprocess::Math::normalizeHand(depth_[idx], rgb_[idx]);
			//preprocess::Math::normalizeAll(depth_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}
	return 0;
}
RegisterBrewFunction(filter)


int rescale() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for (int i = 0; i < count; i++) {
		readData("preprocessed", i * batchSize, last);

		for (int idx = 0; idx < batchSize; idx++) {
			preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], out_dimension);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		readData("preprocessed", count * batchSize, last);

		for (int idx = 0; idx < left; idx++) {
			preprocess::Math::scale(depth_[idx], rgb_[idx], anno_[idx], out_dimension);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}
	return 0;
}
RegisterBrewFunction(rescale)


int calculateMeanImage() {
	readPath("preprocessed");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	int height = outSize;
	int width = outSize;

	cv::Mat final(height, width, CV_32FC1, float(0.0));
	for (int i = 0; i < count; i++) {
		cv::Mat temp(height, width, CV_32FC1, float(0.0));
		readData("preprocessed", i * batchSize, last);

		for (int idx = 0; idx < batchSize; idx++) {
			cv::Mat_<float> depth = depth_[idx].getDepth();
			for (int j = 0; j < height; j++) {
				for (int k = 0; k < width; k++) {
					temp.at<float>(j, k) += depth(j, k);
				}
			}
		}

		temp = temp / batchSize;

		if (i == 0) {
			temp.copyTo(final);
		}
		else {
			for (int j = 0; j < height; j++) {
				for (int k = 0; k < width; k++) {
					final.at<float>(j, k) = (final.at<float>(j, k) + temp.at<float>(j, k)) / 2.0;
				}
			}
		}
		temp.release();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		cv::Mat temp(height, width, CV_32FC1, float(0.0));
		readData("preprocessed", count * batchSize, last);

		for (int idx = 0; idx < left; idx++) {
			cv::Mat_<float> depth = depth_[idx].getDepth();
			for (int j = 0; j < height; j++) {
				for (int k = 0; k < width; k++) {
					temp.at<float>(j, k) += depth(j, k);
				}
			}
		}

		temp = temp / left;

		for (int j = 0; j < height; j++) {
			for (int k = 0; k < width; k++) {
				final.at<float>(j, k) = (final.at<float>(j, k) + temp.at<float>(j, k)) / 2.0;
			}
		}

		temp.release();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	// write the final to file
	preprocess::IO io_;
	boost::filesystem::path out_file("../mean.exr");
	io_.writeDepth(out_file, final);
}
RegisterBrewFunction(calculateMeanImage)


int rename() {
	readPath("raw");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for (int i = 0; i < count; i++) {
		readData("raw", i * batchSize, last);
		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		readData("raw", count * batchSize, last);
		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(rename)


int correct() {
	readPath("raw");
	int count = num / batchSize;
	left = num % batchSize;
	bool last = false;
	if (left > 0) {
		std::cout << count + 1 << " batches in total!" << std::endl;
	}
	else {
		std::cout << count << " batches in total!" << std::endl;
	}

	for (int i = 0; i < count; i++) {
		readWrongData("raw", i * batchSize, last);

		for (int idx = 0; idx < batchSize; idx++) {
			preprocess::Math::offset(rgb_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << i + 1 << " finished!" << std::endl;
	}

	last = true;
	if (left > 0) {
		readWrongData("raw", count * batchSize, last);

		for (int idx = 0; idx < left; idx++) {
			preprocess::Math::offset(rgb_[idx]);
		}

		writeData(last);
		freeData();
		std::cout << "batch " << count + 1 << " finished!" << std::endl;
	}

	return 0;
}
RegisterBrewFunction(correct)


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
		("pad_crop", po::value<int>(&radius), "pad and crop the images")
		("merge", "find the mean image of all masks")
		("calculateMeanImage", po::value<int>(&outSize), "calculat the mean file")
		("findCandidates", po::value<int>(&outSize), "find all candidates")
		("correct", "fix exr bug");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (argc > 1) {
		if (vm.count("scale")) {
			std::cout << "Scale!" << std::endl;
			return GetBrewFunction(std::string("scale"))();
		}
		else if (vm.count("crop")) {
			std::cout << "Crop with radius " << radius << "!" << std::endl;
			return GetBrewFunction(std::string("crop"))();
		}
		else if (vm.count("filter")) {
			std::cout << "high-pass filter!" << std::endl;
			return GetBrewFunction(std::string("filter"))();
		}
		else if (vm.count("rescale")) {
			std::cout << "resacle the images to " << out_dimension << " by " << out_dimension << "!" << std::endl;
			return GetBrewFunction(std::string("rescale"))();
		}
		else if (vm.count("rename")) {
			std::cout << "rename all images!" << std::endl;
			return GetBrewFunction(std::string("rename"))();
		}
		else if (vm.count("crop_test_1")) {
			std::cout << "crop test 1!" << std::endl;
			return GetBrewFunction(std::string("crop_test_1"))();
		}
		else if (vm.count("crop_test_2")) {
			std::cout << "crop test 2!" << std::endl;
			return GetBrewFunction(std::string("crop_test_2"))();
		}
		else if (vm.count("pad_crop")) {
			std::cout << "pad and crop!" << std::endl;
			return GetBrewFunction(std::string("pad_crop"))();
		}
		else if (vm.count("merge")) {
			std::cout << "merge!" << std::endl;
			return GetBrewFunction(std::string("merge"))();
		}
		else if (vm.count("calculateMeanImage")) {
			std::cout << "calculate the mean file!" << std::endl;
			return GetBrewFunction(std::string("calculateMeanImage"))();
		}
		else if (vm.count("findCandidates")) {
			std::cout << "find all candidates!" << std::endl;
			return GetBrewFunction(std::string("findCandidates"))();
		}
		else if (vm.count("correct")) {
			std::cout << "fix exr bug!" << std::endl;
			return GetBrewFunction(std::string("correct"))();
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