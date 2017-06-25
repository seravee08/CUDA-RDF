#include <rdf/io.hpp>
#include <rdf/depthImage.hpp>
#include <rdf/rgbImage.hpp>
#include <rdf/rdf.hpp>
#include <rdf/forest.hpp>
#include <rdf/target.hpp>
#include <rdf/sample.hpp>
#include <util/fileutil.h>
#include <proto/rdf.pb.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <map>

#include <opencv2/opencv.hpp>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
//#include <glog/logging.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <io.h>
#include <stdio.h>
#include "RDF_CU.cuh"

using namespace std;
using google::protobuf::Message;
using google::protobuf::io::FileInputStream;

std::vector<std::string> test_args;
std::string filename;
int labelIndex;

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


void readFiles(rdf::IO& io_, 
	const std::vector<boost::filesystem::path>& all_depth_paths, 
	std::vector<rdf::DepthImage>& depth_, 
	std::vector<rdf::RGBImage>& rgb_, 
	const int& idx){

	boost::filesystem::path depth_path = all_depth_paths[idx];
	std::string id, ts;
	io_.getIdTs(depth_path, id, ts);

	cv::Mat_<float> depth = io_.readDepth(depth_path);
	depth_[idx].setDepth(depth);
	depth_[idx].setFileNames(id, ts);

	boost::filesystem::path rgb_path = io_.rgbPath(depth_path);
	if(!boost::filesystem::exists(rgb_path))
			return;
	cv::Mat_<cv::Vec3i> rgb = io_.readRGB(rgb_path);
	rgb_[idx].setRGB(rgb);
	rgb_[idx].setFileNames(id, ts);
}

void readFiles(rdf::IO& io_, 
	const std::vector<boost::filesystem::path>& all_depth_paths, 
	std::vector<rdf::DepthImage>& depth_, 
	std::vector<rdf::RGBImage>& rgb_, 
	const int& idx,
	const int& idx_global){

	boost::filesystem::path depth_path = all_depth_paths[idx_global];
	std::string id, ts;
	io_.getIdTs(depth_path, id, ts);

	cv::Mat_<float> depth = io_.readDepth(depth_path);
	depth_[idx].setDepth(depth);
	depth_[idx].setFileNames(id, ts);

	boost::filesystem::path rgb_path = io_.rgbPath(depth_path);
	if(!boost::filesystem::exists(rgb_path))
			return;
	cv::Mat_<cv::Vec3i> rgb = io_.readRGB(rgb_path);
	rgb_[idx].setRGB(rgb);
	rgb_[idx].setFileNames(id, ts);
}


void train_initialize(std::vector<rdf::Sample>& samples, 
	rdf::IO& io_, 
	const std::vector<boost::filesystem::path>& all_depth_paths, 
	std::vector<rdf::DepthImage>& depth_, 
	std::vector<rdf::RGBImage>& rgb_, 
	const int& idx,
	const int& idx_global,
	std::vector<int>& labels,
	std::vector<rdf::RDFParameter::Col>& colors,
	const int& numSamples) {

	//read files
	readFiles(io_, all_depth_paths, depth_, rgb_, idx, idx_global);

	//record each color in mask
	cv::Mat_<int> mask(labels.size(), 3, 0);
	for(int i = 0; i < labels.size(); i++) {
		if(colors[i] == rdf::RDFParameter::RED) {
			mask(i, 2) = 255;
		}
		else if(colors[i] == rdf::RDFParameter::GREEN) {
			mask(i, 1) = 255;
		}
		else if(colors[i] == rdf::RDFParameter::BLUE) {
			mask(i, 0) = 255;
		}
		else if(colors[i] == rdf::RDFParameter::BLACK) {
			//
		}
		else if(colors[i] == rdf::RDFParameter::WHITE) {
			mask(i, 0) = 255;
			mask(i, 1) = 255;
			mask(i, 2) = 255;
		}
		else if(colors[i] == rdf::RDFParameter::YELLOW) {
			mask(i, 1) = 255;
			mask(i, 2) = 255;
		}
		else if(colors[i] == rdf::RDFParameter::PURPLE) {
			mask(i, 0) = 255;
			mask(i, 2) = 255;
		}
		else if(colors[i] == rdf::RDFParameter::AZURE) {
			mask(i, 0) = 255;
			mask(i, 1) = 255;
		}
		else {
			std::cout << "Invalid color!" << std::endl;
		}
	}

	//classcify
	std::vector<std::vector<cv::Point2i> > sampleClass;
	sampleClass.resize(labels.size() + 1);

	std::vector<int> actual_num(labels.size() + 1, 0);

	for(int i = 0; i < depth_[idx].getDepth().rows; i++) {
		for(int j = 0; j < depth_[idx].getDepth().cols; j++) {
			cv::Point2i point(j,i);
			for(int k = 0; k < colors.size(); k++) {
				if(mask(k, 0) == rgb_[idx].getRGB()(i, j)[0] && mask(k, 1) == rgb_[idx].getRGB()(i, j)[1] && mask(k, 2) == rgb_[idx].getRGB()(i, j)[2]) {
					sampleClass[k].push_back(point);
					actual_num[k]++;
					break;
				}
				if(k == colors.size() - 1) {
					sampleClass[colors.size()].push_back(point);
					actual_num[k + 1]++;
				}
			}
		}
	}

	// handle the situation that if there are less pixels for this class than defined in proto file
	for(int i = 0; i < labels.size(); i++) {
		actual_num[i] = std::min(actual_num[i], labels[i]);
	}

	int left = numSamples;
	for(int i = 0; i < labels.size(); i++) {
		left -= actual_num[i];
	}
	if(left < actual_num[labels.size()]) {
		actual_num[labels.size()] = left;
	}

	std::random_device rand;
	std::mt19937 gen(rand());
	std::vector<std::uniform_int_distribution<> > distribution;
	distribution.resize(labels.size() + 1);


	for(int i = 0; i < labels.size() + 1; i++) {
		distribution[i] = std::uniform_int_distribution<>(0, actual_num[i] - 1);
	}

	for(int i = 0; i < sampleClass.size(); i++) {
		int count = 0;
		int sumup = 0;
		for(int j = 0; j < i; j++) {
			sumup += actual_num[j];
		}
		while(count < actual_num[i]) {
			int id = distribution[i](gen);
			rdf::Sample sample;
			sample.setCoor(sampleClass[i][id]);
			sample.setDepth(depth_[idx].getDepth());
			sample.setRGB(rgb_[idx].getRGB());
			sample.setIdx(numSamples * idx + sumup + count);
			sample.setLabel(i);
			sample.setDepthID(idx);
			samples.push_back(sample);
			count++;
		}
	}

}

bool readProtoFromText(const std::string& filename, Message* proto) {
	//std::ifstream fs;
	//fs.open(filename.c_str());
	int fd = _open(filename.c_str(), _O_RDONLY);
	if(fd == -1) {
		std::cout << "File not found: " << filename << std::endl;
	}
	FileInputStream* input = new FileInputStream(fd);
  	bool success = google::protobuf::TextFormat::Parse(input, proto);
  	delete input;
  	_close(fd);
	return success;
	//return true;
}


int train() {
	std::cout << "Start!" << std::endl;
	auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    //std::cout << std::put_time(std::localtime(&now_c), "%c") << std::endl;

    ////////////////////////////////////////////////////

    rdf::RDFParameter rdfParam;
	filename = "S:/ProjectsVS/RDF_GPU/RDF-master/rdf_param.prototxt";

    if(!readProtoFromText(filename, &rdfParam)) {
		std::cout << "Cannot read proto text!" << std::endl;
	}

	std::string p = rdfParam.input_path();
	boost::filesystem::path in("S:/ProjectsVS/RDF_GPU/RDF-master/data");

	std::vector<boost::filesystem::path> all_depth_paths;
	addPaths(in, ".*.exr", all_depth_paths);

	const int numTrees = rdfParam.num_trees();
	const int numImages = std::min((int)all_depth_paths.size(), (int)rdfParam.num_images());
	const int numPerTree = (int)(numImages / numTrees); 
	const int maxDepth = rdfParam.num_depth();
	const int numSamples = rdfParam.num_samples();
	const int numLabels = rdfParam.num_pixels_size();
	const int maxSpan = rdfParam.max_span();
	const int spaceSize = rdfParam.space_size();
	const int numFeatures = rdfParam.num_features();
	const int numThresholds = rdfParam.num_thresholds();

	std::vector<int> labels;
	labels.resize(numLabels);
	std::vector<rdf::RDFParameter::Col> colors;
	colors.resize(numLabels);
	for(int i = 0; i < numLabels; i++) {
		labels[i] = rdfParam.num_pixels(i);
		colors[i] = rdfParam.color(i);
	}
	
	std::vector<std::vector<rdf::DepthImage> > depth_;
	std::vector<std::vector<rdf::RGBImage> > rgb_;

	depth_.resize(numTrees);
	rgb_.resize(numTrees);

	for(int i = 0; i < numTrees; i++) {
		depth_[i].resize(numPerTree);
		rgb_[i].resize(numPerTree);
	}

	boost::shared_ptr<std::vector<std::vector<rdf::Sample> > > samples = boost::make_shared<std::vector<std::vector<rdf::Sample> > >();
	(*samples).resize(numTrees);

	rdf::IO io_;

	for(int i = 0; i < numTrees; i++) {
		for(int j = 0; j < numPerTree; j++) {
		    train_initialize((*samples)[i], io_, all_depth_paths, depth_[i], rgb_[i], j, i * numPerTree + j, labels, colors, numSamples);
	    }
	    std::random_shuffle((*samples)[i].begin(), (*samples)[i].end());
	}

	std::cout << "Start training!" << std::endl;
	now = std::chrono::system_clock::now();
    now_c = std::chrono::system_clock::to_time_t(now);
    //std::cout << std::put_time(std::localtime(&now_c), "%c") << std::endl;


	rdf::RDFPtr rdf = boost::make_shared<rdf::RDF>(samples);
	rdf->initialize_cu(maxSpan, spaceSize, numLabels + 1, numFeatures, numThresholds, numTrees, maxDepth, numSamples, numImages, numPerTree);
	rdf::ForestPtr forest = rdf->trainForest(maxDepth);

	std::cout << "Training complete!" <<std::endl;
	now = std::chrono::system_clock::now();
    now_c = std::chrono::system_clock::to_time_t(now);
    //std::cout << std::put_time(std::localtime(&now_c), "%c") << std::endl;

    std::string outName = rdfParam.out_name();
	boost::filesystem::path out(outName);
	forest->save(out);

	std::cout << "Finished!" <<std::endl;
	now = std::chrono::system_clock::now();
    now_c = std::chrono::system_clock::to_time_t(now);
    //std::cout << std::put_time(std::localtime(&now_c), "%c") << std::endl;


	return 0;
}
RegisterBrewFunction(train)

int test() {
	std::cout << "Start!" << std::endl;
	auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    //std::cout << std::put_time(std::localtime(&now_c), "%c") << std::endl;

    ////////////////////////////////////////////////////

    rdf::RDFParameter rdfParam;
	filename = "S:/ProjectsVS/RDF_GPU/RDF-master/rdf_param.prototxt";

    if(!readProtoFromText(filename, &rdfParam)) {
		std::cout << "Cannot read proto text!" << std::endl;
	}

	const int numTrees = rdfParam.num_trees();
	const int numLabels = rdfParam.num_pixels_size();
	const int maxDepth = rdfParam.num_depth();
	const float min_prob = rdfParam.min_prob();

	//std::string p = rdfParam.test_input_path();
	//boost::filesystem::path in(p);
	boost::filesystem::path in("S:/ProjectsVS/RDF_GPU/RDF-master/data");

	std::vector<boost::filesystem::path> all_depth_paths;
	addPaths(in, ".*.exr", all_depth_paths);
	rdf::IO io_;
	int num = all_depth_paths.size();

	std::vector<rdf::DepthImage> depth_;
	std::vector<rdf::RGBImage> rgb_;

	depth_.resize(num);
	rgb_.resize(num);

	for(int idx = 0; idx < num; idx++) {
		readFiles(io_, all_depth_paths, depth_, rgb_, idx);
	}

	// ===== Load trained forest =====
	std::vector<std::vector<Node_CU> > forest_CU(numTrees);
	std::string fp = rdfParam.out_name();
	boost::filesystem::path fin(fp);
	rdf::Forest forest_(numTrees, maxDepth);
	forest_.readForest(fin, numLabels + 1, forest_CU);

	// Automatically determine if to use shared memory based on device settings
	bool forestInSharedMem = false;
	int forest_size = 0;
	for (int i = 0; i < numTrees; i++) {
		forest_size += forest_CU[i].size() * sizeof(Node_CU);
	}
	if (forest_size > queryDeviceParams("sharedMemPerBlock") * 0.8) {
		cout << "Storing the forest into global memory ..." << endl;
	}
	else {
		forestInSharedMem = true;
		cout << "Storing the forest into shared memory ..." << endl;
	}

	// =========================== GPU Branch =============================== //
#ifdef USE_GPU_INFERENCE

	clock_t inference_start_GPU = clock();

	// Collect information for GPU
	upload_TreeInfo_Inf(numTrees, numLabels, maxDepth, labelIndex, min_prob, forest_CU);

	// Call CUDA functions to do inference
	control_Inf(depth_, rgb_, forest_CU, forestInSharedMem);

	// Calculate elapsed time
	clock_t inference_end_GPU = clock();
	float time_elapsed_GPU = (float)(inference_end_GPU - inference_start_GPU) / CLOCKS_PER_SEC;
	printf("GPU inference time: %f\n", time_elapsed_GPU);
	printf("Writing out results ...\n");

	// Output results
	std::string ou = rdfParam.test_output_path();
	boost::filesystem::path out(ou);
	if (!boost::filesystem::exists(out)){
		boost::filesystem::create_directory(out);
	}

	for (int idx = 0; idx < num; idx++) {
		const cv::Mat_<cv::Vec3i>& result_rgb = rgb_[idx].getRGB();
		std::string id, ts;
		rgb_[idx].getFileNames(id, ts);
		boost::format fmt_result_rgb("%s_%s_result_rgb.png");
		boost::filesystem::path out_rgb = out / (fmt_result_rgb % id % ts).str();
		io_.writeRGB(out_rgb, result_rgb);
	}

#else

	// =========================== CPU Branch =============================== //
	std::vector<cv::Mat_<int> > masks;

	masks.resize(num);

	std::cout << "Loaded all images, start testing!" << std::endl;

	clock_t inference_start_CPU = clock();

	for(int idx = 0; idx < num; idx++) {

		cv::Mat_<float> depth = depth_[idx].getDepth();
		cv::Mat_<int> mask = cv::Mat_<int>::zeros(depth.rows, depth.cols);
		for(int i = 0; i < depth.rows; i++) {
			for(int j = 0; j < depth.cols; j++) {
				rdf::Target result(numLabels + 1);
				rdf::Sample sample;
				sample.setCoor(j, i);
				sample.setDepth(depth);

				forest_.inference(result, sample, numLabels + 1);

				float p = result.Prob()[labelIndex];
				if(p > min_prob)
					mask(i, j) = 1;
			}
		}
		masks[idx] = mask;
		std::cout << "Image " << idx + 1 << " is done!" << std::endl;
	}

	// Calculate elapsed time
	clock_t inference_end_CPU = clock();
	float time_elapsed_CPU = (float)(inference_end_CPU - inference_start_CPU) / CLOCKS_PER_SEC;
	printf("CPU inference time: %f\n", time_elapsed_CPU);
	std::cout << "Finished! Wait for writting!" << std::endl;

	//output results
	std::string ou = rdfParam.test_output_path();
	boost::filesystem::path out(ou);
	if(!boost::filesystem::exists(out)){
		boost::filesystem::create_directory(out);
	}

	for(int idx = 0; idx < num; idx++) {
		cv::Mat_<cv::Vec3i> result_rgb = rgb_[idx].getRGB();
		for(int i = 0; i < result_rgb.rows; i++) {
			for(int j = 0; j < result_rgb.cols; j++) {
				result_rgb(i, j)[0] = 0;
				result_rgb(i, j)[1] = 0;
				result_rgb(i, j)[2] = 0;

				if(masks[idx](i, j) == 1) {
					result_rgb(i, j)[2 - labelIndex] = 255;
				}
			}
		}

		std::string id, ts;
		rgb_[idx].getFileNames(id, ts);
		boost::format fmt_result_rgb("%s_%s_result_rgb.png");
		boost::filesystem::path out_rgb = out / (fmt_result_rgb % id % ts).str();
		io_.writeRGB(out_rgb, result_rgb);
	}
#endif

	std::cout << "Done!" << std::endl;
	return 0;
}
RegisterBrewFunction(test)


int main(int argc, char** argv){
	//namespace po = boost::program_options;
	//po::options_description desc("options");
	//desc.add_options()
	//	("train", po::value<std::string>(&filename), "train the rdf.")
	//	("test", po::value<std::vector<std::string> >(&test_args)->multitoken(), "test the rdf.");

	//po::variables_map vm;
	//po::store(po::parse_command_line(argc, argv, desc), vm);
	//po::notify(vm);

	//if(argc > 1) {
	//	if(vm.count("train")) {
	//		std::cout << "Train!" << std::endl;
	//		return GetBrewFunction(std::string("train"))();
	//	}
	//	else if(vm.count("test")) {
	//		std::cout << "Test!" << std::endl;
	//		filename = test_args[0];
	//		labelIndex = std::stoi(test_args[1]);
	//		return GetBrewFunction(std::string("test"))();
	//	} 
	//	else {
	//		std::cout << "Illegal options!" << std::endl;
	//	}
	//}
	//else {
	//	std::cout << "Should have at least one option!" << std::endl;
	//}

	createCuContext(false);
	// train();
	test();
	destroyCuContext();

	system("pause");

	return 0;
}
