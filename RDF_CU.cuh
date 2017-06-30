#ifndef RDF_CU_CUH
#define RDF_CU_CUH

#include <npp.h>
#include <vector>

#include <rdf/feature.hpp>
#include <rdf/aggregator.hpp>
#include <rdf/node.hpp>

#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "../../include/rdf/depthImage.hpp"
#include "../../include/rdf/rgbImage.hpp"

#define NODE_NUM_LABELS 3

#define SAMPLE_PER_IMAGE 2000

typedef struct {
	// RDF Default Parameters
	int			numTrees;
	int			numImages;
	int			numPerTree;
	int			maxDepth;
	int			numSamples;
	int			numLabels;
	int			maxSpan;
	int			spaceSize;
	int			numFeatures;
	int			numTresholds;

	// Utility Parameters
	int			sample_per_tree;
} RDF_CU_Param;

// CUDA Node Class
typedef struct {
	int			isSplit;
	float4		feature;
	float		threshold;
	size_t		aggregator[NODE_NUM_LABELS];
	int			leftChild;
	int			rightChild;
} Node_CU;


// ===== CUDA-RDF training Class =====
class RDF_CU
{
public:
	RDF_CU() {};
	~RDF_CU() {};

	// ===== Function Section =====

	// Reset GPU and CPU memory to calculate next tree
	void cu_reset();

	// Free up all host and device memory after completion
	void cu_free();

	// Set all utility parameters
	void setParam(
		RDF_CU_Param& params_
		);

	// Declare all device and host memory
	void cu_initialize(
		);

	// Upload all cpu generated features upto device
	void cu_featureTransfer(
		float4 *features_
		);

	// Control function used to call cu_trainLevel
	void cu_train(
		std::vector<rdf::Node>&		nodes
		);

	// Compute responses in batch mode
	void compute_responses(
		std::vector<rdf::Sample>&	samples_per_tree
		);

	// Upload depth image upto device
	void cu_depth_const_upload(
		int		depth_image_num,
		int		width_,
		int		height_
		);

	// Train each level of the current tree
	void cu_trainLevel(
		std::vector<int>&			idx,
		std::vector<int>&			idx_S,
		std::vector<int>&			idx_E,
		std::vector<rdf::Node>&		nodes,
		int							op_nodes,
		int							recurseDepth
		);

	// Compute entropy
	float entropy_compute(
		unsigned int*				stat,
		int							boundary
		);

	// Decide if should terminate the node
	bool shouldTerminate(
		float						maxGain,
		int							recurseDepth
		);

#ifdef ENABLE_GPU_OBSOLETE

	void cu_curLevel(
		std::vector<int>&			idx,
		std::vector<int>&			idx_S,
		std::vector<int>&			idx_E,
		int							op_nodes,
		std::vector<rdf::Node>&		nodes,
		int							recurseDepth
		);

	void cu_init(
		int2*	sample_coords,
		int*	sample_labels,
		int*	sample_depID,
		float4* features
		);

	void cu_dataTransfer(
		int2*	sample_coords,
		int*	sample_labels,
		int*	sample_depID,
		float4* features
		);

	void cu_depth_init(
		int		depth_image_num,
		int		width_,
		int		height_,
		float*	depth_array
		);

	void cu_depthTransfer(
		float* depth_array
		);

	void host_free();

#endif

	// ===== Data Section =====
public:
	// GPU Streams
	cudaStream_t				execStream;
	cudaStream_t				copyStream;

	// GPU Arrays
	int*						cu_sapID;
	int*						cu_labels;
	int*						cu_thresh_num;			// Number of thresholds per feature: 1 x featureNum
	float*						cu_response_array;
	float*						cu_thresh;				// Generated thresholds: 1 x featureNum x (thresholdNum_default + 1)
	float*						cu_gain;				// Gain for each threshold: 1 x featureNum x (thresholdNum_default + 1)
	unsigned int*				cu_partitionStatistics;
	unsigned int*				cu_leftStatistics;
	unsigned int*				cu_rightStatistics;
	float4*						cu_features;

	// Batch GPU arrays (two-stream architecture)
	int*						cu_depID1;
	int*						cu_depID2;
	int*						cu_sequence1;
	int*						cu_sequence2;
	int2*						cu_coords1;
	int2*						cu_coords2;
	float*						cu_depth_array1;
	float*						cu_depth_array2;

	// Host Arrays
	thrust::host_vector<int>	host_labels;
	thrust::host_vector<int>	host_sapID;
	thrust::host_vector<float>	host_response_array;
	unsigned int*				parentStatistics;

	// Host Variables
	int							nodes_size;
	int							depth_num;
	int							width;
	int							height;
	RDF_CU_Param				params;
};

// ===== CUDA-RDF inference Class =====
class CUDA_INFER {

// Functions
public:

	// Constructor
	CUDA_INFER(
		const int							parallel_proc_,
		const int							depth_width_,
		const int							depth_height_,
		const bool							createCUContext_,
		std::vector<std::vector<Node_CU>>&	forest_,
		std::string							out_directory_ = ""
		);

	// Destructor, free both device and host 
	~CUDA_INFER();

	// Upload depth information into constant memory
	void upload_DepthInfo_Inf(
		int width_,
		int height_
		);

	// Upload forest informatin to constant device memory
	// This function has to be called right after the constructor
	void cu_upload_TreeInfo(
		int									numLabels_,
		int									maxDepth_,
		float								minProb_,
		std::vector<std::vector<Node_CU>>&	forest_
		);

	// Determine if to put forest in global memory or shared memory
	int cu_ifInShared(std::vector<std::vector<Node_CU>>& forest_);

	// Do inference for the frame using the forest
	cv::Mat_<cv::Vec3i> cu_inferFrame(const cv::Mat_<float>& depth_img);

	// Write out results
	void writeResult(const cv::Mat_<cv::Vec3i>& result);

	void writeResult(const std::vector<cv::Mat_<cv::Vec3i>>& result);

	// Output last three frames
	std::vector<cv::Mat_<cv::Vec3i>> flushOutLastThree();

	// Free CPU memory
	void cu_hostMemFree();

	// Free GPU memory
	void cu_deviceMemFree();

// Data
public:

	// Constant information
	const int				parallel_proc;		// Maximum allowed images processed in parallel
	const int				depth_width;		// Width of the depth image, should be consistant across images
	const int				depth_height;		// Height of the depth image, should be consistant across images

	// Constant information for sizes of device arrays
	const int				depthArray_size;
	const int				rgbArray_size;

	// Declare input GPU memory
	float*					device_depthArray1;
	float*					device_depthArray2;
	int3*					device_rgbArray1;
	int3*					device_rgbArray2;

	// Declare output host memory to hold RGB information
	int3*					host_rgbArray1;
	int3*					host_rgbArray2;

	// Declare space to hold forest in both host and device
	Node_CU*				host_forest;
	Node_CU*				device_forest;

	// Declare temporay address only for address swap purposes
	float*					device_depthArray_t;
	int3*					device_rgbArray_t;

	// Declare cuda streams to hide memory copy latency
	cudaStream_t			execStream;
	cudaStream_t			copyStream;
	
	bool					firstFrame;			// Used for two stream architecture
	bool					forestInSharedMem;	// Boolean indicating whether the forest in global or shared memory
	bool					writeOut;			// Used to indicate whether to output results
	const bool				createCUContext;	// Set ture to create context from inference
	
	int						blk_Inference;		// Kernel launch setting parameters
	int						frame_cntr;			// Internally record the number of frames processed
	int						forest_size;		// The size of the forest in bytes

	std::string				out_directory;		// Result output directory
	boost::filesystem::path	out;				// For output purpose
};

// ===== Device functions =====

// Query device parameters
int queryDeviceParams(
	char* query
	);

// Check CUDA environment and choose device to operate on
int createCuContext(
	bool displayDeviceInfo
	);

// Destroy CUDA context after completion
void destroyCuContext();

#endif