#ifndef RDF_CU_CUH
#define RDF_CU_CUH

#include <npp.h>
#include <vector>

#include <rdf/feature.hpp>
#include <rdf/aggregator.hpp>
#include <rdf/node.hpp>
#include <rdf/logSpace.hpp>
#include <rdf/forest.hpp>

#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "../../include/rdf/depthImage.hpp"
#include "../../include/rdf/rgbImage.hpp"

// ===== Train =====

#define DPInfoCount 4

#define UTInfoCount 1

#define PROCESS_LIMIT 3

#define NODE_NUM_LABELS 3

#define SAMPLE_PER_IMAGE 2000

// ===== Test =====

#define DepthInfoCount_Inf 3

#define TREE_CONTAINER_SIZE	3

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
	int			numThresholds;

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
class CUDA_TRAIN
{
public:
	CUDA_TRAIN(
		const int numTrees_,
		const int numImages_,
		const int numPerTree_,
		const int maxDepth_,
		const int numSamples_,
		const int numLabels_,
		const int maxSpan_,
		const int spaceSize_,
		const int numFeatures_,
		const int numThreshold,
		boost::shared_ptr<std::vector<std::vector<rdf::Sample> > > samples_,
		const bool createCUContext_
		);

	~CUDA_TRAIN();

	// ===== Function Section =====

	// Reset GPU and CPU memory to calculate next tree
	void cu_reset();

	// Free up device memory
	void cu_deviceMemFree();

	// Free up host memory
	void cu_hostMemFree();

	// Declare all device and host memory
	void cu_initialize();

	// Control function used to call cu_trainLevel
	void cu_train(std::vector<rdf::Node>& nodes);

	// Compute responses in batch mode
	void compute_responses(std::vector<rdf::Sample>& samples_per_tree);

	// Train forest
	rdf::Forest* cu_startTrain(boost::shared_ptr<std::vector<std::vector<rdf::Sample>>> samples_);

	// Retrieve depth image width
	int getWidth();

	// Retrieve depth image height
	int getHeight();

	// Retrieve number of trees
	int getNumTrees();

	// Retrieve number of labels
	int getNumLabels();

	// Retrieve maximum depth
	int getMaxDepth();

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

	rdf::LogSpace				logSpace;
	rdf::Forest*				forest;
	RDF_CU_Param				params;

	// Determine whether to create cuda context with the class
	const bool					createCUContext;
};

// ===== CUDA-RDF inference Class =====
class CUDA_INFER {

// Functions
public:

	// Constructor
	CUDA_INFER(
		const int							parallel_proc_,			// Number of images processed in parallel
		const int							depth_width_,			// The width of the depth image
		const int							depth_height_,			// The height of the depth image
		const int							numTrees_,				// Number of trees in the forest
		const int							numLabels_,				// Number of labels
		const int							maxDepth_,				// Max depth in the tree
		const float							minProb_,				// Probability threshold to determine the color of a pixel
		const bool							createCUContext_,		// If create cuda context inside the class
		const bool							realTime_,				// If use real time mode, low performance warning
		const std::string					in_directory_,			// Directory to the input folder
		const std::string					out_directory_ = ""		// Directory to the output folder, leave blank if no output required
		);

	CUDA_INFER::CUDA_INFER(
		const int							parallel_proc_,			// Number of images processed in parallel
		CUDA_TRAIN&							cuda_train,				// Initialize from CUDA_TRAIN instance
		const float							minProb_,				// Probability threshold to determine the color of a pixel
		const bool							createCUContext_,		// If create cuda context inside the class
		const bool							realTime_,				// If use real time mode, low performance warning
		const std::string					in_directory_,			// Directory to the input folder
		const std::string					out_directory_			// Directory to the output folder, leave blank if no output required
		);

	// Destructor, free both device and host 
	~CUDA_INFER();

	// Read in the forest
	void readForest();

	// Upload forest informatin to constant device memory
	void cu_upload_TreeInfo();

	// Upload depth information into constant memory
	void upload_DepthInfo_Inf(
		int width_,
		int height_
		);

	// Do inference for the frame using the forest
	cv::Mat_<cv::Vec3i> cu_inferFrame(const cv::Mat_<float>& depth_img);

	// Do inference for the frame without buffers, low performance warning
	cv::Mat_<cv::Vec3i> cu_inferFrame_hard(const cv::Mat_<float>& depth_img);

	// Write out results
	void writeResult(const cv::Mat_<cv::Vec3i>& result);

	void writeResult(const std::vector<cv::Mat_<cv::Vec3i>>& result);

	// Output last three frames
	std::vector<cv::Mat_<cv::Vec3i>> flushOutLastThree();

	// Determine if to put forest in global memory or shared memory
	int cu_ifInShared();

	// Free CPU memory
	void cu_hostMemFree();

	// Free GPU memory
	void cu_deviceMemFree();

// Data
public:

	// Constant information
	const int							parallel_proc;		// Maximum allowed images processed in parallel
	const int							depth_width;		// Width of the depth image, should be consistant across images
	const int							depth_height;		// Height of the depth image, should be consistant across images
	const int							numLabels;
	const int							numTrees;
	const int							maxDepth;
	const float							minProb;

	// Constant information for sizes of device arrays
	const int							depthArray_size;
	const int							rgbArray_size;

	// Declare input GPU memory
	float*								device_depthArray1;
	float*								device_depthArray2;
	int3*								device_rgbArray1;
	int3*								device_rgbArray2;

	// Declare output host memory to hold RGB information
	int3*								host_rgbArray1;
	int3*								host_rgbArray2;

	// Declare space to hold forest in both host and device
	Node_CU*							host_forest;
	Node_CU*							device_forest;

	// Declare temporay address only for address swap purposes
	float*								device_depthArray_t;
	int3*								device_rgbArray_t;

	// Declare cuda streams to hide memory copy latency
	cudaStream_t						execStream;
	cudaStream_t						copyStream;
	
	bool								firstFrame;			// Used for two stream architecture
	bool								forestInSharedMem;	// Boolean indicating whether the forest in global or shared memory
	bool								writeOut;			// Used to indicate whether to output results
	const bool							realTime;			// Process the frame in real time, low performance warning!
	const bool							createCUContext;	// Set ture to create context from inference
	
	int									blk_Inference;		// Kernel launch setting parameters
	int									frame_cntr;			// Internally record the number of frames processed
	int									forest_size;		// The size of the forest in bytes

	const std::string					in_directory;		// Input directory
	const std::string					out_directory;		// Result output directory

	boost::filesystem::path				in_path;			// For input purpose
	boost::filesystem::path				out_path;				// For output purpose

	std::vector<std::vector<Node_CU>>	forest_CU;			// Stores the raw forest
};

// ===== Device functions =====

// Query device parameters
int queryDeviceParams(char* query);

// Check CUDA environment and choose device to operate on
int createCuContext(bool displayDeviceInfo);

// Destroy CUDA context after completion
void destroyCuContext();

// ========================== Declaration for kernels ======================== //

// ===== Train =====

__global__ void sapID_ini(int* sap_ID);

__global__ void kernel_compute_response_batch(
	int			valid_images,
	int*		cu_depthID,
	int*		sequence,
	int2*		cu_coords,
	float*		cu_depth_array,
	float*		cu_response_array,
	float4*		cu_features
	);

__global__ void kernel_generate_thresholds(
	float*	response,
	int*	thresh_num,
	float*	thresh,
	int*	sample_ID,
	int		start_index,
	int		sample_num
	);

__global__ void kernel_compute_partitionStatistics(
	float*			response,
	float*			thresh,
	int*			labels,
	int*			thresh_num,
	int*			sample_ID,
	int				start_index,
	int				sample_num,
	unsigned int*	partition_statistics
	);

__global__ void kernel_compute_gain(
	float*				gain,
	int*				thresh_num,
	float				parent_entropy,
	unsigned int*		left_statistics,
	unsigned int*		right_statistics,
	unsigned int*		partition_statistics
	);

// ===== Test =====

__global__ void kernel_inference_ForestInShared(
	float*		depth,
	int3*		rgb,
	Node_CU*	forest,
	int			parallel_depth_img
	);

__global__ void kernel_inference_ForestInGlobal(
	float*		depth,
	int3*		rgb,
	Node_CU*	forest,
	int			parallel_depth_img
	);

// ================== Functions for constant memory upload ==================

// ===== Train =====

void const_DP_Info_upload			(const int* host_DP_Info);

void const_UT_Info_upload			(const int* host_UT_Info);

void const_CU_Params_upload			(const RDF_CU_Param* params);

// ===== Test =====

void const_DepthInfo_Inf_upload		(const int* host_DepthInfo_Inf_);

void const_numTrees_Inf_upload		(const int* numTrees_);

void const_numLabels_Inf_upload		(const int* numLabels_);

void const_maxDepth_Inf_upload		(const int* maxDepth_);

void const_labelIndex_Inf_upload	(const int* labelIndex_);

void const_minProb_Inf_upload		(const float* minProb_);

void const_totalNode_Inf_upload		(const int* host_total_nodeNum_);

void const_treePartition_Inf_upload	(const int* host_partitionInfo_);

#endif