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

// ===== Inference functions =====

void upload_TreeInfo_Inf(
	int									numTrees_,
	int									numLabels_,
	int									maxDepth_,
	int									labelIndex_,
	float								minProb_,
	std::vector<std::vector<Node_CU>>&	forest_
	);

void control_Inf(
	std::vector<rdf::DepthImage>&		depth_vector,
	std::vector<rdf::RGBImage>&			rgb_vecotr,
	std::vector<std::vector<Node_CU>>&	forest_CU,
	bool								forestInSharedMem
	);

#endif