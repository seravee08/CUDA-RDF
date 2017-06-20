#ifndef RDF_CU_CUH
#define RDF_CU_CUH

#include <npp.h>
#include <vector>
#include <random>

#include <rdf/feature.hpp>
#include <rdf/aggregator.hpp>
#include <rdf/node.hpp>

#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "../../include/rdf/depthImage.hpp"
#include "../../include/rdf/rgbImage.hpp"

#define NODE_NUM_LABELS 4

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

	// Function Section
	void setParam(
		RDF_CU_Param& params_
		);

	//void cu_init(
	//	int2*	sample_coords,
	//	int*	sample_labels,
	//	int*	sample_depID,
	//	float4* features
	//	);

	//void cu_dataTransfer(
	//	int2*	sample_coords,
	//	int*	sample_labels,
	//	int*	sample_depID,
	//	float4* features
	//	);

	//void cu_depth_init(
	//	int		depth_image_num,
	//	int		width_,
	//	int		height_,
	//	float*	depth_array
	//	);

	void cu_initialize(
		);

	void cu_depth_const_upload(
		int		depth_image_num,
		int		width_,
		int		height_
		);

	//void cu_depthTransfer(
	//	float* depth_array
	//	);

	void cu_featureTransfer(
		float4 *features_
		);

	void cu_curLevel(
		std::vector<int>&		idx,
		std::vector<int>&		idx_S,
		std::vector<int>&		idx_E,
		int						op_nodes,
		std::vector<rdf::Node>& nodes,
		int						recurseDepth
		);

	void cu_trainLevel(
		std::vector<int>&		idx,
		std::vector<int>&		idx_S,
		std::vector<int>&		idx_E,
		std::vector<rdf::Node>& nodes,
		int						op_nodes,
		int						recurseDepth
		);

	float entropy_compute(
		size_t* stat,
		int		boundary
		);

	bool shouldTerminate(
		float	maxGain,
		int		recurseDepth
		);

	void cu_train(
		std::vector<rdf::Node>& nodes
		);

	void compute_responses(
		std::vector<rdf::Sample>& samples_per_tree
		);

	void cu_reset();

	void cu_free();

	void host_free();

	// Data Section
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
	size_t*						cu_partitionStatistics;
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
	size_t*						parentStatistics;

	// Host Variables
	int							nodes_size;
	int							depth_num;
	int							width;
	int							height;
	RDF_CU_Param				params;
};

// Device functions

int queryDeviceParams(
	char* query
	);

int createCuContext(
	bool displayDeviceInfo
	);

void destroyCuContext();

// Inference functions

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