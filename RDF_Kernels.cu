#include "device_launch_parameters.h"
#include "RDF_CU.cuh"

#include <curand.h>
#include <curand_kernel.h>

// ================================================================================================================
// ============================================= TRAINING =========================================================
// ================================================================================================================

__constant__ RDF_CU_Param CU_Params[1];

__constant__ int const_DP_Info[DPInfoCount];

__constant__ int const_UT_Info[UTInfoCount];

__global__ void sapID_ini(int* sap_ID)
{
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= CU_Params[0].sample_per_tree)
	{
		return;
	}

	sap_ID[x_id] = x_id;
}

__device__ float entropy_gain(unsigned int* stat, unsigned int cntr)
{
	if (cntr == 0)
	{
		return 0.0;
	}

	float res = 0.0;
	for (int b = 0; b < CU_Params[0].numLabels; b++)
	{
		float p = stat[b] * 1.0 / cntr;
		res -= (p == 0.0) ? 0.0 : p * logf(p) / logf(2.0);
	}
	return res;
}

__global__ void kernel_compute_response_batch(
	int			valid_images,
	int*		cu_depthID,
	int*		sequence,
	int2*		cu_coords,
	float*		cu_depth_array,
	float*		cu_response_array,
	float4*		cu_features
	)
{
	// Copy valid_images into shared memory
	__shared__ int shared_sampleNum[1];
	if (threadIdx.x == 0) {
		shared_sampleNum[0] = valid_images * SAMPLE_PER_IMAGE;
	}
	__syncthreads();

	// Get the operation position of current thread
	const int x_id = threadIdx.x + blockDim.x * blockIdx.x;

	// Determine feature and sample index from the x_id
	const int sample_id = x_id % (PROCESS_LIMIT * SAMPLE_PER_IMAGE);
	const int feature_id = x_id / (PROCESS_LIMIT * SAMPLE_PER_IMAGE);

	// Structure of response array: 
	// Feature1: sap1, sap2, ... Feature2: sap1, sap2, ... Feature n
	if (feature_id >= CU_Params[0].numFeatures || sample_id >= shared_sampleNum[0]) {
		return;
	}

	float depth = cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] +		// Offset to the start of the depth image
		cu_coords[sample_id].y * const_DP_Info[1] +					// Offset to the start of the corresponding row
		cu_coords[sample_id].x];										// Offset to the exact depth info

	// Handle the case when depth is zero
	if (depth == 0.0) {
		cu_response_array[feature_id * CU_Params[0].sample_per_tree + sequence[sample_id]] = -10000.0;
		return;
	}

	// Calculate responses
	float x = cu_features[feature_id].x / depth + cu_coords[sample_id].x;
	float y = cu_features[feature_id].y / depth + cu_coords[sample_id].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	float depth2 = cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];
	x = cu_features[feature_id].z / depth + cu_coords[sample_id].x;
	y = cu_features[feature_id].w / depth + cu_coords[sample_id].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	// ##### The curand causes memory issues ##### //
	//curandState state;
	//curand_init(clock(), x_id, 0, &state);
	//if (round(curand_uniform(&state)) == 1)
	//{
	//	cu_response_array[feature_id * CU_Params[0].sample_per_tree + sequence[sample_id]] = depth2 - depth;
	//}
	//else
	//{
	//	float depth3 = cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];
	//	cu_response_array[feature_id * CU_Params[0].sample_per_tree + sequence[sample_id]] = depth2 - depth3;
	//}
	// ########################################### //

	cu_response_array[feature_id * CU_Params[0].sample_per_tree + sequence[sample_id]] = depth2 - depth;
}

__global__ void kernel_compute_response(
	int2*	cu_coords,
	int*	cu_depthID,
	float*	cu_depth_array,
	int*	sample_ID,
	float4* cu_features,
	float*	cu_response_array
	)
{
	// Structure of response array: 
	// Feature1: sap1, sap2, ... Feature2: sap1, sap2, ... Feature n
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= const_UT_Info[0])
	{
		return;
	}

	const int sample_id = x_id % CU_Params[0].sample_per_tree;
	const int feature_id = x_id / CU_Params[0].sample_per_tree;

	float depth = cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] +			// Offset to the start of the depth image
		cu_coords[sample_id].y * const_DP_Info[1] +						// Offset to the start of the corresponding row
		cu_coords[sample_id].x];											// Offset to the exact depth info

	// Handle the case when depth is zero
	if (depth == 0.0) {
		cu_response_array[x_id] = -10000.0;
		return;
	}

	// Calculate responses
	float x = cu_features[feature_id].x / depth + cu_coords[sample_id].x;
	float y = cu_features[feature_id].y / depth + cu_coords[sample_id].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	float depth2 = cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];
	x = cu_features[feature_id].z / depth + cu_coords[sample_id].x;
	y = cu_features[feature_id].w / depth + cu_coords[sample_id].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	// ##### The curand causes memory issues ##### //
	//curandState state;
	//curand_init(clock(), x_id, 0, &state);
	//if (round(curand_uniform(&state)) == 1)
	//{
	//	cu_response_array[x_id] = depth2 - depth;
	//}
	//else
	//{
	//	float depth3 = cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];
	//	cu_response_array[x_id] = depth2 - depth3;
	//}
	// ########################################### //

	cu_response_array[x_id] = depth2 - depth;
}

__global__ void kernel_compute_gain(
	float*				gain,
	int*				thresh_num,
	float				parent_entropy,
	unsigned int*		left_statistics,
	unsigned int*		right_statistics,
	unsigned int*		partition_statistics
	)
{
	// Get the thread index
	const int x_id = threadIdx.x + blockDim.x * blockIdx.x;

	// Abort thread if the thread is out of boundary
	if (x_id >= (CU_Params[0].numThresholds + 1) * CU_Params[0].numFeatures) {
		return;
	}

	// Calculate the feature and threshold index
	const int feature_id = x_id / (CU_Params[0].numThresholds + 1);								 // Index of the feature
	const int thresh_id = x_id % (CU_Params[0].numThresholds + 1);								 // Index of the threshold
	const int thresh_number = thresh_num[feature_id];												 // Retrieve the number of thresholds for this feature
	const int feature_offset = feature_id * (CU_Params[0].numThresholds + 1) * CU_Params[0].numLabels; // Calculate feature offset

	// Abort thread if the threshold exceeds current total number of thresholds
	if (thresh_id >= thresh_number) {
		return;
	}

	unsigned int  left_counter = 0;
	unsigned int  right_counter = 0;
	// Aggregate histograms into left and right statistics
	for (int p = 0; p < thresh_number + 1; p++) {
		if (p <= thresh_id) {
			for (int i = 0; i < CU_Params[0].numLabels; i++) {
				left_statistics[feature_offset + thresh_id * CU_Params[0].numLabels + i] +=
					partition_statistics[feature_offset + p * CU_Params[0].numLabels + i];
			}
		}
		else {
			for (int i = 0; i < CU_Params[0].numLabels; i++) {
				right_statistics[feature_offset + thresh_id * CU_Params[0].numLabels + i] +=
					partition_statistics[feature_offset + p * CU_Params[0].numLabels + i];
			}
		}
	}

	for (int i = 0; i < CU_Params[0].numLabels; i++) {
		left_counter += left_statistics[feature_offset + thresh_id * CU_Params[0].numLabels + i];
		right_counter += right_statistics[feature_offset + thresh_id * CU_Params[0].numLabels + i];
	}

	// Calculate gain for the current threshold
	if ((left_counter + right_counter) <= 1) {
		gain[feature_id * (CU_Params[0].numThresholds + 1) + thresh_id] = 0.0;
	}
	else {
		gain[feature_id * (CU_Params[0].numThresholds + 1) + thresh_id] =
			parent_entropy - (left_counter * entropy_gain(&left_statistics[feature_offset + thresh_id * CU_Params[0].numLabels], left_counter)
			+ right_counter * entropy_gain(&right_statistics[feature_offset + thresh_id * CU_Params[0].numLabels], right_counter)) / (left_counter + right_counter);
	}
}

__global__ void kernel_compute_partitionStatistics(
	float*			response,
	float*			thresh,
	int*			labels,
	int*			thresh_num,
	int*			sample_ID,
	int				start_index,
	int				sample_num,
	unsigned int*	partition_statistics
	)
{
	// Copy number of samples into shared memory
	__shared__ int shared_sample_num[1];
	if (threadIdx.x == 0) {
		shared_sample_num[0] = sample_num;
	}
	__syncthreads();

	// Decide if the thread is out of boundary
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= shared_sample_num[0] * CU_Params[0].numFeatures) {
		return;
	}

	// Decide the feature index and relative sample index
	const int feature_id = x_id / shared_sample_num[0];
	const int rel_sample_id = x_id % shared_sample_num[0];
	const int thresh_number = thresh_num[feature_id];

	// Put the sample label into right bin
	int b = 0;
	while (
		b < thresh_number &&
		response[feature_id * CU_Params[0].sample_per_tree + sample_ID[start_index + rel_sample_id]] // Retrieve the response
		>= thresh[feature_id * (CU_Params[0].numThresholds + 1) + b]								     // Retrieve the thresholds
		)
		b++;

	// Add one to the corresponding histogram
	atomicAdd(
		&partition_statistics[feature_id * (CU_Params[0].numThresholds + 1) * CU_Params[0].numLabels // Offset to the start of the feature
		+ b * CU_Params[0].numLabels																// Offset to the start of the threshold
		+ labels[sample_ID[start_index + rel_sample_id]]],											// Offset to the bin
		1
		);
}

__global__ void kernel_generate_thresholds(
	float*	response,
	int*	thresh_num,
	float*	thresh,
	int*	sample_ID,
	int		start_index,
	int		sample_num
	)
{
	// Copy number of samples into shared memory
	__shared__ int shared_sample_num[1];
	if (threadIdx.x == 0) {
		shared_sample_num[0] = sample_num;
	}
	__syncthreads();

	// Decide if the thread is out of boundary
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= CU_Params[0].numFeatures) {
		return;
	}

	// Generate thresholds for each of the features
	curandState state;
	curand_init(clock(), x_id, 0, &state);
	const int numThresholds = CU_Params[0].numThresholds;
	if (shared_sample_num[0] > numThresholds) {

		thresh_num[x_id] = numThresholds;
		for (int i = 0; i < numThresholds + 1; i++) {
			int randIdx = round(curand_uniform(&state) * (shared_sample_num[0] - 1));
			thresh[x_id * (numThresholds + 1) + i] = response[x_id * CU_Params[0].sample_per_tree + sample_ID[start_index + randIdx]];
		}
	}
	else {
		thresh_num[x_id] = shared_sample_num[0] - 1;
		for (int i = 0; i < shared_sample_num[0]; i++) {
			thresh[x_id * (numThresholds + 1) + i] = response[x_id * CU_Params[0].sample_per_tree + sample_ID[start_index + i]];
		}
	}

	// Sort the generated thresholds
	thrust::sort(thrust::seq, &thresh[x_id * (numThresholds + 1)], &thresh[x_id * (numThresholds + 1)] + thresh_num[x_id] + 1);

	// Decide the validity of the thresholds
	if (thresh[x_id * (numThresholds + 1)] == thresh[x_id * (numThresholds + 1) + thresh_num[x_id]]) {
		thresh_num[x_id] = 0;
		return;
	}

	for (int i = 0; i < thresh_num[x_id]; i++) {
		int difference = curand_uniform(&state) * (thresh[x_id * (numThresholds + 1) + i + 1] -
			thresh[x_id * (numThresholds + 1) + i]);
		thresh[x_id * (numThresholds + 1) + i] += difference;
	}
}

// ================================================================================================================
// =============================================== TEST ===========================================================
// ================================================================================================================

__constant__ int	const_Depth_Info_Inf[DepthInfoCount_Inf];

__constant__ int	const_numTrees_Inf[1];								// Number of trees

__constant__ int	const_numLabels_Inf[1];								// Number of labels

__constant__ int	const_maxDepth_Inf[1];								// Max depth

__constant__ int	const_labelIndex_Inf[1];							// Label index

__constant__ float	const_minProb_Inf[1];								// Min probability

__constant__ int	const_totalNode_Inf[1];								// Total number of nodes in the forest

__constant__ int	const_treePartition_Inf[TREE_CONTAINER_SIZE + 1];	// Store the ACCUMULATIVE offsets of each starting index

extern __shared__ Node_CU forest_shared[];

__global__ void kernel_inference_ForestInShared(
	float*		depth,
	int3*		rgb,
	Node_CU*	forest,
	int			parallel_depth_img
	)
{
	// Copy forest into the shared memory
	if (threadIdx.x < const_totalNode_Inf[0]) {
		forest_shared[threadIdx.x] = forest[threadIdx.x];
	}
	__syncthreads();

	// Decide if the thread is out of boundary
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= parallel_depth_img * const_Depth_Info_Inf[2]) {
		return;
	}

	// Decide the basic information of the thread
	const int depth_idx = x_id / const_Depth_Info_Inf[2];											// Get the index of depth image
	const int y_coord = (x_id - const_Depth_Info_Inf[2] * depth_idx) / const_Depth_Info_Inf[0];	// Get the row of the sample
	const int x_coord = (x_id - const_Depth_Info_Inf[2] * depth_idx) % const_Depth_Info_Inf[0];	// Get the column of the sample

	// Declare the probability array for the forest
	float prob[NODE_NUM_LABELS] = { 0 };

	// Initialize the pixel of the output
	rgb[x_id].x = 0;
	rgb[x_id].y = 0;
	rgb[x_id].z = 0;

	// ===== Calculate response =====
	const float depth_local = depth[x_id];

	// Handle the case when depth is zero
	if (depth_local == 0.0) {
		return;
	}

	// ===== Go through the forest =====
	for (int i = 0; i < const_numTrees_Inf[0]; i++) {
		// ===== Go through each tree =====
		unsigned int idx = const_treePartition_Inf[i];

		while (forest_shared[idx].isSplit == 1) {

			// Calculate responses
			float x = forest_shared[idx].feature.x / depth_local + x_coord;
			float y = forest_shared[idx].feature.y / depth_local + y_coord;

			x = (x < 0) ? 0 :
				(x >= const_Depth_Info_Inf[0]) ? const_Depth_Info_Inf[0] - 1 : x;

			y = (y < 0) ? 0 :
				(y >= const_Depth_Info_Inf[1]) ? const_Depth_Info_Inf[1] - 1 : y;

			float depth2 = depth[depth_idx * const_Depth_Info_Inf[2] + int(y) * const_Depth_Info_Inf[0] + int(x)];

			// Calculate the response of the second set of offsets, disabled
			//x				= forest_shared[idx].feature.z / depth_local + x_coord;
			//y				= forest_shared[idx].feature.w / depth_local + y_coord;

			//x = (x < 0) ? 0 :
			//	(x >= const_Depth_Info_Inf[0]) ? const_Depth_Info_Inf[0] - 1 : x;
			//
			//y = (y < 0) ? 0 :
			//	(y >= const_Depth_Info_Inf[1]) ? const_Depth_Info_Inf[1] - 1 : y;

			// ##### The curand causes memory issues ##### //
			// float response;
			//curandState_t state;
			//curand_init(clock(), x_id, 0, &state);
			//if (round(curand_uniform(&state)) == 1) {
			//	response = depth2 - depth_local;
			//}
			//else {
			//response = depth2 - depth[depth_idx * const_Depth_Info_Inf[2] + int(y) * const_Depth_Info_Inf[0] + int(x)];
			//}
			// ########################################### //

			float response = depth2 - depth_local;
			// ============================

			// Decide which branch to goto
			if (response < forest_shared[idx].threshold) {
				idx = const_treePartition_Inf[i] + forest_shared[idx].leftChild;	// Goto left branch
			}
			else {
				idx = const_treePartition_Inf[i] + forest_shared[idx].rightChild;	// Goto right branch
			}
		}

		// Decide if the tree is valid
		if (forest_shared[idx].isSplit != 0) {
			printf("Error: non leaf node reached ...\n");
			return;
		}

		// Retrieve aggregator and calculate probabilities
		int sampleCount = 0;
		for (int j = 0; j < NODE_NUM_LABELS; j++) {
			sampleCount += forest_shared[idx].aggregator[j];
		}

		for (int j = 0; j < NODE_NUM_LABELS; j++) {
			prob[j] += forest_shared[idx].aggregator[j] * 1.0 / (sampleCount * const_numTrees_Inf[0]);
		}
	}

	// Decide the label of the pixel
	if (prob[const_labelIndex_Inf[0]] > const_minProb_Inf[0]) {
		switch (2 - const_labelIndex_Inf[0]) {
		case 0: rgb[x_id].x = 255; break;
		case 1: rgb[x_id].y = 255; break;
		case 2: rgb[x_id].z = 255; break;
		default: printf("Error: color decision error ...");
		}
	}
}

// ================== Inference Kernels with Forest in global memory ==================

__global__ void kernel_inference_ForestInGlobal(
	float*		depth,
	int3*		rgb,
	Node_CU*	forest,
	int			parallel_depth_img
	)
{
	// Decide if the thread is out of boundary
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= parallel_depth_img * const_Depth_Info_Inf[2]) {
		return;
	}

	// Decide the basic information of the thread
	const int depth_idx = x_id / const_Depth_Info_Inf[2];											// Get the index of depth image
	const int y_coord = (x_id - const_Depth_Info_Inf[2] * depth_idx) / const_Depth_Info_Inf[0];	// Get the row of the sample
	const int x_coord = (x_id - const_Depth_Info_Inf[2] * depth_idx) % const_Depth_Info_Inf[0];	// Get the column of the sample

	// Declare the probability array for the forest
	float prob[NODE_NUM_LABELS] = { 0 };

	// Initialize the pixel of the output
	rgb[x_id].x = 0;
	rgb[x_id].y = 0;
	rgb[x_id].z = 0;

	// ===== Calculate response =====
	const float depth_local = depth[x_id];

	// Handle the case when depth is zero
	if (depth_local == 0.0) {
		return;
	}

	// ===== Go through the forest =====
	for (int i = 0; i < const_numTrees_Inf[0]; i++) {
		// ===== Go through each tree =====
		unsigned int idx = const_treePartition_Inf[i];

		while (forest[idx].isSplit == 1) {

			// Calculate responses
			float x = forest[idx].feature.x / depth_local + x_coord;
			float y = forest[idx].feature.y / depth_local + y_coord;

			x = (x < 0) ? 0 :
				(x >= const_Depth_Info_Inf[0]) ? const_Depth_Info_Inf[0] - 1 : x;

			y = (y < 0) ? 0 :
				(y >= const_Depth_Info_Inf[1]) ? const_Depth_Info_Inf[1] - 1 : y;

			float depth2 = depth[depth_idx * const_Depth_Info_Inf[2] + int(y) * const_Depth_Info_Inf[0] + int(x)];

			// Calculate the response of the second set of offsets, disabled
			//x = forest[idx].feature.z / depth_local + x_coord;
			//y = forest[idx].feature.w / depth_local + y_coord;

			//x = (x < 0) ? 0 :
			//	(x >= const_Depth_Info_Inf[0]) ? const_Depth_Info_Inf[0] - 1 : x;

			//y = (y < 0) ? 0 :
			//	(y >= const_Depth_Info_Inf[1]) ? const_Depth_Info_Inf[1] - 1 : y;

			// ##### The curand causes memory issues ##### //
			//float response;
			//curandState state;
			//curand_init(clock(), x_id, 0, &state);
			//if (round(curand_uniform(&state)) == 1) {
			//	response = depth2 - depth_local;
			//}
			//else {
			//	response = depth2 - depth[depth_idx * const_Depth_Info_Inf[2] + int(y) * const_Depth_Info_Inf[0] + int(x)];
			//}
			// ########################################### //

			float response = depth2 - depth_local;
			// ============================

			// Decide which branch to goto
			if (response < forest[idx].threshold) {
				idx = const_treePartition_Inf[i] + forest[idx].leftChild;	// Goto left branch
			}
			else {
				idx = const_treePartition_Inf[i] + forest[idx].rightChild;	// Goto right branch
			}
		}

		// Decide if the tree is valid
		if (forest[idx].isSplit != 0) {
			printf("Error: non leaf node reached ...\n");
			return;
		}

		// Retrieve aggregator and calculate probabilities
		int sampleCount = 0;
		for (int j = 0; j < NODE_NUM_LABELS; j++) {
			sampleCount += forest[idx].aggregator[j];
		}

		for (int j = 0; j < NODE_NUM_LABELS; j++) {
			prob[j] += forest[idx].aggregator[j] * 1.0 / (sampleCount * const_numTrees_Inf[0]);
		}
	}

	// Decide the label of the pixel
	if (prob[const_labelIndex_Inf[0]] > const_minProb_Inf[0]) {
		switch (2 - const_labelIndex_Inf[0]) {
		case 0: rgb[x_id].x = 255; break;
		case 1: rgb[x_id].y = 255; break;
		case 2: rgb[x_id].z = 255; break;
		default: printf("Error: color decision error ...");
		}
	}
}

// ================== Functions for constant memory upload ==================

// ===== Train =====

void const_DP_Info_upload(const int* host_DP_Info)
{
	cudaMemcpyToSymbol(const_DP_Info, host_DP_Info, sizeof(int) * DPInfoCount);
}

void const_UT_Info_upload(const int* host_UT_Info)
{
	cudaMemcpyToSymbol(const_UT_Info, host_UT_Info, sizeof(int) * UTInfoCount);
}

void const_CU_Params_upload(const RDF_CU_Param* params)
{
	cudaMemcpyToSymbol(CU_Params, params, sizeof(RDF_CU_Param));
}

// ===== Test =====

void const_DepthInfo_Inf_upload(const int* host_DepthInfo_Inf_)
{
	cudaMemcpyToSymbol(const_Depth_Info_Inf, host_DepthInfo_Inf_, sizeof(int) * DepthInfoCount_Inf);
}

void const_numTrees_Inf_upload(const int* numTrees_)
{
	cudaMemcpyToSymbol(const_numTrees_Inf, numTrees_, sizeof(int));
}

void const_numLabels_Inf_upload(const int* numLabels_)
{
	cudaMemcpyToSymbol(const_numLabels_Inf, numLabels_, sizeof(int));
}

void const_maxDepth_Inf_upload(const int* maxDepth_)
{
	cudaMemcpyToSymbol(const_maxDepth_Inf, maxDepth_, sizeof(int));
}

void const_labelIndex_Inf_upload(const int* labelIndex_)
{
	cudaMemcpyToSymbol(const_labelIndex_Inf, labelIndex_, sizeof(int));
}

void const_minProb_Inf_upload(const float* minProb_)
{
	cudaMemcpyToSymbol(const_minProb_Inf, minProb_, sizeof(float));
}

void const_totalNode_Inf_upload(const int* host_total_nodeNum_)
{
	cudaMemcpyToSymbol(const_totalNode_Inf, host_total_nodeNum_, sizeof(int));
}

void const_treePartition_Inf_upload(const int* host_partitionInfo_)
{
	cudaMemcpyToSymbol(const_treePartition_Inf, host_partitionInfo_, sizeof(int) * (TREE_CONTAINER_SIZE + 1));
}