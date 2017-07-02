// ================================================== //
// CUDA Random Decision Forest
// Author: Fan Wang
// Date: 06/25/2017
// Note: add desired preprocessor for compilation
//		 USE_GPU_TRAINING    : for GPU RDF training
//		 USE_GPU_INFERENCE   : for GPU Inference
//		 ENABLE_GPU_OBSOLETE : obsolete functions, not tested
// ================================================== //

#include <windows.h>

#include "RDF_CU.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define Thres 0.01

#define DPInfoCount 4

#define UTInfoCount 1

#define space_log_size 600

#define default_block_X 256

#define block_X_512 512

#define block_X_1024 1024

#define PROCESS_LIMIT 3

// #define ENABLE_OLD_KERNELS

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define SWAP_ADDRESS(a, b, t) t = (a); a = (b); b = (t)

__constant__ RDF_CU_Param CU_Params[1];

__constant__ int const_DP_Info[DPInfoCount];

__constant__ int const_UT_Info[UTInfoCount];

cudaDeviceProp inExecution_DeviceProp;

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define cucheck_dev(call)                                   \
{                                                           \
  cudaError_t cucheck_err = (call);                         \
  if(cucheck_err != cudaSuccess) {                          \
    const char *err_str = cudaGetErrorString(cucheck_err);  \
    printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
    assert(0);                                              \
  }                                                         \
}

void CU_MemChecker()
{
	size_t free_byte;
	size_t total_byte;
	gpuErrchk(cudaMemGetInfo(&free_byte, &total_byte));

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

int queryDeviceParams(char* query)
{
	// Returned in bytes
	if (strcmp(query, "sharedMemPerBlock") == 0) {
		return inExecution_DeviceProp.sharedMemPerBlock;
	}

	// Returned in Gigabytes
	if (strcmp(query, "totalGlobalMem") == 0) {
		return inExecution_DeviceProp.totalGlobalMem / 1024 / 1024 / 1024;
	}

	// Returned in bytes
	if (strcmp(query, "totalConstMem") == 0) {
		return inExecution_DeviceProp.totalConstMem;
	}

	return -1;
}

int createCuContext(bool displayDeviceInfo)
{
	int i;
	int count = 0;
	gpuErrchk(cudaGetDeviceCount(&count));
	if (count == 0)
	{
		printf("CUDA: no CUDA device found!\n");
		return -1;
	}
	
	cudaDeviceProp prop;

	// Display CUDA device information
	if (displayDeviceInfo) {
		for (i = 0; i < count; i++) {
			gpuErrchk(cudaGetDeviceProperties(&prop, i));
			printf("======== Device %d ========\n", i);
			printf("Device name: %s\n", prop.name);
			printf("Compute capability: %d.%d\n", prop.major, prop.minor);
			printf("Device copy overlap: ");
			if (prop.deviceOverlap) {
				printf("Enabled\n");
			}
			else {
				printf("Disabled\n");
			}
			printf("Kernel execution timeout: ");
			if (prop.kernelExecTimeoutEnabled) {
				printf("Enabled\n");
			}
			else {
				printf("Disabled\n");
			}

			printf("\n");
			printf("Global memory: %d GB\n", prop.totalGlobalMem / 1024 / 1024 / 1024);
			printf("Constant memory: %d KB\n", prop.totalConstMem / 1024);
			printf("Stream processors count: %d\n", prop.multiProcessorCount);
			printf("Shared memory per stream processor: %d KB\n", prop.sharedMemPerBlock / 1024);
			printf("Registers per stram processor: %d\n", prop.regsPerBlock);
			printf("\n");

			system("pause");
		}
	}

	// Choose the first adequate device for cuda execution
	for (i = 0; i < count; i++)
	{
		// ==== For CUDA Dynamic Parallelism
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
			if ((prop.major >= 3 && prop.minor >= 5) || prop.major >= 4) {
				inExecution_DeviceProp = prop;
				break;
			}
	}

	if (i >= count)
	{
		printf("CUDA: no device has enough capability!\n");
		return -1;
	}

	// Set device i for cuda execution
	cudaSetDevice(i);
	return i;
}

void destroyCuContext()
{
	gpuErrchk(cudaDeviceReset());
}

// ================================================================================================================
// ============================================= TRAINING =========================================================
// ================================================================================================================


// ======================================== Utility Kernels ===========================================

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

// ========================================= Old Kernels ================================================

#ifdef ENABLE_OLD_KERNELS

__global__ void bstResp_gen(int2* cu_coords, float4 bst_feat, int* cu_depID, int* cu_sapID, float* cu_depth_array,
	float* cu_response_array, int sap_num, float thresh_bst, int* partiPos)
{
	__shared__ float4 bstFeat[1];
	if (threadIdx.x == 0)
		bstFeat[0] = bst_feat;
	__syncthreads();

	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= sap_num)
	{
		return;
	}

	float depth = cu_depth_array[cu_depID[cu_sapID[x_id]] * const_DP_Info[3] + cu_coords[cu_sapID[x_id]].y * const_DP_Info[1] + cu_coords[cu_sapID[x_id]].x];
	float x = bstFeat[0].w / depth + cu_coords[cu_sapID[x_id]].x;
	float y = bstFeat[0].x / depth + cu_coords[cu_sapID[x_id]].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	float depth_new = cu_depth_array[cu_depID[cu_sapID[x_id]] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];

	x = bstFeat[0].y / depth + cu_coords[cu_sapID[x_id]].x;
	y = bstFeat[0].z / depth + cu_coords[cu_sapID[x_id]].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	curandState state;
	curand_init(clock(), x_id, 0, &state);
	if (round(curand_uniform(&state)) == 1)
		cu_response_array[x_id] = depth_new - depth;
	else
		cu_response_array[x_id] = depth_new - cu_depth_array[cu_depID[cu_sapID[x_id]] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];

	if (cu_response_array[x_id] < thresh_bst)
		atomicAdd(partiPos, 1);
}

__global__ void response_gen(int2* cu_coords, float4* cu_features, int* cu_depID, float* cu_depth_array,
	float* cu_response_array, int* sap_ID, int* sec_num, int* sec_cntr, int* idxS, int nodesCurrentLevel)
{
	__shared__ int sh[1];
	if (threadIdx.x == 0)
		sh[0] = nodesCurrentLevel;
	__syncthreads();
	// response_array: sap1 sap2 ... sap1 sap2
	int id_sec, id_offset;
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	for (id_sec = 0; id_sec < sh[0]; id_sec++)
		if (x_id < sec_num[id_sec])
			break;
	if (id_sec >= sh[0])
		return;
	id_offset = x_id;
	if (id_sec >= 1)
		id_offset -= sec_num[id_sec - 1];

	int sap_id = idxS[id_sec] + id_offset % sec_cntr[id_sec];
	int fea_id = id_offset / sec_cntr[id_sec];
	float depth = cu_depth_array[cu_depID[sap_ID[sap_id]] * const_DP_Info[3] + cu_coords[sap_ID[sap_id]].y * const_DP_Info[1] + cu_coords[sap_ID[sap_id]].x];
	float x = cu_features[id_sec * CU_Params[0].numFeatures + fea_id].w / depth + cu_coords[sap_ID[sap_id]].x;
	float y = cu_features[id_sec * CU_Params[0].numFeatures + fea_id].x / depth + cu_coords[sap_ID[sap_id]].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	float depth_new = cu_depth_array[cu_depID[sap_ID[sap_id]] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];

	x = cu_features[id_sec * CU_Params[0].numFeatures + fea_id].y / depth + cu_coords[sap_ID[sap_id]].x;
	y = cu_features[id_sec * CU_Params[0].numFeatures + fea_id].z / depth + cu_coords[sap_ID[sap_id]].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	curandState state;
	curand_init(clock(), x_id, 0, &state);
	if (round(curand_uniform(&state)) == 1)
		cu_response_array[x_id] = depth_new - depth;
	else
		cu_response_array[x_id] = depth_new - cu_depth_array[cu_depID[sap_ID[sap_id]] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];
}

__global__ void gain_gen(float* gain, size_t* par_stat, int thresh_num, float par_entropy)
{
	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= thresh_num)
	{
		return;
	}

	unsigned int cntr_l = 0, cntr_r  = 0;
	unsigned int* leftStat  = new unsigned int[CU_Params[0].numLabels];
	unsigned int* rightStat = new unsigned int[CU_Params[0].numLabels];
	memset(leftStat,  0, CU_Params[0].numLabels * sizeof(unsigned int));
	memset(rightStat, 0, CU_Params[0].numLabels * sizeof(unsigned int));

	for (int p = 0; p < thresh_num + 1; p++)
	{
		if (p <= x_id)
			for (int i = 0; i < CU_Params[0].numLabels; i++)
				leftStat[i] += par_stat[p * CU_Params[0].numLabels + i];
		else
			for (int i = 0; i < CU_Params[0].numLabels; i++)
				rightStat[i] += par_stat[p * CU_Params[0].numLabels + i];
	}
	for (int i = 0; i < CU_Params[0].numLabels; i++)
	{
		cntr_l += leftStat[i];
		cntr_r += rightStat[i];
	}

	if ((cntr_l + cntr_r) <= 1)
		gain[x_id] = 0.0;
	else
		gain[x_id] = par_entropy - (cntr_l * entropy_gain(leftStat, cntr_l) + cntr_r * entropy_gain(rightStat, cntr_r)) / (cntr_l + cntr_r);

	delete[] leftStat;
	delete[] rightStat;
}

__global__ void parstat_gen(float* response_array, int* labs, int* sap_ID, int sap_num, int thresh_num, float* thresh, size_t* par_stat)
{
	__shared__ int sh[1];
	if (threadIdx.x == 0)
		sh[0] = thresh_num;
	__syncthreads();

	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= sap_num)
	{
		return;
	}

	int b = 0;
	while (b < sh[0] && response_array[x_id] >= thresh[b])
		b++;
	atomicAdd(&par_stat[b * CU_Params[0].numLabels + labs[sap_ID[x_id]]], 1);
}

__global__ void thresholds_gen(float* cu_response_array, int sap_num, int* labs, int* thresh_num, int* sap_ID, float* thresh, float* gain, size_t* parstat, float par_entropy)
{
	__shared__ int sh[1];
	if (threadIdx.x == 0)
		sh[0] = sap_num;
	__syncthreads();

	int x_id = threadIdx.x + blockDim.x * blockIdx.x;
	if (x_id >= CU_Params[0].numFeatures)
	{
		return;
	}

	curandState state;
	curand_init(clock(), x_id, 0, &state);
	if (sh[0] > CU_Params[0].numTresholds)
	{
		thresh_num[x_id] = CU_Params[0].numTresholds;
		for (int i = 0; i < CU_Params[0].numTresholds + 1; i++)
		{
			int randIdx = round(curand_uniform(&state) * (sh[0] - 1));
			thresh[x_id * (CU_Params[0].numTresholds + 1) + i] = cu_response_array[x_id *sh[0] + randIdx];
		}
	}
	else
	{
		thresh_num[x_id] = sh[0] - 1;
		for (int i = 0; i < sh[0]; i++)
			thresh[x_id * (CU_Params[0].numTresholds + 1) + i] = cu_response_array[x_id *sh[0] + i];
	}

	thrust::sort(thrust::seq, &thresh[x_id * (CU_Params[0].numTresholds + 1)], &thresh[x_id * (CU_Params[0].numTresholds + 1)] + thresh_num[x_id] + 1);
	if (thresh[x_id * (CU_Params[0].numTresholds + 1)] == thresh[x_id * (CU_Params[0].numTresholds + 1) + thresh_num[x_id]])
	{
		thresh_num[x_id] = 0;
		return;
	}

	for (int i = 0; i < thresh_num[x_id]; i++)
		thresh[x_id * (CU_Params[0].numTresholds + 1) + i] += curand_uniform(&state) *
		(thresh[x_id * (CU_Params[0].numTresholds + 1) + i + 1] - thresh[x_id * (CU_Params[0].numTresholds + 1) + i]);

	// ===== Dynamic Parallelism: compute historgram across samples =====
	int blk_genStat = (int)ceil(sh[0] * 1.0 / default_block_X);
	parstat_gen << < blk_genStat, default_block_X >> >(&cu_response_array[x_id * sh[0]], labs, sap_ID, sh[0], thresh_num[x_id], &thresh[x_id * (CU_Params[0].numTresholds + 1)], &parstat[x_id * (CU_Params[0].numTresholds + 1) * CU_Params[0].numLabels]);
	cucheck_dev(cudaGetLastError());
	// ===== Dynamic Parallelism: compute gain across threshold candidates =====
	int blk_genGain = (int)ceil(thresh_num[x_id] * 1.0 / default_block_X);
	gain_gen << < blk_genGain, default_block_X >> > (&gain[x_id * (CU_Params[0].numTresholds + 1)], &parstat[x_id * (CU_Params[0].numTresholds + 1) * CU_Params[0].numLabels], thresh_num[x_id], par_entropy);
	cucheck_dev(cudaGetLastError());
}

#endif

// ========================================= New Kernels ================================================

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
	const int sample_id  = x_id % (PROCESS_LIMIT * SAMPLE_PER_IMAGE);
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

	const int sample_id		= x_id % CU_Params[0].sample_per_tree;
	const int feature_id	= x_id / CU_Params[0].sample_per_tree;

	float depth = cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] +			// Offset to the start of the depth image
				  cu_coords[sample_id].y * const_DP_Info[1] +						// Offset to the start of the corresponding row
				  cu_coords[sample_id].x];											// Offset to the exact depth info

	// Handle the case when depth is zero
	if (depth == 0.0) {
		cu_response_array[x_id] = -10000.0;
		return;
	}

	// Calculate responses
	float x		= cu_features[feature_id].x / depth + cu_coords[sample_id].x;
	float y		= cu_features[feature_id].y / depth + cu_coords[sample_id].y;

	x = (x < 0) ? 0 :
		(x >= const_DP_Info[1]) ? const_DP_Info[1] - 1 : x;
	y = (y < 0) ? 0 :
		(y >= const_DP_Info[2]) ? const_DP_Info[2] - 1 : y;

	float depth2	= cu_depth_array[cu_depthID[sample_id] * const_DP_Info[3] + int(y) * const_DP_Info[1] + int(x)];
	x				= cu_features[feature_id].z / depth + cu_coords[sample_id].x;
	y				= cu_features[feature_id].w / depth + cu_coords[sample_id].y;

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
	if (x_id >= (CU_Params[0].numTresholds + 1) * CU_Params[0].numFeatures) {
		return;
	}

	// Calculate the feature and threshold index
	const int feature_id		= x_id / (CU_Params[0].numTresholds + 1);								 // Index of the feature
	const int thresh_id			= x_id % (CU_Params[0].numTresholds + 1);								 // Index of the threshold
	const int thresh_number		= thresh_num[feature_id];												 // Retrieve the number of thresholds for this feature
	const int feature_offset	= feature_id * (CU_Params[0].numTresholds + 1) * CU_Params[0].numLabels; // Calculate feature offset

	// Abort thread if the threshold exceeds current total number of thresholds
	if (thresh_id >= thresh_number) {
		return;
	}

	unsigned int  left_counter	  = 0;
	unsigned int  right_counter	  = 0;
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
		left_counter	+= left_statistics[feature_offset + thresh_id * CU_Params[0].numLabels + i];
		right_counter	+= right_statistics[feature_offset + thresh_id * CU_Params[0].numLabels + i];
	}

	// Calculate gain for the current threshold
	if ((left_counter + right_counter) <= 1) {
		gain[feature_id * (CU_Params[0].numTresholds + 1) + thresh_id] = 0.0;
	}
	else {
		gain[feature_id * (CU_Params[0].numTresholds + 1) + thresh_id] =
			parent_entropy - (left_counter * entropy_gain(&left_statistics[feature_offset + thresh_id * CU_Params[0].numLabels], left_counter)
			+ right_counter * entropy_gain(&right_statistics[feature_offset + thresh_id * CU_Params[0].numLabels], right_counter) ) / (left_counter + right_counter);
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
	const int feature_id	= x_id / shared_sample_num[0];
	const int rel_sample_id = x_id % shared_sample_num[0];
	const int thresh_number = thresh_num[feature_id];

	// Put the sample label into right bin
	int b = 0;
	while (
		b < thresh_number &&
		response[feature_id * CU_Params[0].sample_per_tree + sample_ID[start_index + rel_sample_id]] // Retrieve the response
		>= thresh[feature_id * (CU_Params[0].numTresholds + 1) + b]								     // Retrieve the thresholds
		)							 
		b++;

	// Add one to the corresponding histogram
	atomicAdd(
		&partition_statistics[feature_id * (CU_Params[0].numTresholds + 1) * CU_Params[0].numLabels // Offset to the start of the feature
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
	const int numThresholds = CU_Params[0].numTresholds;
	if (shared_sample_num[0] > numThresholds) {

		thresh_num[x_id] = numThresholds;
		for (int i = 0; i < numThresholds + 1; i++) {
			int randIdx = round(curand_uniform(&state) * (shared_sample_num[0] - 1));
			thresh[x_id * (numThresholds + 1) + i] = response[x_id * CU_Params[0].sample_per_tree + sample_ID[start_index + randIdx]];
		}
	}
	else {
		thresh_num[x_id] = shared_sample_num[0] - 1;
		for (int i = 0; i < shared_sample_num[0]; i++ ) {
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

// ========================================= Class Methods ================================================

bool RDF_CU::shouldTerminate(float maxGain, int recurseDepth)
{
	return (maxGain < Thres || recurseDepth >= params.maxDepth);
}

void RDF_CU::setParam(RDF_CU_Param& params_)
{
	params = params_;

	// Copy parameters into constant memory
	cudaMemcpyToSymbol(CU_Params, &params, sizeof(RDF_CU_Param));

	return;
}

void RDF_CU::cu_free()
{
	// Synchronize device
	gpuErrchk(cudaDeviceSynchronize());

	// Synchronize cuda streams
	gpuErrchk(cudaStreamSynchronize(execStream));
	gpuErrchk(cudaStreamSynchronize(copyStream));

	// Destroy cuda streams
	gpuErrchk(cudaStreamDestroy(execStream));
	gpuErrchk(cudaStreamDestroy(copyStream));

	// Free batch GPU memory
	gpuErrchk(cudaFree(cu_coords1));
	gpuErrchk(cudaFree(cu_coords2));
	gpuErrchk(cudaFree(cu_depID1));
	gpuErrchk(cudaFree(cu_depID2));
	gpuErrchk(cudaFree(cu_depth_array1));
	gpuErrchk(cudaFree(cu_depth_array2));
	gpuErrchk(cudaFree(cu_sequence1));
	gpuErrchk(cudaFree(cu_sequence2));

	// Clean up all GPU memory
	gpuErrchk(cudaFree(cu_sapID));
	gpuErrchk(cudaFree(cu_labels));
	gpuErrchk(cudaFree(cu_features));
	gpuErrchk(cudaFree(cu_response_array));
	gpuErrchk(cudaFree(cu_partitionStatistics));
	gpuErrchk(cudaFree(cu_leftStatistics));
	gpuErrchk(cudaFree(cu_rightStatistics));
	gpuErrchk(cudaFree(cu_thresh_num));
	gpuErrchk(cudaFree(cu_thresh));
	gpuErrchk(cudaFree(cu_gain));

	// clean up CPU memory
	delete[] parentStatistics;

	host_labels.clear();
	host_sapID.clear();
	host_response_array.clear();
}

// Compute entropy on the give statistics
float RDF_CU::entropy_compute(unsigned int* stat, int boundary)
{
	int cntr = 0;
	for (int i = 0; i < boundary; i++)
		cntr += stat[i];
	if (cntr == 0)
		return 0.0;

	float res = 0.0;
	for (int b = 0; b < params.numLabels; b++)
	{
		float p = stat[b] * 1.0 / cntr;
		res -= (p == 0.0) ? 0.0 : p * log(p) / log(2.0);
	}
	return res;
}

void RDF_CU::cu_initialize()
{
	const int numFeatures   = params.numFeatures;
	const int samplePerTree = params.sample_per_tree;

	// Constant GPU Memory
	const int thresh_num_size	= numFeatures * sizeof(int);
	const int feature_size		= numFeatures * sizeof(float4);
	const int thresh_size		= (params.numTresholds + 1) * numFeatures * sizeof(float);
	const int parstat_size		= params.numLabels * (params.numTresholds + 1) * numFeatures * sizeof(unsigned int);
	
	gpuErrchk(cudaMalloc((void**)&cu_features,				feature_size));
	gpuErrchk(cudaMalloc((void**)&cu_partitionStatistics,	parstat_size));
	gpuErrchk(cudaMalloc((void**)&cu_leftStatistics,		parstat_size));
	gpuErrchk(cudaMalloc((void**)&cu_rightStatistics,		parstat_size));
	gpuErrchk(cudaMalloc((void**)&cu_thresh_num,			thresh_num_size));
	gpuErrchk(cudaMalloc((void**)&cu_thresh,				thresh_size));
	gpuErrchk(cudaMalloc((void**)&cu_gain,					thresh_size));
	
	// Linear GPU memory
	const int labels_size	= samplePerTree * sizeof(int);
	const int response_size = samplePerTree * numFeatures * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&cu_sapID,				labels_size));
	gpuErrchk(cudaMalloc((void**)&cu_labels,			labels_size));
	gpuErrchk(cudaMalloc((void**)&cu_response_array,	response_size));

	// Constant under new architecure
	const int depthID_size	= PROCESS_LIMIT * params.numSamples * sizeof(int);
	const int coords_size	= PROCESS_LIMIT * params.numSamples * sizeof(int2);
	const int depth_size	= PROCESS_LIMIT * width * height * sizeof(float);

	gpuErrchk(cudaMalloc((void**)&cu_coords1,		coords_size));
	gpuErrchk(cudaMalloc((void**)&cu_coords2,		coords_size));
	gpuErrchk(cudaMalloc((void**)&cu_sequence1,		depthID_size));
	gpuErrchk(cudaMalloc((void**)&cu_sequence2,		depthID_size));
	gpuErrchk(cudaMalloc((void**)&cu_depID1,		depthID_size));
	gpuErrchk(cudaMalloc((void**)&cu_depID2,		depthID_size));
	gpuErrchk(cudaMalloc((void**)&cu_depth_array1,	depth_size));
	gpuErrchk(cudaMalloc((void**)&cu_depth_array2,	depth_size));

	// Create cuda streams
	gpuErrchk(cudaStreamCreate(&execStream));
	gpuErrchk(cudaStreamCreate(&copyStream));

	// Synchronize device
	gpuErrchk(cudaDeviceSynchronize());

	// Declare CPU Memory
	parentStatistics = new unsigned int[params.numLabels];

	// Determine the number of nodes in the balanced binary tree
	nodes_size = (1 << params.maxDepth) - 1;
}

void RDF_CU::cu_reset()
{
	const int numFeatures	= params.numFeatures;
	const int samplePerTree = params.sample_per_tree;

	// Calculate the sizes of the arrays
	const int labels_size	= samplePerTree * sizeof(int);
	const int feature_size	= numFeatures * sizeof(float4);
	const int response_size	= samplePerTree * numFeatures * sizeof(float);

	// Reset gpu memory to zero
	cudaMemset(cu_sapID,			0, labels_size);
	cudaMemset(cu_labels,			0, labels_size);
	cudaMemset(cu_features,			0, feature_size);
	cudaMemset(cu_response_array,	0, response_size);

	// Constant under new architecure
	const int depthID_size	= PROCESS_LIMIT * params.numSamples * sizeof(int);
	const int coords_size	= PROCESS_LIMIT * params.numSamples * sizeof(int2);
	const int depth_size	= PROCESS_LIMIT * width * height * sizeof(float);

	// Synchronize cuda streams
	gpuErrchk(cudaStreamSynchronize(execStream));
	gpuErrchk(cudaStreamSynchronize(copyStream));

	cudaMemset(cu_coords1,			0, coords_size);
	cudaMemset(cu_coords2,			0, coords_size);
	cudaMemset(cu_sequence1,		0, depthID_size);
	cudaMemset(cu_sequence2,		0, depthID_size);
	cudaMemset(cu_depID1,			0, depthID_size);
	cudaMemset(cu_depID2,			0, depthID_size);
	cudaMemset(cu_depth_array1,		0, depth_size);
	cudaMemset(cu_depth_array2,		0, depth_size);

	// Reset cpu memory
	host_labels.clear();
	host_response_array.clear();
	host_sapID.clear();

	// Initialize Sample ID on Host
	host_sapID.reserve(samplePerTree);
	for (int i = 0; i < samplePerTree; i++) {
		host_sapID[i] = i;
	}

	// Call CUDA kernel to initialize Sample ID on Device
	int blk_sapIDIni = (int)ceil(params.sample_per_tree * 1.0 / default_block_X);
	sapID_ini << <blk_sapIDIni, default_block_X >> >(cu_sapID);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

void RDF_CU::cu_depth_const_upload(int depth_image_num, int width_, int height_)
{
	// Basic depth information
	depth_num	= depth_image_num;
	width		= width_;
	height		= height_;

	// copy depth information into constant memory
	int host_DP_Info[DPInfoCount] = { depth_num, width, height, width * height };
	cudaMemcpyToSymbol(const_DP_Info, host_DP_Info, sizeof(int) * DPInfoCount);

	// copy other information into constant memory
	int host_UT_Info[UTInfoCount] = { params.sample_per_tree * params.numFeatures };
	cudaMemcpyToSymbol(const_UT_Info, host_UT_Info, sizeof(int) * UTInfoCount);
}

#ifdef ENABLE_GPU_OBSOLETE

void RDF_CU::cu_init(int2* sample_coords, int* sample_labels, int* sample_depID, float4* features)
{
	const int numFeatures	= params.numFeatures;
	const int samplePerTree = params.sample_per_tree;

	int feature_size		= numFeatures * sizeof(float4);
	int coords_size			= samplePerTree * sizeof(int2);
	int labels_size			= samplePerTree * sizeof(int);
	int thresh_num_size		= numFeatures * sizeof(int);
	int response_size		= samplePerTree * numFeatures * sizeof(float);
	int thresh_size			= (params.numTresholds + 1) * numFeatures * sizeof(float);
	int parstat_size		= params.numLabels * (params.numTresholds + 1) * numFeatures * sizeof(size_t);

	// Declare CUDA Memory
	cudaMalloc((void**)&cu_coords, coords_size);
	cudaMalloc((void**)&cu_sapID, labels_size);
	cudaMalloc((void**)&cu_labels, labels_size);
	cudaMalloc((void**)&cu_depID, labels_size);
	cudaMalloc((void**)&cu_features, feature_size);
	cudaMalloc((void**)&cu_response_array, response_size);
	cudaMalloc((void**)&cu_partitionStatistics, parstat_size);
	cudaMalloc((void**)&cu_thresh_num, thresh_num_size);
	cudaMalloc((void**)&cu_thresh, thresh_size);
	cudaMalloc((void**)&cu_gain, thresh_size);

	// Declare CPU Memory
	parentStatistics = new size_t[params.numLabels];

	// Copy to Constant GPU Memory
	// cudaMemcpyToSymbol(CU_Params, &params, sizeof(RDF_CU_Param));

	// Copy to GPU Memory
	cudaMemcpy(cu_coords, sample_coords, coords_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cu_labels, sample_labels, labels_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cu_depID, sample_depID, labels_size, cudaMemcpyHostToDevice);
	cudaMemcpy(cu_features, features, feature_size, cudaMemcpyHostToDevice);

	// Copy to CPU Memory
	host_labels = thrust::host_vector<int>(sample_labels, sample_labels + samplePerTree);

	// Call CUDA kernel to initialize Sample ID on Device
	int blk_sapIDIni = (int)ceil(params.sample_per_tree * 1.0 / default_block_X);
	sapID_ini << <blk_sapIDIni, default_block_X >> >(cu_sapID);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Initialize Sample ID on Host
	host_sapID.reserve(samplePerTree);
	for (int i = 0; i < samplePerTree; i++) {
		host_sapID[i] = i;
	}
	nodes_size = (1 << params.maxDepth) - 1;

	//CU_MemChecker();
}

void RDF_CU::cu_dataTransfer(int2* sample_coords, int* sample_labels, int* sample_depID, float4* features)
{
	const int numFeatures	= params.numFeatures;
	const int samplePerTree = params.sample_per_tree;

	int labels_size		= samplePerTree * sizeof(int);
	int coords_size		= samplePerTree * sizeof(int2);
	int feature_size	= numFeatures * sizeof(float4);

	// Copy to GPU Memory
	cudaMemcpy(cu_coords,	sample_coords,	coords_size,	cudaMemcpyHostToDevice);
	cudaMemcpy(cu_labels,	sample_labels,	labels_size,	cudaMemcpyHostToDevice);
	cudaMemcpy(cu_depID,	sample_depID,	labels_size,	cudaMemcpyHostToDevice);
	cudaMemcpy(cu_features, features,		feature_size,	cudaMemcpyHostToDevice);

	// Copy to CPU Memory
	host_labels = thrust::host_vector<int>(sample_labels, sample_labels + samplePerTree);
}

void RDF_CU::cu_depthTransfer(float* depth_array)
{
	// Copy depth information into constant memory
	int depth_array_size = depth_num * width * height * sizeof(float);
	cudaMemcpy(cu_depth_array, depth_array, depth_array_size, cudaMemcpyHostToDevice);
}

void RDF_CU::cu_depth_init(int depth_image_num, int width_, int height_, float* depth_array)
{
	// Basic depth information
	depth_num	= depth_image_num;
	width		= width_;
	height		= height_;

	// Copy depth information into constant memory
	int depth_array_size = depth_num * width * height * sizeof(float);
	int host_DP_Info[DPInfoCount] = { depth_num, width, height, width * height };
	cudaMemcpyToSymbol(const_DP_Info, host_DP_Info, sizeof(int) * DPInfoCount);

	// Copy other information into constant memory
	int host_UT_Info[UTInfoCount] = { params.sample_per_tree * params.numFeatures };
	cudaMemcpyToSymbol(const_UT_Info, host_UT_Info, sizeof(int) * UTInfoCount);

	// Allocate memory and copy depth into global memory
	cudaMalloc((void**)&cu_depth_array, depth_array_size);
	cudaMemcpy(cu_depth_array, depth_array, depth_array_size, cudaMemcpyHostToDevice);

	//CU_MemChecker();
}

void RDF_CU::host_free()
{
	host_labels.clear();
	host_sapID.clear();
	host_response_array.clear();
}

#endif

void RDF_CU::cu_featureTransfer(float4 *features_)
{
	const int feature_size = params.numFeatures * sizeof(float4);
	cudaMemcpy(cu_features, features_, feature_size, cudaMemcpyHostToDevice);
}

void RDF_CU::compute_responses(std::vector<rdf::Sample>& samples_per_tree)
{
	const int numImages			= params.numPerTree;
	const int samplesPerImage	= params.numSamples;
	const int totalSamples		= samples_per_tree.size();

	// Batch copy sizes
	const int depthID_size		= PROCESS_LIMIT * samplesPerImage * sizeof(int);
	const int coords_size		= PROCESS_LIMIT * samplesPerImage * sizeof(int2);
	const int depth_array_size	= PROCESS_LIMIT * width * height * sizeof(float);

	// Check the validity of the memory settings
	assert(totalSamples > 0);
	assert(samplesPerImage == SAMPLE_PER_IMAGE);
	assert(samplesPerImage * numImages == totalSamples);

	// Declare arrays for memory swapping purpose
	int*	cu_depID_t;
	int*	cu_sequence_t;
	int2*	cu_coords_t;
	float*	cu_depth_array_t;

	// Declare CPU memory
	int *host_depthID	= new int[PROCESS_LIMIT * samplesPerImage];
	int *host_sequence	= new int[PROCESS_LIMIT * samplesPerImage];
	int2 *host_coords	= new int2[PROCESS_LIMIT * samplesPerImage];
	float *host_depth	= new float[PROCESS_LIMIT * width  * height];

	// Declare variables to hold start and end indices of the depth images
	int start_index;
	int end_index;

	// Fill in the labels of the samples
	host_labels.resize(totalSamples);
	for (int i = 0; i < totalSamples; i++) {
		host_labels[i] = samples_per_tree[i].getLabel();
	}

	// Copy labels from host to device
	cudaMemcpy(cu_labels, &host_labels[0], totalSamples * sizeof(int), cudaMemcpyHostToDevice);

	// Utility for copying depths
	float* worker_array;
	std::vector<int> copied_indicator(numImages, 0);

	// Calculate data segmentation
	int launch_rounds = ceil(numImages * 1.0 / PROCESS_LIMIT);
	for (int i = 0; i <= launch_rounds; i++) {

		// ===== Prepare CPU data =====
		if (i < launch_rounds) {

			// Determine the start and end indices of the depth images
			start_index = i * PROCESS_LIMIT;
			end_index = ((i + 1) * PROCESS_LIMIT > numImages) ? numImages : (i + 1) * PROCESS_LIMIT; // Exlusive

			int cntr = 0;
			for (int j = 0; j < totalSamples; j++) {

				const rdf::Sample& sample = samples_per_tree[j];
				const int& depthID = sample.getDepthID();
				if (depthID >= start_index && depthID < end_index) {
					host_depthID[cntr] = depthID % PROCESS_LIMIT;
					host_sequence[cntr] = j;
					host_coords[cntr].x = sample.getCoor().x;
					host_coords[cntr].y = sample.getCoor().y;

					if (copied_indicator[depthID] != 1) {
						copied_indicator[depthID] = 1;

						// Check if the dimensions are consistant
						assert(sample.getDepth().rows == height);
						assert(sample.getDepth().cols == width);

						worker_array = (float*)sample.getDepth().data;
						std::memcpy(&host_depth[host_depthID[cntr] * width * height], worker_array, width * height * sizeof(float));
					}

					cntr++;
				}
			}
			assert(cntr == PROCESS_LIMIT * samplesPerImage);
		}
		// =============================

		// Synchronize cuda streams
		gpuErrchk(cudaStreamSynchronize(copyStream));
		gpuErrchk(cudaStreamSynchronize(execStream));

		// Swap memory buffers
		SWAP_ADDRESS(cu_depID1,			cu_depID2,			cu_depID_t);
		SWAP_ADDRESS(cu_sequence1,		cu_sequence2,		cu_sequence_t);
		SWAP_ADDRESS(cu_coords1,		cu_coords2,			cu_coords_t);
		SWAP_ADDRESS(cu_depth_array1,	cu_depth_array2,	cu_depth_array_t);

		// Copy data from host to device
		if (i < launch_rounds) {

			cudaMemcpyAsync(cu_depID2,			host_depthID,	depthID_size,		cudaMemcpyHostToDevice, copyStream);
			cudaMemcpyAsync(cu_sequence2,		host_sequence,	depthID_size,		cudaMemcpyHostToDevice, copyStream);
			cudaMemcpyAsync(cu_coords2,			host_coords,	coords_size,		cudaMemcpyHostToDevice, copyStream);
			cudaMemcpyAsync(cu_depth_array2,	host_depth,		depth_array_size,	cudaMemcpyHostToDevice, copyStream);
		}

		// Lanuch response computation kernel
		if (i > 0) {

			// Call CUDA kernel to calculate responses: FeatureNum x a subset of samples
			int blkSet_compute_responses_batch = (int)ceil(PROCESS_LIMIT * samplesPerImage * params.numFeatures * 1.0 / default_block_X);
			kernel_compute_response_batch << <blkSet_compute_responses_batch, default_block_X, 0, execStream >> >(
				end_index - start_index,
				cu_depID1,
				cu_sequence1,
				cu_coords1,
				cu_depth_array1,
				cu_response_array,
				cu_features
				);
		}
	}

	// Synchonize cuda execution stream
	gpuErrchk(cudaStreamSynchronize(execStream));

	// Check for kernel errors
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Copy responses from global memory to host memory
	thrust::device_ptr<float> dev_response_ptr(cu_response_array);
	host_response_array = thrust::host_vector<float>(dev_response_ptr, dev_response_ptr + params.sample_per_tree * params.numFeatures);

	// Clean up memory
	delete[] host_depthID;
	delete[] host_sequence;
	delete[] host_coords;
	delete[] host_depth;

	copied_indicator.clear();
}

void RDF_CU::cu_train(std::vector<rdf::Node>& nodes)
{
	// ===== Initialization =====
	std::vector<int> idx(1);
	std::vector<int> idx_S(1);
	std::vector<int> idx_E(1);
	idx[0] = 0;
	idx_S[0] = 0;
	idx_E[0] = params.sample_per_tree;

	int recurseDepth = 0;
	nodes.resize(nodes_size);
	while (idx_S.size() != 0)
	{
		// Train current level
		cu_trainLevel(idx, idx_S, idx_E, nodes, idx_S.size(), recurseDepth);
		recurseDepth++;
	}

	// ===== Free Memory =====
	idx.clear();
	idx_S.clear();
	idx_E.clear();
}

void RDF_CU::cu_trainLevel(
	std::vector<int>&		idx,
	std::vector<int>&		idx_S,
	std::vector<int>&		idx_E,
	std::vector<rdf::Node>& nodes,
	int						nodesCurrentLevel,
	int						recurseDepth)
{
	std::vector<int> idx_Nodes;		// Indices of the nodes
	std::vector<int> idx_Start;		// Start indices of the samples for each node
	std::vector<int> idx_End;		// End indices of the samples for each node

	// ===== Calculate array sizes =====
	int		size_thresh_num  = params.numFeatures * sizeof(int);
	int		size_thresh		 = (params.numTresholds + 1) * params.numFeatures * sizeof(float);
	int		size_parstat	 = params.numLabels * (params.numTresholds + 1) * params.numFeatures * sizeof(unsigned int);

	for (int i = 0; i < nodesCurrentLevel; i++) {

		// Initialization for current node
		memset(parentStatistics,			0, params.numLabels * sizeof(unsigned int));	// Initialize for parent statistics in CPU memory
		cudaMemset(cu_thresh_num,			0, size_thresh_num);							// Initialize for thresh_num array in GPU
		cudaMemset(cu_thresh,				0, size_thresh);								// Initialize for thresholds array in GPU
		cudaMemset(cu_gain,					0, size_thresh);								// Initialize for gain array in GPU		
		cudaMemset(cu_partitionStatistics,	0, size_parstat);								// Initialize for partition statistics in GPU
		cudaMemset(cu_leftStatistics,		0, size_parstat);								// Initialize for left partition statistics on GPU
		cudaMemset(cu_rightStatistics,		0, size_parstat);								// Initialize for right partition statistics on GPU

		// Calculate parent statistics
		float parent_entropy;
		for (int j = idx_S[i]; j < idx_E[i]; j++) {
			parentStatistics[host_labels[host_sapID[j]]] += 1;
		}
		parent_entropy = entropy_compute(parentStatistics, params.numLabels);

		// Decide if the node is leaf or not
		const int idx_node   = idx[i];
		const int sample_num = idx_E[i] - idx_S[i];

		if (idx_node >= nodes_size / 2 || sample_num <= 1) {
			rdf::Aggregator statistics;
			statistics.manualSet(parentStatistics, params.numLabels);
			nodes[idx_node].initializeLeaf(statistics, idx_node);

			continue;
		}

		// Call CUDA kernel to generate thresholds for each feature
		int blkSet_generate_thresholds = (int)ceil(params.numFeatures * 1.0 / default_block_X);
		kernel_generate_thresholds << <blkSet_generate_thresholds, default_block_X >> >(
			cu_response_array,
			cu_thresh_num,
			cu_thresh,
			cu_sapID,
			idx_S[i],
			sample_num
			);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Call CUDA kernel to compute histograms across all samples
		int blkSet_compute_histograms = (int)ceil(params.numFeatures * sample_num * 1.0 / default_block_X);
		kernel_compute_partitionStatistics << <blkSet_compute_histograms, default_block_X >> >(
			cu_response_array,
			cu_thresh,
			cu_labels,
			cu_thresh_num,
			cu_sapID,
			idx_S[i],
			sample_num,
			cu_partitionStatistics
			);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Call CUDA kernel to compute gain for each of the thresholds
		int blkSet_compute_gain = (int)ceil(params.numFeatures * (params.numTresholds + 1) * 1.0 / default_block_X);
		kernel_compute_gain << <blkSet_compute_gain, default_block_X >> > (
			cu_gain,
			cu_thresh_num,
			parent_entropy,
			cu_leftStatistics,
			cu_rightStatistics,
			cu_partitionStatistics
			);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// Sort the computed gains and find the best feature and best threshold
		thrust::device_vector<float> thrust_gain(cu_gain, cu_gain + (params.numTresholds + 1) * params.numFeatures);
		thrust::device_vector<float>::iterator iter = thrust::max_element(thrust_gain.begin(), thrust_gain.end());
		float		 max_gain			= *iter;
		unsigned int position			= iter - thrust_gain.begin();
		int			 best_feature_idx	= position / (params.numTresholds + 1);

		// Decide the status of current node and abort if necessary
		if (max_gain == 0.0 || shouldTerminate(max_gain, recurseDepth)) {
			rdf::Aggregator statistics;
			statistics.manualSet(parentStatistics, params.numLabels);
			nodes[idx_node].initializeLeaf(statistics, idx_node);

			continue;
		}

		// Copy the best threshold and best feature from GPU memory to host memory
		float  best_threshold;
		float4 best_feature;
		cudaMemcpy(&best_threshold, &cu_thresh[position],           sizeof(float),  cudaMemcpyDeviceToHost);
		cudaMemcpy(&best_feature,   &cu_features[best_feature_idx], sizeof(float4), cudaMemcpyDeviceToHost);

		// Set current node as leaf
		rdf::Feature bestFeature_rdf;
		bestFeature_rdf.manualSet(best_feature.x, best_feature.y, best_feature.z, best_feature.w);
		nodes[idx_node].initializeSplit(bestFeature_rdf, best_threshold, idx_node);

		// ===== Partition =====	
		thrust::host_vector<float> selected_responses;
		selected_responses.reserve(sample_num);
		for (int j = 0; j < sample_num; j++) {
			selected_responses.push_back(host_response_array[best_feature_idx * params.sample_per_tree + host_sapID[idx_S[i] + j]]);
		}

		// ===== Calculate the partition position =====
		// This partition calculation should give results more similar to CPU version
		int start_pointer = 0;
		int end_pointer	  = selected_responses.size() - 1;

		while (start_pointer != end_pointer) {
			if (selected_responses[start_pointer] >= best_threshold) {\
				int	  id_swap = host_sapID[idx_S[i] + start_pointer];
				float key_swap	= selected_responses[start_pointer];
				
				// Swap the two responses and indices
				selected_responses[start_pointer]		= selected_responses[end_pointer];
				host_sapID[idx_S[i] + start_pointer]	= host_sapID[idx_S[i] + end_pointer];

				selected_responses[end_pointer]		= key_swap;
				host_sapID[idx_S[i] + end_pointer]	= id_swap;

				end_pointer--;
			}
			else {
				start_pointer++;
			}
		}

		// Determine the parition index
		int partition_index = selected_responses[start_pointer] >= best_threshold ? start_pointer : start_pointer + 1;

		// Clean the vector
		selected_responses.clear();
		// ===============================================================

		// ====== Sort the responses: unknown impact on the results =====
		//thrust::sort_by_key(selected_responses.begin(), selected_responses.end(), &host_sapID[idx_S[i]]);

		//// Determine where to partition the indices
		//int partition_index = 0;
		//for (int j = 0; j < sample_num; j++) {
		//	if (selected_responses[j] < best_threshold) {
		//		partition_index++;
		//	}
		//	else {
		//		break;
		//	}
		//}
		// ===============================================================

		// Push partitions
		idx_Nodes.push_back(idx_node * 2 + 1);
		idx_Nodes.push_back(idx_node * 2 + 2);
		idx_Start.push_back(idx_S[i]);
		idx_Start.push_back(idx_S[i] + partition_index);
		idx_End.push_back(idx_S[i] + partition_index);
		idx_End.push_back(idx_E[i]);
	}

	// ===== Current Level Complete =====
	idx		= idx_Nodes;
	idx_S	= idx_Start;
	idx_E	= idx_End;
	
	// Synchronize device sample indices
	cudaMemcpy(cu_sapID, host_sapID.data(), params.sample_per_tree * sizeof(int), cudaMemcpyHostToDevice);
}

// ================================================================================================================
// ============================================= INFERENCE ========================================================
// ================================================================================================================

#define DepthInfoCount_Inf 3

#define MAX_PARALLEL_IMG 4

#define TREE_CONTAINER_SIZE	3

// Declare constant memory
__constant__ int	const_Depth_Info_Inf[DepthInfoCount_Inf];

__constant__ int	const_numTrees_Inf[1];								// Number of trees

__constant__ int	const_numLabels_Inf[1];								// Number of labels

__constant__ int	const_maxDepth_Inf[1];								// Max depth

__constant__ int	const_labelIndex_Inf[1];							// Label index

__constant__ float	const_minProb_Inf[1];								// Min probability

__constant__ int	const_totalNode_Inf[1];								// Total number of nodes in the forest

__constant__ int	const_treePartition_Inf[TREE_CONTAINER_SIZE + 1];	// Store the ACCUMULATIVE offsets of each starting index

// ================== Inference Kernels with Forest in shared memory ==================

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
	const int y_coord	= (x_id - const_Depth_Info_Inf[2] * depth_idx) / const_Depth_Info_Inf[0];	// Get the row of the sample
	const int x_coord	= (x_id - const_Depth_Info_Inf[2] * depth_idx) % const_Depth_Info_Inf[0];	// Get the column of the sample

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

			float depth2	= depth[depth_idx * const_Depth_Info_Inf[2] + int(y) * const_Depth_Info_Inf[0] + int(x)];

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
	const int y_coord   = (x_id - const_Depth_Info_Inf[2] * depth_idx) / const_Depth_Info_Inf[0];	// Get the row of the sample
	const int x_coord   = (x_id - const_Depth_Info_Inf[2] * depth_idx) % const_Depth_Info_Inf[0];	// Get the column of the sample

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

CUDA_INFER::CUDA_INFER(
	const int							parallel_proc_,
	const int							depth_width_,
	const int							depth_height_,
	const bool							createCUContext_,
	std::vector<std::vector<Node_CU>>&	forest_,
	const bool							realTime_,
	std::string							out_directory_
	) :
	parallel_proc	(parallel_proc_),
	depth_width		(depth_width_),
	depth_height	(depth_height_),
	createCUContext (createCUContext_),
	depthArray_size	(parallel_proc_ * depth_width_ * depth_height_ * sizeof(float)),
	rgbArray_size	(parallel_proc_ * depth_width_ * depth_height_ * sizeof(int3)),
	out_directory	(out_directory_),
	realTime		(realTime_),
	firstFrame		(true)
{
	// Create CUDA context for the program
	if (createCUContext) {
		createCuContext(false);
	}

	// Determine if to put forest in shared memory
	const int node_num = cu_ifInShared(forest_);

	// Upload depth information to constant memory
	upload_DepthInfo_Inf(depth_width_, depth_height_);

	// Calculate the kernel launch parameters
	blk_Inference = (int)ceil(depth_width_ * depth_height_ * parallel_proc_ * 1.0f / default_block_X);

	frame_cntr = 0;								// Set the frame counter to zero

	// Allocate device memory
	gpuErrchk(cudaMalloc(&device_depthArray1, depthArray_size));
	gpuErrchk(cudaMalloc(&device_rgbArray1, rgbArray_size));
	gpuErrchk(cudaMalloc(&device_forest, forest_size));

	// Initialize device memory
	gpuErrchk(cudaMemset(device_depthArray1, 0, depthArray_size));
	gpuErrchk(cudaMemset(device_rgbArray1, 0, rgbArray_size));

	// Allocate the host output array
	host_rgbArray1 = new int3[depth_width_ * depth_height_];

	// Initialize host array
	memset(host_rgbArray1, 0, depth_width_ * depth_height_ * sizeof(int3));

	if (!realTime) {
		// Allocate device memory
		gpuErrchk(cudaMalloc(&device_depthArray2, depthArray_size));
		gpuErrchk(cudaMalloc(&device_rgbArray2, rgbArray_size));

		// Initialize device memory
		gpuErrchk(cudaMemset(device_depthArray2, 0, depthArray_size));
		gpuErrchk(cudaMemset(device_rgbArray2, 0, rgbArray_size));

		// Create cuda streams
		gpuErrchk(cudaStreamCreate(&execStream));
		gpuErrchk(cudaStreamCreate(&copyStream));

		// Allocate the host output array
		host_rgbArray2 = new int3[depth_width_ * depth_height_];

		// Initialize host array
		memset(host_rgbArray2, 0, depth_width_ * depth_height_ * sizeof(int3));
	}
	
	// Prepare host forest
	host_forest = new Node_CU[node_num];

	// Copy forest into host array
	int cntr = 0;
	for (int i = 0; i < forest_.size(); i++) {
		for (int j = 0; j < forest_[i].size(); j++) {
			host_forest[cntr++] = forest_[i][j];
		}
	}
	assert(cntr == node_num);

	// Synchronously copy forest into device memory
	cudaMemcpy(device_forest, host_forest, forest_size, cudaMemcpyHostToDevice);

	// Decide if to output results
	writeOut = (out_directory == "") ? false : true;
	if (writeOut) {
		out = boost::filesystem::path(out_directory);
		if (!boost::filesystem::exists(out)) {
			boost::filesystem::create_directory(out);
		}
	}

	// Free host forest immediately for performance purpose
	free(host_forest);
}

CUDA_INFER::~CUDA_INFER()
{
	// Free all device memory
	cu_deviceMemFree();

	// Free all host memory
	cu_hostMemFree();

	// Reset device
	if (createCUContext) {
		destroyCuContext();
	}
}

void CUDA_INFER::upload_DepthInfo_Inf(
	int width_,
	int height_
	)
{
	int host_DepthInfo_Inf[DepthInfoCount_Inf] = { width_, height_, width_ * height_ };
	cudaMemcpyToSymbol(const_Depth_Info_Inf, host_DepthInfo_Inf, sizeof(int) * DepthInfoCount_Inf);
}

void CUDA_INFER::cu_upload_TreeInfo(
	int									numLabels_,
	int									maxDepth_,
	float								minProb_,
	std::vector<std::vector<Node_CU>>&	forest_
	)
{
	// Assuming forest contains correct number of trees, MIGHT BE RISKY
	int numTrees_	= forest_.size();

	// Assuming 0 for labelIndex_, MIGHT BE RISKY
	int	labelIndex_ = 0;

	// Determine if the input number of trees exceed the partition container limit
	if (numTrees_ > TREE_CONTAINER_SIZE) {
		printf("Error: size of tree partition container is hardcoded ...\n");
		exit(-1);
	}

	numLabels_ = numLabels_ + 1;

	// Determine if the given number of labels matches the program setting
	if (numLabels_ != NODE_NUM_LABELS) {
		printf("Error: the number of labels is hardcoded ...\n");
		exit(-1);
	}

	cudaMemcpyToSymbol(const_numTrees_Inf, &numTrees_, sizeof(int));
	cudaMemcpyToSymbol(const_numLabels_Inf, &numLabels_, sizeof(int));
	cudaMemcpyToSymbol(const_maxDepth_Inf, &maxDepth_, sizeof(int));
	cudaMemcpyToSymbol(const_labelIndex_Inf, &labelIndex_, sizeof(int));
	cudaMemcpyToSymbol(const_minProb_Inf, &minProb_, sizeof(float));

	// Upload trees partition information to constant memory
	int host_total_nodeNum = 0;
	int host_partitionInfo[TREE_CONTAINER_SIZE + 1] = { 0 };
	for (int i = 0; i < numTrees_; i++) {
		host_partitionInfo[i + 1] = forest_[i].size() + host_partitionInfo[i];
		host_total_nodeNum += forest_[i].size();
	}
	cudaMemcpyToSymbol(const_totalNode_Inf, &host_total_nodeNum, sizeof(int));
	cudaMemcpyToSymbol(const_treePartition_Inf, host_partitionInfo, sizeof(int) * (TREE_CONTAINER_SIZE + 1));
}

int CUDA_INFER::cu_ifInShared(std::vector<std::vector<Node_CU>>& forest_)
{
	// Default value
	forestInSharedMem = false;

	// Calculate the total number of bytes of the forest
	int node_num = 0;
	for (int i = 0; i < forest_.size(); i++) {
		node_num += forest_[i].size();
	}

	forest_size = node_num * sizeof(Node_CU);

	// Compare the size of the forst with the maximum allowed size of the shared memory
	if (forest_size > queryDeviceParams("sharedMemPerBlock") * 0.8) {
		printf("Forest in global memory ...\n");
	}
	else {
		forestInSharedMem = true;
		printf("Forest in shared memory ...\n");
	}

	return node_num;
}

// Infer frame function for batch mode
cv::Mat_<cv::Vec3i> CUDA_INFER::cu_inferFrame(const cv::Mat_<float>& depth_img)
{
	// Note: under this architecure, the first three frames are blank
	// The last three frames need to be flushed out

	// Check if this is the right function to call
	if (realTime) {
		printf("Call \"cu_inferFrame_hard\" instead ...\n");
		exit(1);
	}

	// Check the validity of the input depth image
	assert(depth_image.rows == depth_height);
	assert(depth_image.cols == depth_width);

	// Decide if this is the first frame
	if (firstFrame) {
		// ====== First frame: copy the depth into the buffer and return

		firstFrame = false;
		cudaMemcpyAsync(device_depthArray2, depth_img.data, depthArray_size, cudaMemcpyHostToDevice, copyStream);
	}
	else {
		// ===== Not first frame: go through regular sequence

		// Synchronize streams
		gpuErrchk(cudaStreamSynchronize(copyStream));
		gpuErrchk(cudaStreamSynchronize(execStream));

		// Swap containers
		SWAP_ADDRESS(device_depthArray1, device_depthArray2, device_depthArray_t);
		SWAP_ADDRESS(device_rgbArray1, device_rgbArray2, device_rgbArray_t);
		SWAP_ADDRESS(host_rgbArray1, host_rgbArray2, device_rgbArray_t);

		// Asynchronisely copy depth into device array
		cudaMemcpyAsync(device_depthArray2, depth_img.data, depthArray_size, cudaMemcpyHostToDevice, copyStream);
		cudaMemcpyAsync(host_rgbArray2, device_rgbArray2, rgbArray_size, cudaMemcpyDeviceToHost, copyStream);

		if (forestInSharedMem) {
			kernel_inference_ForestInShared << <blk_Inference, default_block_X, forest_size, execStream >> > (
				device_depthArray1,
				device_rgbArray1,
				device_forest,
				parallel_proc
				);
		}
		else {
			kernel_inference_ForestInGlobal << <blk_Inference, default_block_X, 0, execStream >> > (
				device_depthArray1,
				device_rgbArray1,
				device_forest,
				parallel_proc
				);
		}
	}

	// Copy the array into a mat strucutre
	cv::Mat_<cv::Vec3i> result(depth_height, depth_width, (cv::Vec3i*)host_rgbArray1);

	// Output results
	if (writeOut) {
		writeResult(result);
	}

	// Increase frame counter by 1
	frame_cntr++;

	// Return the rgb result
	return result;
}

// Infer frame function for real time
cv::Mat_<cv::Vec3i> CUDA_INFER::cu_inferFrame_hard(const cv::Mat_<float>& depth_img)
{
	// Check if this is the right function to call
	if (!realTime) {
		printf("Call \"cu_inferFrame\" instead ...\n");
		exit(1);
	}

	// Check the validity of the input depth image
	assert(depth_image.rows == depth_height);
	assert(depth_image.cols == depth_width);

	// Copy depth from host to device
	cudaMemcpy(device_depthArray1, depth_img.data, depthArray_size, cudaMemcpyHostToDevice);

	// Launch kernel to compute inference
	if (forestInSharedMem) {
		kernel_inference_ForestInShared << <blk_Inference, default_block_X, forest_size >> > (
			device_depthArray1,
			device_rgbArray1,
			device_forest,
			parallel_proc
			);
	}
	else {
		kernel_inference_ForestInGlobal << <blk_Inference, default_block_X, 0 >> > (
			device_depthArray1,
			device_rgbArray1,
			device_forest,
			parallel_proc
			);
	}

	// Check for kernel errors
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// Copy rgb from device to host
	cudaMemcpy(host_rgbArray1, device_rgbArray1, depthArray_size, cudaMemcpyDeviceToHost);

	// Copy the array into a mat structure
	cv::Mat_<cv::Vec3i> result(depth_height, depth_width, (cv::Vec3i*)host_rgbArray1);

	// Output results
	if (writeOut) {
		writeResult(result);
	}

	// Increase frame counter by one
	frame_cntr++;

	// Return the rgb result
	return result;
}

std::vector<cv::Mat_<cv::Vec3i>> CUDA_INFER::flushOutLastThree()
{
	// Check if it is necessary to use this function
	if (realTime) {
		printf("Use this function in batch mode only ...\n");
		exit(1);
	}

	// Vectors to contain last three frames
	std::vector<cv::Mat_<cv::Vec3i>> last3Frame(3);

	for (int i = 0; i < 3; i++) {

		// Synchronize streams
		gpuErrchk(cudaStreamSynchronize(copyStream));
		gpuErrchk(cudaStreamSynchronize(execStream));

		// Swap containers
		SWAP_ADDRESS(device_depthArray1, device_depthArray2, device_depthArray_t);
		SWAP_ADDRESS(device_rgbArray1, device_rgbArray2, device_rgbArray_t);
		SWAP_ADDRESS(host_rgbArray1, host_rgbArray2, device_rgbArray_t);

		// Asynchronisely copy result out to host array
		cudaMemcpyAsync(host_rgbArray2, device_rgbArray2, rgbArray_size, cudaMemcpyDeviceToHost, copyStream);

		if (i == 0) {
			if (forestInSharedMem) {
				kernel_inference_ForestInShared << <blk_Inference, default_block_X, forest_size, execStream >> > (
					device_depthArray1,
					device_rgbArray1,
					device_forest,
					parallel_proc
					);
			}
			else {
				kernel_inference_ForestInGlobal << <blk_Inference, default_block_X, 0, execStream >> > (
					device_depthArray1,
					device_rgbArray1,
					device_forest,
					parallel_proc
					);
			}
		}

		// Copy the array into a mat strucutre
		cv::Mat_<cv::Vec3i> result(depth_height, depth_width, (cv::Vec3i*)host_rgbArray1);
		last3Frame[i] = result;
	}

	if (writeOut) {
		writeResult(last3Frame);
	}

	return last3Frame;
}

void CUDA_INFER::writeResult(const cv::Mat_<cv::Vec3i>& result)
{
	boost::format fmt_result_rgb("%s_result_rgb.png");
	boost::filesystem::path out_rgb;

	// For real time output, no three frame lag
	if (!realTime) {

		// First frames are blank
		if (frame_cntr < 3) {
			return;
		}

		// Assemble names
		out_rgb = out / (fmt_result_rgb % std::to_string(frame_cntr - 3)).str();
	}

	// For batch outputs, three frames lag but better performance
	else {

		out_rgb = out / (fmt_result_rgb % std::to_string(frame_cntr)).str();
	}

	// Write out results
	cv::imwrite(out_rgb.string(), result);
}

void CUDA_INFER::writeResult(const std::vector<cv::Mat_<cv::Vec3i>>& result)
{
	for (int i = 0; i < 3; i++) {
		// Assemble names
		boost::format fmt_result_rgb("%s_result_rgb.png");
		boost::filesystem::path out_rgb = out / (fmt_result_rgb % std::to_string(frame_cntr - (3 - i))).str();

		// Write out results
		cv::imwrite(out_rgb.string(), result[i]);
	}
}

void CUDA_INFER::cu_hostMemFree()
{
	// Free all host memory
	free(host_rgbArray1);

	if (!realTime) {
		free(host_rgbArray2);
	}
}

void CUDA_INFER::cu_deviceMemFree()
{
	if (!realTime) {
		// Synchornize cuda streams
		gpuErrchk(cudaStreamSynchronize(execStream));
		gpuErrchk(cudaStreamSynchronize(copyStream));

		// Destroy cuda streams
		gpuErrchk(cudaStreamDestroy(execStream));
		gpuErrchk(cudaStreamDestroy(copyStream));

		// Clean up device memory
		gpuErrchk(cudaFree(device_depthArray2));
		gpuErrchk(cudaFree(device_rgbArray2));
	}

	// Clean up device memory
	gpuErrchk(cudaFree(device_depthArray1));
	gpuErrchk(cudaFree(device_rgbArray1));
	gpuErrchk(cudaFree(device_forest));
}