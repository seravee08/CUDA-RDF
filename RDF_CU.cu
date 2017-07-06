// ================================================== //
// CUDA Random Decision Forest
// Author: Fan Wang
// Date: 06/25/2017
// Note: add desired preprocessor for compilation
//		 USE_GPU_TRAINING    : for GPU RDF training
//		 USE_GPU_INFERENCE   : for GPU Inference
//		 ENABLE_GPU_OBSOLETE : obsolete functions, not tested
// ================================================== //

#include "RDF_CU.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <time.h>

#define Thres 0.01

#define space_log_size 600

#define default_block_X 256

#define block_X_512 512

#define block_X_1024 1024

// #define ENABLE_OLD_KERNELS

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define SWAP_ADDRESS(a, b, t) t = (a); a = (b); b = (t)

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

// ========================================= Class Methods ================================================

CUDA_TRAIN::CUDA_TRAIN(
	const int	numTrees_,
	const int	numImages_,
	const int	numPerTree_,
	const int	maxDepth_,
	const int	numSamples_,
	const int	numLabels_,
	const int	maxSpan_,
	const int	spaceSize_,
	const int	numFeatures_,
	const int	numThresholds_,
	boost::shared_ptr<std::vector<std::vector<rdf::Sample> > > samples_,
	const bool	createCUContext_
	) :
	createCUContext(createCUContext_)
{
	// Create CUDA context for the program
	if (createCUContext) {
		createCuContext(false);
	}

	// Check the validity of the inputs
	assert(numTrees_ > 0);
	int num_per_tree = (*samples_)[0].size();
	assert(num_per_tree > 0);

	for (int i = 0; i < numTrees_; i++) {
		assert((*samples_)[i].size() == num_per_tree);
	}

	// Initialize RDF forest parameters
	params.numTrees			= numTrees_;
	params.numImages		= numImages_;
	params.numPerTree		= numPerTree_;
	params.maxDepth			= maxDepth_;
	params.numSamples		= numSamples_;
	params.numLabels		= numLabels_ + 1;
	params.maxSpan			= maxSpan_;
	params.spaceSize		= spaceSize_;
	params.numFeatures		= numFeatures_;
	params.numThresholds	= numThresholds_;
	params.sample_per_tree	= (*samples_)[0].size();

	// Determine the width and height of the depth images
	width		= (*samples_)[0][0].getDepth().cols;
	height		= (*samples_)[0][0].getDepth().rows;
	depth_num	= numPerTree_;

	// Prepare for depth and other information in host
	int host_DP_Info[DPInfoCount] = { depth_num, width, height, width * height };
	int host_UT_Info[UTInfoCount] = { params.sample_per_tree * numFeatures_ };

	// Upload information to device constant memory
	const_DP_Info_upload(host_DP_Info);
	const_UT_Info_upload(host_UT_Info);

	// Copy parameters into constant memory
	const_CU_Params_upload(&params);

	// Initialize the forest to be trained
	forest = new rdf::Forest();

	// Initialize the log space for feature generation
	logSpace.initialize(maxSpan_, spaceSize_);

	// Initialize device memory
	cu_initialize();
}

CUDA_TRAIN::~CUDA_TRAIN()
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

void CUDA_TRAIN::cu_deviceMemFree()
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
}

void CUDA_TRAIN::cu_hostMemFree()
{
	// clean up CPU memory
	delete[] parentStatistics;

	host_labels.clear();
	host_sapID.clear();
	host_response_array.clear();
}

bool CUDA_TRAIN::shouldTerminate(float maxGain, int recurseDepth)
{
	return (maxGain < Thres || recurseDepth >= params.maxDepth);
}

// Compute entropy on the give statistics
float CUDA_TRAIN::entropy_compute(unsigned int* stat, int boundary)
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

void CUDA_TRAIN::cu_initialize()
{
	const int numFeatures   = params.numFeatures;
	const int samplePerTree = params.sample_per_tree;

	// Constant GPU Memory
	const int thresh_num_size	= numFeatures * sizeof(int);
	const int feature_size		= numFeatures * sizeof(float4);
	const int thresh_size		= (params.numThresholds + 1) * numFeatures * sizeof(float);
	const int parstat_size		= params.numLabels * (params.numThresholds + 1) * numFeatures * sizeof(unsigned int);
	
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

void CUDA_TRAIN::cu_reset()
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

void CUDA_TRAIN::compute_responses(std::vector<rdf::Sample>& samples_per_tree)
{
	// Generate a set of features for the current tree
	float4* host_features = new float4[params.numFeatures];

	for (int i = 0; i < params.numFeatures; i++) {
		rdf::Feature feat(logSpace);
		host_features[i].x = feat.getX();
		host_features[i].y = feat.getY();
		host_features[i].z = feat.getXX();
		host_features[i].w = feat.getYY();
	}

	// Copy features into device memory
	const int feature_size = params.numFeatures * sizeof(float4);
	cudaMemcpy(cu_features, host_features, feature_size, cudaMemcpyHostToDevice);

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
	delete[] host_features;

	copied_indicator.clear();
}

rdf::Forest* CUDA_TRAIN::cu_startTrain(boost::shared_ptr<std::vector<std::vector<rdf::Sample>>> samples_)
{
	// Start training the forest
	for (int t = 0; t < params.numTrees; t++) {

		// Timing each tree
		clock_t tree_start = clock();

		rdf::Tree tree(params.maxDepth);
		cu_reset();
		compute_responses((*samples_)[t]);
		cu_train(tree.getNodes());
		cudaDeviceSynchronize();

		clock_t tree_end = clock();
		float tree_time = (float)(tree_end - tree_start) / CLOCKS_PER_SEC;
		printf("Tree %d complete: %f seconds\n", t, tree_time);

		// Add tree to the forest
		forest->addTree(tree);
	}

	return forest;
}

void CUDA_TRAIN::cu_train(std::vector<rdf::Node>& nodes)
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

void CUDA_TRAIN::cu_trainLevel(
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
	int		size_thresh		 = (params.numThresholds + 1) * params.numFeatures * sizeof(float);
	int		size_parstat	 = params.numLabels * (params.numThresholds + 1) * params.numFeatures * sizeof(unsigned int);

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
		int blkSet_compute_gain = (int)ceil(params.numFeatures * (params.numThresholds + 1) * 1.0 / default_block_X);
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
		thrust::device_vector<float> thrust_gain(cu_gain, cu_gain + (params.numThresholds + 1) * params.numFeatures);
		thrust::device_vector<float>::iterator iter = thrust::max_element(thrust_gain.begin(), thrust_gain.end());
		float		 max_gain			= *iter;
		unsigned int position			= iter - thrust_gain.begin();
		int			 best_feature_idx	= position / (params.numThresholds + 1);

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

int CUDA_TRAIN::getWidth()
{
	return width;
}

int CUDA_TRAIN::getHeight()
{
	return height;
}

int CUDA_TRAIN::getNumTrees()
{
	return params.numTrees;
}

int CUDA_TRAIN::getNumLabels()
{
	return params.numLabels;
}

int CUDA_TRAIN::getMaxDepth()
{
	return params.maxDepth;
}

// ================================================================================================================
// ============================================= INFERENCE ========================================================
// ================================================================================================================

#define MAX_PARALLEL_IMG 4

// ================== Inference Kernels with Forest in shared memory ==================

CUDA_INFER::CUDA_INFER(
	const int							parallel_proc_,
	const int							depth_width_,
	const int							depth_height_,
	const int							numTrees_,
	const int							numLabels_,
	const int							maxDepth_,
	const float							minProb_,
	const bool							createCUContext_,
	const bool							realTime_,
	const std::string					in_directory_,
	const std::string					out_directory_
	) :
	parallel_proc	(parallel_proc_),
	depth_width		(depth_width_),
	depth_height	(depth_height_),
	numTrees		(numTrees_),
	numLabels		(numLabels_ + 1),
	maxDepth		(maxDepth_),
	minProb			(minProb_),
	in_directory	(in_directory_),
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

	// Determine if the given number of labels matches the program setting
	if (numLabels != NODE_NUM_LABELS) {
		printf("Error: the number of labels is fixed ...\n");
		exit(-1);
	}

	// Read in forest from file
	readForest();

	// Upload tree information to constant memory
	cu_upload_TreeInfo();

	// Determine if to put forest in shared memory
	const int node_num = cu_ifInShared();

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
	for (int i = 0; i < forest_CU.size(); i++) {
		for (int j = 0; j < forest_CU[i].size(); j++) {
			host_forest[cntr++] = forest_CU[i][j];
		}
	}
	assert(cntr == node_num);

	// Synchronously copy forest into device memory
	cudaMemcpy(device_forest, host_forest, forest_size, cudaMemcpyHostToDevice);

	// Decide if to output results
	writeOut = (out_directory == "") ? false : true;
	if (writeOut) {
		out_path = boost::filesystem::path(out_directory);
		if (!boost::filesystem::exists(out_path)) {
			boost::filesystem::create_directory(out_path);
		}
	}

	// Free host forest immediately for performance purpose
	free(host_forest);
}

CUDA_INFER::CUDA_INFER(
	const int							parallel_proc_,
	CUDA_TRAIN&							cuda_train,
	const float							minProb_,
	const bool							createCUContext_,
	const bool							realTime_,
	const std::string					in_directory_,
	const std::string					out_directory_
	) :
	parallel_proc	(parallel_proc_),
	depth_width		(cuda_train.getWidth()),
	depth_height	(cuda_train.getHeight()),
	numTrees		(cuda_train.getNumTrees()),
	numLabels		(cuda_train.getNumLabels()),
	maxDepth		(cuda_train.getMaxDepth()),
	minProb			(minProb_),
	in_directory	(in_directory_),
	createCUContext	(createCUContext_),
	depthArray_size	(parallel_proc_ * depth_width * depth_height * sizeof(float)),
	rgbArray_size	(parallel_proc_ * depth_width * depth_height * sizeof(int3)),
	out_directory	(out_directory_),
	realTime		(realTime_),
	firstFrame		(true)
{
	// Create CUDA context for the program
	if (createCUContext) {
		createCuContext(false);
	}

	// Determine if the given number of labels matches the program setting
	if (numLabels != NODE_NUM_LABELS) {
		printf("Error: the number of labels is fixed ...\n");
		exit(-1);
	}

	// Read in forest from file
	readForest();

	// Upload tree information to constant memory
	cu_upload_TreeInfo();

	// Determine if to put forest in shared memory
	const int node_num = cu_ifInShared();

	// Upload depth information to constant memory
	upload_DepthInfo_Inf(depth_width, depth_height);

	// Calculate the kernel launch parameters
	blk_Inference = (int)ceil(depth_width * depth_height * parallel_proc_ * 1.0f / default_block_X);

	frame_cntr = 0;								// Set the frame counter to zero

	// Allocate device memory
	gpuErrchk(cudaMalloc(&device_depthArray1, depthArray_size));
	gpuErrchk(cudaMalloc(&device_rgbArray1, rgbArray_size));
	gpuErrchk(cudaMalloc(&device_forest, forest_size));

	// Initialize device memory
	gpuErrchk(cudaMemset(device_depthArray1, 0, depthArray_size));
	gpuErrchk(cudaMemset(device_rgbArray1, 0, rgbArray_size));

	// Allocate the host output array
	host_rgbArray1 = new int3[depth_width * depth_height];

	// Initialize host array
	memset(host_rgbArray1, 0, depth_width * depth_height * sizeof(int3));

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
		host_rgbArray2 = new int3[depth_width * depth_height];

		// Initialize host array
		memset(host_rgbArray2, 0, depth_width * depth_height * sizeof(int3));
	}

	// Prepare host forest
	host_forest = new Node_CU[node_num];

	// Copy forest into host array
	int cntr = 0;
	for (int i = 0; i < forest_CU.size(); i++) {
		for (int j = 0; j < forest_CU[i].size(); j++) {
			host_forest[cntr++] = forest_CU[i][j];
		}
	}
	assert(cntr == node_num);

	// Synchronously copy forest into device memory
	cudaMemcpy(device_forest, host_forest, forest_size, cudaMemcpyHostToDevice);

	// Decide if to output results
	writeOut = (out_directory == "") ? false : true;
	if (writeOut) {
		out_path = boost::filesystem::path(out_directory);
		if (!boost::filesystem::exists(out_path)) {
			boost::filesystem::create_directory(out_path);
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

void CUDA_INFER::readForest()
{
	in_path = boost::filesystem::path(in_directory);
	boost::filesystem::ifstream in(in_path);

	forest_CU.resize(numTrees);
	const int treeSize = (1 << maxDepth) - 1;

	// Declare to hold input values
	int				intValue;
	unsigned int	uintValue;
	float			floatValue;

	for (size_t i = 0; i < numTrees; i++) {
		int node_counter = 0;
		std::vector<Node_CU>& nodes_CU = forest_CU[i];
		std::vector<unsigned int> parent_pointer(treeSize, -1);

		for (size_t j = 0; j < treeSize; j++) {
			in >> intValue;
			int label = intValue;

			// Split node
			if (label == 1) {
				in >> uintValue;
				unsigned int idx = uintValue;

				// Indicate the true position in the array
				parent_pointer[idx] = node_counter++;

				// Initialize a split CUDA node and push back
				Node_CU node_operation;
				node_operation.isSplit = 1;
				in >> node_operation.feature.x;			// x of feature
				in >> node_operation.feature.y;			// y of feature
				node_operation.feature.z = 0.0;			// No value set
				node_operation.feature.w = 0.0;			// No value set
				in >> node_operation.threshold;			// Threshold
				for (unsigned int k = 0; k < NODE_NUM_LABELS; k++) {
					node_operation.aggregator[k] = 0;
				}
				node_operation.leftChild = -1;
				node_operation.rightChild = -1;

				// Push current node into the vector
				nodes_CU.push_back(node_operation);

				if (idx == 0) {
					continue;
				}

				// Set left and right child indicator of parent node
				(idx % 2 != 0) ?
					nodes_CU[parent_pointer[(idx - 1) / 2]].leftChild  = node_counter - 1 :
					nodes_CU[parent_pointer[(idx - 2) / 2]].rightChild = node_counter - 1;
			}

			// Leaf node
			else if (label == 0) {
				in >> uintValue;
				unsigned int idx = uintValue;

				// Indicate the true position in the array
				parent_pointer[idx] = node_counter++;

				// Initialize a leaf CUDA node and push back
				Node_CU node_operation;
				node_operation.isSplit = 0;
				node_operation.feature.x = 0.0;
				node_operation.feature.y = 0.0;
				node_operation.feature.z = 0.0;
				node_operation.feature.w = 0.0;
				node_operation.threshold = 0.0;

				in >> uintValue;
				for (unsigned int k = 0; k < NODE_NUM_LABELS; k++) {
					in >> node_operation.aggregator[k];
				}

				node_operation.leftChild = -1;
				node_operation.rightChild = -1;

				// Push current node into the vector
				nodes_CU.push_back(node_operation);

				if (idx == 0) {
					continue;
				}

				// Set left and right child indicator of parent node
				(idx % 2 != 0) ?
					nodes_CU[parent_pointer[(idx - 1) / 2]].leftChild = node_counter - 1 :
					nodes_CU[parent_pointer[(idx - 2) / 2]].rightChild = node_counter - 1;
			}
		}
	}

	in.close();
}

void CUDA_INFER::upload_DepthInfo_Inf(
	int width_,
	int height_
	)
{
	int host_DepthInfo_Inf[DepthInfoCount_Inf] = { width_, height_, width_ * height_ };
	const_DepthInfo_Inf_upload(host_DepthInfo_Inf);
}

void CUDA_INFER::cu_upload_TreeInfo()
{
	// Assuming forest contains correct number of trees, MIGHT BE RISKY
	int numTrees_	= forest_CU.size();

	// Assuming 0 for labelIndex_, MIGHT BE RISKY
	int	labelIndex_ = 0;

	// Determine if the input number of trees exceed the partition container limit
	if (numTrees_ > TREE_CONTAINER_SIZE) {
		printf("Error: size of tree partition container is fixed ...\n");
		exit(-1);
	}

	const_numTrees_Inf_upload(&numTrees_);
	const_numLabels_Inf_upload(&numLabels);
	const_maxDepth_Inf_upload(&maxDepth);
	const_labelIndex_Inf_upload(&labelIndex_);
	const_minProb_Inf_upload(&minProb);

	// Upload trees partition information to constant memory
	int host_total_nodeNum = 0;
	int host_partitionInfo[TREE_CONTAINER_SIZE + 1] = { 0 };
	for (int i = 0; i < numTrees_; i++) {
		host_partitionInfo[i + 1] = forest_CU[i].size() + host_partitionInfo[i];
		host_total_nodeNum += forest_CU[i].size();
	}
	const_totalNode_Inf_upload(&host_total_nodeNum);
	const_treePartition_Inf_upload(host_partitionInfo);
}

int CUDA_INFER::cu_ifInShared()
{
	// Default value
	forestInSharedMem = false;

	// Calculate the total number of bytes of the forest
	int node_num = 0;
	for (int i = 0; i < forest_CU.size(); i++) {
		node_num += forest_CU[i].size();
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
	cudaMemcpy(host_rgbArray1, device_rgbArray1, rgbArray_size, cudaMemcpyDeviceToHost);

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
		out_rgb = out_path / (fmt_result_rgb % std::to_string(frame_cntr - 3)).str();
	}

	// For batch outputs, three frames lag but better performance
	else {

		out_rgb = out_path / (fmt_result_rgb % std::to_string(frame_cntr)).str();
	}

	// Write out results
	cv::imwrite(out_rgb.string(), result);
}

void CUDA_INFER::writeResult(const std::vector<cv::Mat_<cv::Vec3i>>& result)
{
	for (int i = 0; i < 3; i++) {
		// Assemble names
		boost::format fmt_result_rgb("%s_result_rgb.png");
		boost::filesystem::path out_rgb = out_path / (fmt_result_rgb % std::to_string(frame_cntr - (3 - i))).str();

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