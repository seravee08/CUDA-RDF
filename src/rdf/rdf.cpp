#include <npp.h>
#include <rdf/rdf.hpp>
#include <boost/make_shared.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

#define Thres 0.01

using namespace std;

namespace rdf{

	RDF::RDF(boost::shared_ptr<std::vector<std::vector<rdf::Sample> > > samples){
		samples_ = samples;
	}

	void RDF::initialize(const int& maxSpan,
		const int& spaceSize,
		const int& numLabels,
		const int& numFeatures,
		const int& numThresholds,
		const int& numTrees,
		const int& maxDecisionLevels) {
		double max = maxSpan;
		space_.initialize(max, spaceSize);

		numOfLabels_ = numLabels;
		parentStatistics_.initialize(numOfLabels_);
		leftchildStatistics_.initialize(numOfLabels_);
		rightchildStatistics_.initialize(numOfLabels_);

		numOfTrees_ = numTrees;
		numOfCandidateFeatures_ = numFeatures;
		numOfCandidateThresholdsPerFeature_ = numThresholds;
		maxDecisionLevels_ = maxDecisionLevels;

		partitionStatistics_.resize(numOfCandidateThresholdsPerFeature_ + 1);
		for (int b = 0; b < numOfCandidateThresholdsPerFeature_ + 1; b++) {
			partitionStatistics_[b].initialize(numOfLabels_);
		}
	}

	void RDF::initialize_cu(const int& maxSpan,
		const int& spaceSize,
		const int& numLabels,
		const int& numFeatures,
		const int& numThresholds,
		const int& numTrees,
		const int& maxDecisionLevels,
		const int& numSamples,
		const int& numImages,
		const int& numPerTree) {
		double max = maxSpan;
		space_.initialize(max, spaceSize);

		maxSpan_ = maxSpan;
		spaceSize_ = spaceSize;
		numImages_ = numImages;
		numSamples_ = numSamples;
		numPerTree_ = numPerTree;
		numOfLabels_ = numLabels;
		parentStatistics_.initialize(numOfLabels_);
		leftchildStatistics_.initialize(numOfLabels_);
		rightchildStatistics_.initialize(numOfLabels_);

		numOfTrees_ = numTrees;
		numOfCandidateFeatures_ = numFeatures;
		numOfCandidateThresholdsPerFeature_ = numThresholds;
		maxDecisionLevels_ = maxDecisionLevels;

		partitionStatistics_.resize(numOfCandidateThresholdsPerFeature_ + 1);
		for (int b = 0; b < numOfCandidateThresholdsPerFeature_ + 1; b++) {
			partitionStatistics_[b].initialize(numOfLabels_);
		}
	}

	void RDF::reset(const int& idx) {
		indices_.clear();
		response_.clear();
		partitionStatistics_.clear();

		indices_.resize((*samples_)[idx].size());
		for (unsigned int i = 0; i < (*samples_)[idx].size(); i++) {
			indices_[i] = i;
		}

		response_.resize((*samples_)[idx].size());
		partitionStatistics_.resize(numOfCandidateThresholdsPerFeature_ + 1);
		for (int b = 0; b < numOfCandidateThresholdsPerFeature_ + 1; b++) {
			partitionStatistics_[b].initialize(numOfLabels_);
		}
	}

	int RDF::chooseCandidateThresholds(const unsigned int& idx_dataS,
		const unsigned int& idx_dataE,
		const float* response,
		std::vector<float>& thresholds){

		thresholds.resize(numOfCandidateThresholdsPerFeature_ + 1);
		std::vector<float>& thr = thresholds;

		if (idx_dataE - idx_dataS <= 1)
			return 0;

		std::random_device rand;
		std::mt19937 gen(rand());
		std::uniform_int_distribution<> disI(idx_dataS, idx_dataE - 1);
		std::uniform_real_distribution<> disR(0, 1);

		int nThresholds;
		if (idx_dataE - idx_dataS > (unsigned int)numOfCandidateThresholdsPerFeature_){
			nThresholds = numOfCandidateThresholdsPerFeature_;
			for (int i = 0; i < nThresholds + 1; i++)
				thr[i] = response[disI(gen)];
		}
		else{
			nThresholds = idx_dataE - idx_dataS - 1;
			std::copy(&response[idx_dataS], &response[idx_dataE], thr.begin());
		}

		std::sort(thr.begin(), thr.end());

		if (thr[0] == thr[nThresholds])
			return 0;

		for (int i = 0; i < nThresholds; i++){
			thresholds[i] = thr[i] + (float)(disR(gen) * (thr[i + 1] - thr[i]));
		}

		return nThresholds;
	}

	double RDF::computeGain() const {
		double entropyAll = parentStatistics_.entropy();

		unsigned int nSamples = leftchildStatistics_.sampleCount() + rightchildStatistics_.sampleCount();

		if (nSamples <= 1)
			return 0.0;

		double entropyPart = (leftchildStatistics_.sampleCount() * leftchildStatistics_.entropy() + rightchildStatistics_.sampleCount() * rightchildStatistics_.entropy()) / nSamples;

		return entropyAll - entropyPart;
	}

	bool RDF::shouldTerminate(double maxGain, int recurseDepth) {
		return (maxGain < Thres || recurseDepth >= maxDecisionLevels_);
	}


	unsigned int RDF::partition(std::vector<float>&        keys,
		std::vector<unsigned int>& values,
		const unsigned int         idxS,
		const unsigned int         idxE,
		const float                threshold) {

		int i = (int)(idxS);       // index of first element
		int j = (int)(idxE - 1);   // index of last element

		while (i != j){
			if (keys[i] >= threshold){
				// Swap keys[i] with keys[j]
				float key = keys[i];
				unsigned int value = values[i];

				keys[i] = keys[j];
				values[i] = values[j];

				keys[j] = key;
				values[j] = value;

				j--;
			}
			else {
				i++;
			}
		}

		return keys[i] >= threshold ? i : i + 1;
	}


	void RDF::trainNodesRecurse(std::vector<rdf::Node>& nodes,
		const unsigned int& idx_node,
		const unsigned int& idx_dataS,
		const unsigned int& idx_dataE,
		const int& recurseDepth,
		const int& idx_tree,
		rdf::Feature* const_Features
		) {

		parentStatistics_.clear();

		//aggregate statistics over the samples at the parent node
		for (unsigned int i = idx_dataS; i < idx_dataE; i++){
			parentStatistics_.aggregate((*samples_)[idx_tree][indices_[i]]);
		}
		if (idx_node >= nodes.size() / 2){
			nodes[idx_node].initializeLeaf(parentStatistics_, idx_node);
			return;
		}

		double maxGain = 0.0;
		rdf::Feature bestFeature;
		float bestThreshold = 0.0f;

		//iterate over all candidate features
		std::vector<float> thresholds;
		for (int f = 0; f < numOfCandidateFeatures_; f++){
			rdf::Feature feature = const_Features[f];

			for (int b = 0; b < numOfCandidateThresholdsPerFeature_ + 1; b++)
				partitionStatistics_[b].clear();

			//compute response for each sample at this node
			for (unsigned int i = idx_dataS; i < idx_dataE; i++) {
				response_[i] = feature.getResponse((*samples_)[idx_tree][indices_[i]]);
			}

			int nThresholds;
			if ((nThresholds = chooseCandidateThresholds(idx_dataS, idx_dataE, &response_[0], thresholds)) == 0)
				continue;

			for (unsigned int i = idx_dataS; i < idx_dataE; i++){
				int b = 0;
				while (b < nThresholds && response_[i] >= thresholds[b])
					b++;

				partitionStatistics_[b].aggregate((*samples_)[idx_tree][indices_[i]]);
			}

			for (int t = 0; t < nThresholds; t++){
				leftchildStatistics_.clear();
				rightchildStatistics_.clear();
				for (int p = 0; p < nThresholds + 1; p++){
					if (p <= t)
						leftchildStatistics_.aggregate(partitionStatistics_[p]);
					else
						rightchildStatistics_.aggregate(partitionStatistics_[p]);
				}

				//compute gain
				double gain = computeGain();
				if (gain >= maxGain){
					maxGain = gain;
					bestFeature = feature;
					bestThreshold = thresholds[t];
				}
			}
		}

		nodes[idx_node].setIdx(idx_node);

		if (maxGain == 0.0){
			nodes[idx_node].initializeLeaf(parentStatistics_, idx_node);
			return;
		}

		////////////////
		leftchildStatistics_.clear();
		rightchildStatistics_.clear();

		for (unsigned int i = idx_dataS; i < idx_dataE; i++){
			response_[i] = bestFeature.getResponse((*samples_)[idx_tree][indices_[i]]);
			if (response_[i] < bestThreshold)
				leftchildStatistics_.aggregate((*samples_)[idx_tree][indices_[i]]);
			else
				rightchildStatistics_.aggregate((*samples_)[idx_tree][indices_[i]]);
		}

		if (shouldTerminate(maxGain, recurseDepth)){
			nodes[idx_node].initializeLeaf(parentStatistics_, idx_node);
			return;
		}

		//otherwise, split node
		nodes[idx_node].initializeSplit(bestFeature, bestThreshold, idx_node);

		unsigned int idx = partition(response_, indices_, idx_dataS, idx_dataE, bestThreshold);
		trainNodesRecurse(nodes, idx_node * 2 + 1, idx_dataS, idx, recurseDepth + 1, idx_tree, const_Features);
		trainNodesRecurse(nodes, idx_node * 2 + 2, idx, idx_dataE, recurseDepth + 1, idx_tree, const_Features);
	}


	rdf::Tree RDF::trainTree(const int& idx, const int& maxDepth){

		clock_t start = clock();
		rdf::Tree tree(maxDepth);

		// ===== CPU Version =====
		rdf::Feature* const_Features = new rdf::Feature[numOfCandidateFeatures_];
		for (int i = 0; i < numOfCandidateFeatures_; i++) {
			const_Features[i] = rdf::Feature(space_);
		}
		trainNodesRecurse(tree.getNodes(), 0, 0, (*samples_)[idx].size(), 0, idx, const_Features);

		clock_t end = clock();
		float time = (float)(end - start) / CLOCKS_PER_SEC;
		printf("Time: %f\n", time);

		return tree;
	}

	rdf::ForestPtr RDF::trainForest(const int& maxDepth){

		rdf::ForestPtr forest = boost::make_shared<rdf::Forest>();
		for (int t = 0; t < numOfTrees_; t++) {
			reset(t);
			rdf::Tree tree = trainTree(t, maxDepth);
			forest->addTree(tree);

			std::cout << "Tree " << t + 1 << " complete!" << std::endl;
			auto now = std::chrono::system_clock::now();
			auto now_c = std::chrono::system_clock::to_time_t(now);
			//std::cout << std::put_time(std::localtime(&now_c), "%c") << std::endl;
		}

		return forest;
	}
}
