#include <rdf/tree.hpp>

namespace rdf{
	Tree::Tree(const int& maxDepth) {
		decisionLevels_ = maxDepth;
		nodes_.resize((1 << decisionLevels_) - 1);
	}
	
	void Tree::initialize(const int& maxDepth){
		decisionLevels_ = maxDepth;
		nodes_.resize((1 << decisionLevels_) - 1);
	}

	std::vector<rdf::Node>& Tree::getNodes(){
		return nodes_;
	}

	rdf::Node& Tree::getNode(int idx){
		return nodes_[idx];
	}

	bool Tree::infer(rdf::Target& target, const rdf::Sample& sample, const int& numLabels) {

		unsigned int idx = 0;
		while (nodes_[idx].isSplit()) {
			const rdf::Feature& feature = nodes_[idx].getFeature();
			const float threshold = nodes_[idx].getThreshold();
			const float response = feature.getResponse(sample);

			if (response < threshold) {
				idx = idx * 2 + 1;
			} else {
				idx = idx * 2 + 2;
			}
		}

		if (!nodes_[idx].isLeaf()) {
			std::cerr << "invalid tree!" << std::endl;
			return false;
		}

		const rdf::Aggregator& aggregator = nodes_[idx].getAggregator();
		for(int i = 0; i < numLabels; i++) {
			target.Prob()[i] = (float)aggregator.samplePerBin(i) / (float)aggregator.sampleCount();
		}

		return true;
	}
}
