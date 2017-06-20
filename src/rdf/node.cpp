#include <rdf/node.hpp>

namespace rdf{
	Node::Node(){
		isLeaf_ = false;
		isSplit_ = false;
	}

	void Node::initializeLeaf(const Aggregator& trainingStatistics, const unsigned int& idx){
		feature_ = rdf::Feature();
		threshold_ = 0.0f;
		isLeaf_ = true;
		isSplit_ = false;
		trainingStatistics_ = trainingStatistics.clone();
		idx_ = idx;
	}

	void Node::initializeSplit(Feature feature, const float& threshold, const unsigned int& idx){
		isLeaf_ = false;
		isSplit_ = true;
		feature.copyTo(feature_);
		threshold_ = threshold;
		idx_ = idx;
	}

	bool Node::isLeaf(){
		return (isLeaf_ && !isSplit_);
	}

	bool Node::isSplit(){
		return (isSplit_ && !isLeaf_);
	}

	void Node::setIdx(unsigned int idx){
		idx_ = idx;
	}

	unsigned int Node::getIdx(){
		return idx_;
	}

	rdf::Feature Node::getFeature(){
		return feature_;
	}

	rdf::Aggregator Node::getAggregator(){
		return trainingStatistics_;
	}

	float Node::getThreshold(){
		return threshold_;
	}
}