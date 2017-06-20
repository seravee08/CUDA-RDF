#ifndef NODE_HPP
#define NODE_HPP

#include <boost/shared_ptr.hpp>
#include <rdf/aggregator.hpp>
#include <rdf/feature.hpp>


namespace rdf{

	class Node{
	public:
		Node();
		~Node(){}
		void initializeLeaf(const Aggregator& trainingStatistics, const unsigned int& idx);
		void initializeSplit(Feature feature, const float& threshold, const unsigned int& idx);
		bool isLeaf();
		bool isSplit();
		void setIdx(unsigned int idx);
		unsigned int getIdx();
		rdf::Feature getFeature();
		rdf::Aggregator getAggregator();
		float getThreshold();
	private:
		bool isLeaf_;
		bool isSplit_;
		float threshold_;
		rdf::Aggregator trainingStatistics_;
		rdf::Feature feature_;
		unsigned int idx_;
	};

	typedef boost::shared_ptr<Node> NodePtr;

} // namespace rdf


#endif