#ifndef RDF_HPP
#define RDF_HPP

#include <rdf/node.hpp>
#include <rdf/tree.hpp>
#include <rdf/forest.hpp>
#include <rdf/sample.hpp>
#include <rdf/aggregator.hpp>
#include <rdf/feature.hpp>
#include <rdf/logSpace.hpp>

namespace rdf {

    class RDF {
    public:
        RDF(boost::shared_ptr<std::vector<std::vector<rdf::Sample> > > samples);
        ~RDF() {}

		rdf::Tree trainTree(const int& idx, const int& maxDepth);
        rdf::ForestPtr trainForest(const int& maxDepth);
        void reset(const int& idx);
        void initialize(const int& maxSpan, 
                        const int& spaceSize, 
                        const int& numLabels,
                        const int& numFeatures,
                        const int& numThresholds,
                        const int& numTrees,
                        const int& maxDecisionLevels);

		void initialize_cu(const int& maxSpan,
						   const int& spaceSize,
					       const int& numLabels,
						   const int& numFeatures,
						   const int& numThresholds,
						   const int& numTrees,
						   const int& maxDecisionLevels,
						   const int& numSamples,
						   const int& numImages,
						   const int& numPerTree);

        void trainNodesRecurse(std::vector<rdf::Node>& nodes,
                               const unsigned int& idx_node,
                               const unsigned int& idx_dataS,
                               const unsigned int& idx_dataE,
                               const int& recurseDepth,
                               const int& idx_tree,
							   rdf::Feature* const_Features
							   );    

        int chooseCandidateThresholds(const unsigned int& idx_dataS,
                                      const unsigned int& idx_dataE,
                                      const float* response,
                                      std::vector<float>& thresholds);

        double computeGain() const;    

        bool shouldTerminate(double maxGain, int recurseDepth);    

        /**
         * TODO: doc... what does it do? what is the outcome? what does it assume on the input?
         *
         * @param keys
         * @param values
         * @param idxS
         * @param idxE
         * @param threshold
         * @return
         */
        unsigned int partition(std::vector<float>& keys,
                               std::vector<unsigned int>& values,
                               const unsigned int idxS,
                               const unsigned int idxE,
                               const float threshold);
    private:
        rdf::Aggregator                                              parentStatistics_;
        rdf::Aggregator                                              leftchildStatistics_;
        rdf::Aggregator                                              rightchildStatistics_;
        std::vector<rdf::Aggregator>                                 partitionStatistics_;
        std::vector<float>                                           response_;
        boost::shared_ptr<std::vector<std::vector<rdf::Sample>>>     samples_;
        std::vector<unsigned int>                                    indices_;
        rdf::LogSpace                                                space_;

        int numOfCandidateThresholdsPerFeature_;
        int numOfCandidateFeatures_;
        int numOfTrees_;
        int maxDecisionLevels_;
        int numOfLabels_;
		int numSamples_;
		int spaceSize_;
		int numImages_;
		int numPerTree_;
		int maxSpan_;
    };

    typedef boost::shared_ptr<RDF> RDFPtr;

} // namespace rdf

#endif
