#include <rdf/forest.hpp>

namespace rdf{
	
	Forest::Forest() {
		numTrees_ = 0;
	}

    Forest::Forest(const int& numTrees, const int& maxDepth) {
        numTrees_ = 0;
        for(int i = 0; i < numTrees; i++) {
            Tree* tree_ptr = new Tree(maxDepth);
            addTree(*tree_ptr);
        }
    }

	void Forest::addTree(const rdf::Tree& tree) {
		trees_.push_back(tree);
		numTrees_ ++ ;
    }

    std::vector<rdf::Tree>& Forest::getTrees() {
        return trees_;
    }

    int Forest::NumTrees() {
        return numTrees_;
    }

    void Forest::save(const boost::filesystem::path& path) {
    	boost::filesystem::ofstream of(path);
    	for(int i = 0; i < numTrees_; i++) {

    		std::vector<rdf::Node> nodes = trees_[i].getNodes();
    		for(unsigned int j = 0; j < nodes.size(); j++) {

    			if(nodes[j].isSplit()) {
    				rdf::Feature feature = nodes[j].getFeature();
    				float threshold = nodes[j].getThreshold();

    				//output
                    of << 1 << " ";                  //node label, 1 stands for split node
    				of << nodes[j].getIdx() << " ";  //node index
    				of << feature.getX() << " " << feature.getY() << " ";
    				                                 //split feature
    				of << threshold << std::endl;         //feature threshold
    			}

    			else if(nodes[j].isLeaf()) {
    				rdf::Aggregator trainingStatistics = nodes[j].getAggregator();

    				//output
                    of << 0 << " ";                  //node label, 0 stands for leaf node
    				of << nodes[j].getIdx() << " ";  //node index
    				of << trainingStatistics.sampleCount() << " ";
    				                                 //number of samples
    				for(int k = 0; k < trainingStatistics.binCount(); k++)
    					of << trainingStatistics.samplePerBin(k) << " ";
    				                                 //bin values
                    of << std::endl;
    			}

    			else {
                    of << -1 << std::endl;
                }
    		}
    	}
    }

    void Forest::readForest(
		const boost::filesystem::path& path,
		const int& numLabels,
		std::vector<std::vector<Node_CU> >& forest_CU
		) 
	{
        boost::filesystem::ifstream in(path);
        
        for(int i = 0; i < numTrees_; i++){
            std::vector<rdf::Node>& nodes = trees_[i].getNodes();
            int intValue;
            unsigned int uintValue;
            float floatValue;
			
#ifdef USE_GPU_INFERENCE
			int node_counter = 0;
			std::vector<Node_CU>& nodes_CU = forest_CU[i];
			std::vector<unsigned int> parent_pointer(nodes.size(), -1);
#endif

            for(unsigned int j = 0; j < nodes.size(); j++){
                in >> intValue;
                int label = intValue;

                //split node
                if(label == 1) {
                    in >> uintValue;
                    unsigned int idx = uintValue;
                    nodes[idx].setIdx(idx);

                    rdf::Feature feature;
                    in >> floatValue;
                    feature.setX(floatValue);
                    in >> floatValue;
                    feature.setY(floatValue);

                    in >> floatValue;
                    nodes[idx].initializeSplit(feature, floatValue, idx);

#ifdef USE_GPU_INFERENCE
					// Indicate the true position in the array
					parent_pointer[idx] = node_counter++;

					// Initialize a split CUDA node and push back
					Node_CU node_operation;
					node_operation.isSplit		= 1;
					node_operation.feature.x	= feature.getX();	// x of feature
					node_operation.feature.y	= feature.getY();	// y of feature
					node_operation.feature.z	= 0.0;				// No value set
					node_operation.feature.w	= 0.0;				// No value set
					node_operation.threshold	= floatValue;		// Threshold
					for (unsigned int k = 0; k < NODE_NUM_LABELS; k++) {
						node_operation.aggregator[k] = 0;
					}
					node_operation.leftChild	= -1;
					node_operation.rightChild	= -1;

					// Push current node into the vector
					nodes_CU.push_back(node_operation);

					if (idx == 0) {
						continue;
					}

					// Set left and right child indicator of parent node
					(idx % 2 != 0) ?
						nodes_CU[parent_pointer[(idx - 1) / 2]].leftChild	= node_counter - 1:
						nodes_CU[parent_pointer[(idx - 2) / 2]].rightChild	= node_counter - 1;
#endif

                }
                //leaf node
				else if (label == 0) {
					in >> uintValue;
					unsigned int idx = uintValue;
					nodes[idx].setIdx(idx);

					rdf::Aggregator trainingStatistics(numLabels);
					in >> uintValue;
					trainingStatistics.setSampleCount(uintValue);
					for (int k = 0; k < trainingStatistics.binCount(); k++) {
						in >> uintValue;
						trainingStatistics.setBin(k, uintValue);
					}
					nodes[idx].initializeLeaf(trainingStatistics, idx);

#ifdef USE_GPU_INFERENCE
					// Indicate the true position in the array
					parent_pointer[idx] = node_counter++;

					// Initialize a leaf CUDA node and push back
					Node_CU node_operation;
					node_operation.isSplit		= 0;
					node_operation.feature.x	= 0.0;
					node_operation.feature.y	= 0.0;
					node_operation.feature.z	= 0.0;
					node_operation.feature.w	= 0.0;
					node_operation.threshold	= 0.0;
					for (unsigned int k = 0; k < NODE_NUM_LABELS; k++) {
						node_operation.aggregator[k] = trainingStatistics.samplePerBin(k);
					}
					node_operation.leftChild	= -1;
					node_operation.rightChild	= -1;

					// Push current node into the vector
					nodes_CU.push_back(node_operation);

					if (idx == 0) {
						continue;
					}

					// Set left and right child indicator of parent node
					(idx % 2 != 0) ?
						nodes_CU[parent_pointer[(idx - 1) / 2]].leftChild	= node_counter - 1 :
						nodes_CU[parent_pointer[(idx - 2) / 2]].rightChild	= node_counter - 1;
#endif
				}
            }
        }
    }

    void Forest::inference(rdf::Target& result, const rdf::Sample& sample, const int& numLabels) {

        std::vector<rdf::Target> targets;
        targets.resize(numTrees_); 
        for(int i = 0; i < numTrees_; i++) {
            targets[i].initialize(numLabels);
            trees_[i].infer(targets[i], sample, numLabels);
        }
        std::vector<float>& prob = result.Prob();
        for(int i = 0; i < numTrees_; i++) {
            for(int j = 0; j < numLabels; j++) {
                prob[j] += targets[i].Prob()[j];
            }
        }

        for(int i = 0; i < numLabels; i++) {
            prob[i] /= (float)numTrees_;
        }
    }

}