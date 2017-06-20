#ifndef FOREST_HPP
#define FOREST_HPP

#include <vector>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <rdf/tree.hpp>
#include <rdf/aggregator.hpp>
#include <rdf/target.hpp>
#include <rdf/sample.hpp>

#include "../../build/tools/RDF_CU.cuh"

#define USE_GPU_INFERENCE		// Define to enable GPU inference

namespace rdf{
	class Forest{
	public:
		Forest();
		Forest(const int& numTrees, const int& maxDepth);
		~Forest(){}
		void addTree(const rdf::Tree& tree);
		void save(const boost::filesystem::path& path);
		std::vector<rdf::Tree>& getTrees();
		int NumTrees();
		void inference(rdf::Target& result, const rdf::Sample& sample, const int& numLabels);

		void readForest(
			const boost::filesystem::path&			path,
			const int&								numLabels,
			std::vector<std::vector<Node_CU> >&		forest_CU
			);
	protected:
		std::vector<rdf::Tree> trees_;
		int numTrees_;

	};

	typedef boost::shared_ptr<Forest> ForestPtr;
} //namespace rdf

#endif