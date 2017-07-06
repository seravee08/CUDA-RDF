#ifndef FOREST_HPP
#define FOREST_HPP

#include <vector>
#include <boost/filesystem.hpp>
#include <rdf/tree.hpp>
#include <rdf/target.hpp>

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
			const boost::filesystem::path&	path,
			const int&						numLabels
			);
	protected:
		std::vector<rdf::Tree> trees_;
		int numTrees_;

	};

	typedef boost::shared_ptr<Forest> ForestPtr;
} //namespace rdf

#endif