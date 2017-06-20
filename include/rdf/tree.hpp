#ifndef TREE_HPP
#define TREE_HPP

#include <vector>
#include <rdf/node.hpp>
#include <rdf/sample.hpp>
#include <rdf/target.hpp>

namespace rdf {

    class Tree {
    public:
        Tree(const int& maxDepth);
        ~Tree() {}

        void initialize(const int& maxDepth);
        std::vector<rdf::Node>& getNodes();
        rdf::Node& getNode(int idx);
        bool infer(rdf::Target& target, const rdf::Sample& sample, const int& numLabels);    

    protected:
        std::vector<rdf::Node> nodes_;
        int decisionLevels_;
    };    

    typedef boost::shared_ptr<Tree> TreePtr;

} // namespace rdf


#endif
