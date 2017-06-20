/*
 * fileutil.cpp
 *
 *  Created on: Oct 24, 2016
 *      Author: roy_shilkrot
 */

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>


void addPaths(boost::filesystem::path dir, std::string regex, std::vector<boost::filesystem::path>& files) {
    boost::regex expression(regex);
    for (boost::filesystem::directory_iterator end, current(dir); current != end; ++current){
        boost::filesystem::path cp = current->path();
        boost::cmatch what;
        std::string cpStr = cp.filename().string();

        if (boost::regex_match(cpStr.c_str(), what, expression)) {
            files.push_back(cp);
        }
    }
}




