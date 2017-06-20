/*
 * fileutil.h
 *
 *  Created on: Oct 24, 2016
 *      Author: roy_shilkrot
 */

#ifndef INCLUDE_UTIL_FILEUTIL_H_
#define INCLUDE_UTIL_FILEUTIL_H_

/**
 * TODO: doc...
 * @param dir
 * @param regex
 * @param files
 */
void addPaths(boost::filesystem::path dir, std::string regex, std::vector<boost::filesystem::path>& files);

#endif /* INCLUDE_UTIL_FILEUTIL_H_ */
