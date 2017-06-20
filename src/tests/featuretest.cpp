/*
 * featuretest.cpp
 *
 *  Created on: Oct 24, 2016
 *      Author: roy_shilkrot
 */


#include <rdf/feature.hpp>

#define BOOST_TEST_MODULE RDFTests
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(FeatureTest) {
    const cv::Mat_<float> depth(50, 50, 0.1f); //set all to .1
    depth(cv::Rect(0, 0, 25, 50)).setTo(0.5f); //set left half to .5

    rdf::Sample s;
    s.setCoor(25, 25);
    s.setDepth(depth);

    rdf::Feature f;
    rdf::Feature f1;
    f.copyTo(f1);
    BOOST_CHECK(f.getX() == f1.getX());
    BOOST_CHECK(f.getY() == f1.getY());

    f.setX(-1.0f);
    f.setY(1.0f);
    const float depth_a = s.getDepth(25, 25);
    const float depth_b = s.getDepth(35, 15); //x: 25 + (-1.0)/0.1 = 15, y: 25 + (1.0)/0.1 = 35

    const float response = f.getResponse(s);
    BOOST_CHECK(response == (depth_b - depth_a));
}
