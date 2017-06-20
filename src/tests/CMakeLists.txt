include(CTest)

add_definitions (-DBOOST_TEST_DYN_LINK)

set(TESTS_LIBRARIES RdfLibrary 
                    ${rdf_LIBRARIES}
                    ${Boost_FILESYSTEM_LIBRARY}
                    ${Boost_SYSTEM_LIBRARY}
                    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
    
###### Tests:
add_executable       (FeatureTest featuretest.cpp)
target_link_libraries(FeatureTest ${TESTS_LIBRARIES})
add_test        (NAME FeatureTest COMMAND FeatureTest)

    
    
add_custom_target(tests)
add_dependencies(tests 
                    FeatureTest
                )
