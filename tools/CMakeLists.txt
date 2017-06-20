######### Executables ######### 

###### Training RDF
add_executable(myRDF myRDF.cpp)
target_link_libraries(myRDF  ${TOOLS_LIBRARIES}) 

###### Preprocess
add_executable(preprocess preprocess.cpp)
target_link_libraries(preprocess ${TOOLS_LIBRARIES}) 
