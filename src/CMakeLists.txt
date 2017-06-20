######### RDF Library ######### 
file(GLOB RDF_LIB_SOURCES rdf/*.cpp        rdf/*.c 
                          preprocess/*.cpp preprocess/*.c)

add_library(RdfLibrary ${RDF_LIB_SOURCES} ${UTIL_SOURCES} ${PROTO_SOURCES})


set(TOOLS_LIBRARIES RdfLibrary ${rdf_LIBRARIES})

set(TOOLS_LIBRARIES ${TOOLS_LIBRARIES} PARENT_SCOPE)

add_subdirectory(tests)
