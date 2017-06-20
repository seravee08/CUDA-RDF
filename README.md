# CUDA-RDF
A CUDA version random decision forest, provides 10+ times speedups over CPU counterpart.

The CPU version of the RDF is taken from Zhi: https://github.com/czkg/RDF

Compile RDF under Windows
=============================================
Dependencies:
1. Boost 1.62.0: http://www.boost.org/users/history/version_1_62_0.html
2. OpenCV 2.4.11: http://opencv.org/downloads.html
3. ProtoBuf 3.0.0: https://developers.google.com/protocol-buffers/

Tools:
1. Visual Studio 2013
2. CMake 3.7.1
3. Cygwin

Compile ProtoBuf
=============================================
1. Goto the release page and download "protobuf-cpp-3.0.0.zip"
2. Download "protoc-3.0.0-win32.zip" and unzip it to get a bin folder in which
there is the protoc compiler binary
3. Add the path to the protoc compiler to the Windows environment variable (PATH)
4. Unzip the file in step 1 and get a folder called "protobuf-3.0.0" which is referred
to as $PROTO_ROOT$ in the remainder of this document
5. Open CMake and set the $PROTO_ROOT/cmake$ as the path to the source code and make a
folder under $PROTO_ROOT/cmake$ called build, add $PROTO_ROOT/cmake/build$ as the path
to build the binaries
6. Click Configure and then Generate
7. Use Visual Studio to open the "protobuf.sln" under $PROTO_ROOT/cmake/build$
8. Goto the project property and set the Runtime Library to MT under Code Generation Tab
9. Compile the project with x64
10. You will have two folders $PROTO_ROOT/cmake/build/Release$ and $PROTO_ROOT/cmake/build/Debug$
in which there are complied binaries

Compile Boost
=============================================
1. Unzip the downloaded file and get a folder called "boost_1_62_0" which will be
referred to as $BOOST_ROOT$ in the remainder of this document
2. Open a command prompt as an administrator
3. Change directory to $BOOST_ROOT$
4. Run >bootstrap
5. Run >bjam --toolset=msvc-12.0 architecture=x86 address-model=64 stage
6. The compiled binaries are under $BOOST_ROOT/stage$

Compile RDF
=============================================
1. Open Cygwin and change directory to the folder where you extract the RDF codes,
we refer to this folder as $RDF_ROOT$
2. Run $create_proto.sh
3. Create a folder build under $RDF_ROOT$
4. Set $RDF_ROOT$ as the path to the source code and set $RDF_ROOT/build$ as the path to build
the binaries
5. Click Configure
6. If CMake cannot find any path automatically, manually input the path to the necessary
dependencies
7. Click Generate
8. Use Visual Studio to open the .sln file under $RDF_ROOT/build$
9. Try to compile the project in x64
10. "FeatureTest" might require a different boost library we don't have, we can simply
delete this sub-project in that case
11. Add headers <sys/types.h>, <sys/stat.h> <io.h> <stdio.h> to myRDF.cpp, and change the
Linux specific functions open and close to _open and _close respectively
