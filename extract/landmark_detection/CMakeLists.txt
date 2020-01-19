cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD  17)
set(PROJECT_LIB_NAME landmark_detection)

project(boost_lib)

find_package(Boost REQUIRED COMPONENTS python3 numpy3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

set(PROJECT_EXTLIB_DIR third-party)
#pbcvt
set(PBCVT_DIRNAME pyboostcvconverter)
#add_subdirectory(${PROJECT_EXTLIB_DIR}/${PBCVT_DIRNAME})


set(PBCVT_INCLUDE_DIR ${PROJECT_EXTLIB_DIR}/${PBCVT_DIRNAME}/include)

set(PBCVT_LIBRARY ${CMAKE_HOME_DIRECTORY}/${PROJECT_EXTLIB_DIR}/${PBCVT_DIRNAME}/lib/pbcvt.cpython-36m-x86_64-linux-gnu.so)
#set(PBCVT_LIBRARY pbcvt)
#pbcvt_add_pbcvt_library(${PBCVT_LIBRARY} 0)


message(${PBCVT_LIBRARY})

include_directories(
        ${Boost_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
        ${PBCVT_INCLUDE_DIR}
)
link_directories(${Boost_LIBRARY_DIR})

set(ALL_LIBRARIES
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${PBCVT_LIBRARY}
        )


add_library(${PROJECT_LIB_NAME} SHARED
        #binding
        pyInterface.cpp
        #source
        facemark.cpp
        vec.cpp
        #header
        facemark.hpp
        vec.hpp
        )


add_executable(test
        test.cpp
        facemark.cpp
        facemark.hpp
        vec.cpp
        vec.hpp
        )

target_link_libraries(test ${ALL_LIBRARIES})

target_link_libraries(${PROJECT_LIB_NAME} ${ALL_LIBRARIES})

set_target_properties(${PROJECT_LIB_NAME} PROPERTIES SUFFIX .so)
set_target_properties(${PROJECT_LIB_NAME} PROPERTIES PREFIX "")

# compilation flags
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 ")
    set(CMAKE_CXX_FLAGS_DEBUG_INIT "/g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -std=c++${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g")
endif(WIN32)

