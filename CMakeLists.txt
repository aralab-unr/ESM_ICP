cmake_minimum_required(VERSION 3.5)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(useEigen LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "61")

# Enable standard CUDA toolkit
enable_language(CUDA)

# Optional: CUDA flags (e.g., debug printf or relaxed constexpr)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Optimization and warnings
set(CMAKE_CXX_FLAGS "-O3 -march=native -mtune=intel -msse4.2 -mavx2 -mfma -flto -fopenmp -fPIC -Wno-deprecated -Wenum-compare -Wno-ignored-attributes")

find_package(PCL REQUIRED PATHS "/media/ashu/09FDAC46654EECDA/Libraries/pcl_1_15/install_dir/share/pcl-1.15")
include_directories(

     #${CERES_INCLUDE_DIRS}
     # ${OpenCV_INCLUDE_DIRS}
     # ${G2O_INCLUDE_DIRS} 
     # ${Sophus_INCLUDE_DIRS}
     # ${realsense2_INCLUDE_DIR}
     ${PCL_INCLUDE_DIR}
)
#message("g20 libraries=",  ${G2O_LIBRARIES})

list(APPEND thirdparty_libraries
     # ${OpenCV_LIBRARIES}
     # #${CERES_LIBRARIES}
     # ${G2O_LIBRARIES}
     # fmt
     # g2o_core g2o_stuff
     # ${realsense2_LIBRARY}
     ${PCL_LIBRARIES}
)

add_executable(cosmicp cosmicp.cpp)
add_executable(cosmicp_gpu cosmicp_gpu.cu)
# add_executable(cuda_test cuda_test.cu)
# add_executable(practicetest practice_test.cpp)
# add_executable(practiceKF practice_KF.cpp)
# add_executable(opencvcpuutest opencv_cputest.cpp)
target_link_libraries(cosmicp ${thirdparty_libraries})


add_executable (ESM_ICP_demo_Viewer ESM_ICP_demo_Viewer.cpp)
target_link_libraries (ESM_ICP_demo_Viewer ${thirdparty_libraries})

target_link_libraries (cosmicp_gpu ${thirdparty_libraries} ${CUDA_LIBRARIES} -lcusolver -lcublas)

# add_executable(opencvoptflowtest opencv_optflowtest.cpp)
# target_link_libraries(opencvoptflowtest ${thirdparty_libraries})
