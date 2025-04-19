#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/transformation_estimation_correntropy_svd.h>
#include <unordered_set>
#include <fstream>
#include <cstdlib>
#include <thread>
#include <cublas_v2.h>
#include <cusolverDn.h>

using PointT = pcl::PointNormal;
using PointCloudT = pcl::PointCloud<PointT>;
int iteration = 0;

Eigen::Matrix4d generateRandomTransformation() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> trans_dist(-0.2f, 0.2f);
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> axis_dist(5.0f, 5.0f);

    Eigen::Vector3d axis(axis_dist(gen), axis_dist(gen), axis_dist(gen));
    axis.normalize();
    double angle = 2.14;
    std::cout<< "\nangle= "<<angle;

    Eigen::AngleAxisd rotation(angle, axis);
    Eigen::Translation3d translation( 5.0f, 5.0f, 5.0f);

    return (translation * rotation).matrix();
}

void logMatrix(const double* data, int rows, int cols, const std::string& name) {
    std::ofstream f(name + ".txt");
    f << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            f << data[j * rows + i] << " ";
        f << "\n";
    }
    f.close();
}
void logMatrixHost(const double* data, int rows, int cols, const std::string& filename) {
    std::ofstream f(filename);
    f << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            f << data[i * cols + j] << " ";
        f << "\n";
    }
    f.close();
}

PointCloudT::Ptr downsample(const PointCloudT::Ptr& cloud, double leaf_size) {
    pcl::VoxelGrid<PointT> vg;
    PointCloudT::Ptr filtered(new PointCloudT);
    vg.setInputCloud(cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*filtered);
    return filtered;
}

void estimateNormals(PointCloudT::Ptr cloud) {
    pcl::NormalEstimation<PointT, PointT> ne;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);
    ne.compute(*cloud);
}

double computeRMSE(const PointCloudT::Ptr& src, const PointCloudT::Ptr& tgt) {
    double error = 0.0;
    for (size_t i = 0; i < src->size(); ++i) {
        double dx = src->points[i].x - tgt->points[i].x;
        double dy = src->points[i].y - tgt->points[i].y;
        double dz = src->points[i].z - tgt->points[i].z;
        error += dx * dx + dy * dy + dz * dz;
    }
    return std::sqrt(error / src->size());
}

pcl::Registration<PointT, PointT>::Ptr get_user_icp(const std::string& method) {
    if (method == "icp") {
        return pcl::IterativeClosestPoint<PointT, PointT>::Ptr(new pcl::IterativeClosestPoint<PointT, PointT>());
    } else if (method == "gicp") {
        return pcl::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>());
    } else if (method == "ndt") {
        return pcl::NormalDistributionsTransform<PointT, PointT>::Ptr(new pcl::NormalDistributionsTransform<PointT, PointT>());
    } else if (method == "svd") {
        auto icp = pcl::IterativeClosestPoint<PointT, PointT>::Ptr(new pcl::IterativeClosestPoint<PointT, PointT>());
        auto svd = pcl::make_shared<pcl::registration::TransformationEstimationSVD<PointT, PointT>>();
        icp->setTransformationEstimation(svd);
        return icp;
    } else if (method == "point2plane") {
        auto icp = pcl::IterativeClosestPoint<PointT, PointT>::Ptr(new pcl::IterativeClosestPoint<PointT, PointT>());
        auto pt2pl = pcl::make_shared<pcl::registration::TransformationEstimationPointToPlane<PointT, PointT>>();
        icp->setTransformationEstimation(pt2pl);
        return icp;
    } else if (method == "point2plane_weighted") {
        auto icp = pcl::IterativeClosestPoint<PointT, PointT>::Ptr(new pcl::IterativeClosestPoint<PointT, PointT>());
        auto pt2pw = pcl::make_shared<pcl::registration::TransformationEstimationPointToPlaneWeighted<PointT, PointT>>();
        icp->setTransformationEstimation(pt2pw);
        return icp;
    } else if (method == "point2plane_lls") {
        auto icp = pcl::IterativeClosestPoint<PointT, PointT>::Ptr(new pcl::IterativeClosestPoint<PointT, PointT>());
        auto pt2plls = pcl::make_shared<pcl::registration::TransformationEstimationPointToPlaneLLS<PointT, PointT>>();
        icp->setTransformationEstimation(pt2plls);
        return icp;
    } else if (method == "icp_nl") {
        return pcl::IterativeClosestPointNonLinear<PointT, PointT>::Ptr(new pcl::IterativeClosestPointNonLinear<PointT, PointT>());
    } else {
        std::cerr << "Unsupported method. Defaulting to ICP." << std::endl;
        return pcl::IterativeClosestPoint<PointT, PointT>::Ptr(new pcl::IterativeClosestPoint<PointT, PointT>());
    }
}

bool step_icp = false;

// Save PCL cloud to XYZ format
bool saveAsXYZ(const PointCloudT::Ptr& cloud, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << cloud->size()<<"\n";
    for (const auto& pt : cloud->points) {
        file << pt.x << " " << pt.y << " " << pt.z << "\n";
    }
    file.close();
    return true;
}

void corruptRandomSubsetWithNonGaussianNoise(
    PointCloudT::Ptr cloud,
    double corruption_ratio,           // e.g., 0.1 = corrupt 10% of points
    double uniform_noise_amplitude,    // e.g., ±0.05 (5 cm)
    double outlier_range               // e.g., ±10.0 (meters)
) {
    int total_points = cloud->size();
    int num_to_corrupt = static_cast<int>(corruption_ratio * total_points);

    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> index_dist(0, total_points - 1);
    std::uniform_real_distribution<double> uniform_noise(-uniform_noise_amplitude, uniform_noise_amplitude);
    std::uniform_real_distribution<double> outlier_dist(-outlier_range, outlier_range);
    std::uniform_real_distribution<double> prob_dist(0.0f, 0.09f);

    std::unordered_set<int> selected_indices;

    // Pick unique random indices to corrupt
    while (selected_indices.size() < static_cast<size_t>(num_to_corrupt)) {
        selected_indices.insert(index_dist(rng));
    }

    // Corrupt the selected points
    for (int idx : selected_indices) {
        auto& pt = cloud->points[idx];

        // 50% chance for uniform noise, 50% for outlier
        if (prob_dist(rng) < 0.5f) {
            // Uniform bounded noise
            pt.x += uniform_noise(rng);
            pt.y += uniform_noise(rng);
            pt.z += uniform_noise(rng);
        } else {
            // Outlier
            pt.x = outlier_dist(rng);
            pt.y = outlier_dist(rng);
            pt.z = outlier_dist(rng);
        }
    }
}
// Write Go-ICP config
bool writeGoICPConfig(const std::string& source_xyz, const std::string& target_xyz) {
    std::ofstream file("goicp_config.txt");
    if (!file.is_open()) return false;
    file << "[Fixed]\n";
    file << target_xyz << "\n";
    file << "[Moving]\n";
    file << source_xyz << "\n";
    file << "[Parameters]\n";
    file << "transthreshold 0.001\n";
    file << "rotthreshold 0.001\n";
    file << "epsilon 0.001\n";
    file << "icp 1 \n";
    file.close();
    return true;
}

// Run Go-ICP executable
int runGoICP() {
    return std::system("./GoICP target.xyz source.xyz 0 config.txt goicpoutput.txt");
}

// Load result transformation
Eigen::Matrix4d loadGoICPResult(const std::string& result_file) {
      Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
    std::ifstream file(result_file);
    if (!file.is_open()) {
        std::cerr << "[Go-ICP] Could not open result file: " << result_file << std::endl;
        return tf;
    }

    std::string line;
    std::cout << "[Go-ICP] Contents of result.log:" << std::endl;
    while (std::getline(file, line)) {
        std::cout << line << std::endl;
    }

    file.clear();
    file.seekg(0);  // rewind for parsing after printing
    std::getline(file, line); // skip again before parsing

    // Now parse the transformation
    double R[9], t[3];
    for (int i = 0; i < 9; ++i) file >> R[i];
    for (int i = 0; i < 3; ++i) file >> t[i];
    tf.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3d>(R).transpose();
    tf(0, 3) = t[0]; tf(1, 3) = t[1]; tf(2, 3) = t[2];
    std::cout <<"\ntf= \n" <<tf;
    return tf;
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    if (event.keyDown() && event.getKeySym() == "space") {
        step_icp = true;
        std::cout << "[SPACE] Performing one ICP iteration..." << std::endl;
    }
}
// Find the closest point in target for each point in source
std::vector<int> findClosestPoints(const PointCloudT::Ptr& source, const PointCloudT::Ptr& target) {
    std::vector<int> closestIndices(source->size());
    for (size_t i = 0; i < source->size(); ++i) {
        double minDist = std::numeric_limits<double>::max();
        int closestIdx = -1;
        for (size_t j = 0; j < target->size(); ++j) {
            double dist = std::pow(source->points[i].x - target->points[j].x, 2) +
                          std::pow(source->points[i].y - target->points[j].y, 2) +
                          std::pow(source->points[i].z - target->points[j].z, 2);
            if (dist < minDist) {
                minDist = dist;
                closestIdx = j;
            }
        }
        closestIndices[i] = closestIdx;
    }
    return closestIndices;
}
Eigen::MatrixXd computeWFullGPU(const Eigen::MatrixXd& H_src,
                                const Eigen::MatrixXd& H_tgt,
                                const Eigen::MatrixXd& S) {
    int N = H_src.cols();
    double *d_Hsrc, *d_Htgt, *d_S, *d_temp, *d_W;

    cudaMalloc(&d_Hsrc, sizeof(double) * 3 * N);
    cudaMalloc(&d_Htgt, sizeof(double) * 3 * N);
    cudaMalloc(&d_S, sizeof(double) * N * N);
    cudaMalloc(&d_temp, sizeof(double) * N * 3);
    cudaMalloc(&d_W, sizeof(double) * 3 * 3);

    // Convert Eigen to row-major double buffers
    std::vector<double> h_Hsrc(3 * N), h_Htgt(3 * N), h_S(N * N);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < 3; ++i) {
            h_Hsrc[i + 3 * j] = static_cast<double>(H_src(i, j));
            h_Htgt[i + 3 * j] = static_cast<double>(H_tgt(i, j));
        }
    }
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            h_S[r + c * N] = static_cast<double>(S(r, c));  // column-major

    cudaMemcpy(d_Hsrc, h_Hsrc.data(), sizeof(double) * 3 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Htgt, h_Htgt.data(), sizeof(double) * 3 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, h_S.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0f, beta = 0.0f;

    // temp = S^T * H_tgt^T
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, 3, N,
                &alpha, d_S, N, d_Htgt, 3, &beta, d_temp, N);

    // W = H_src * temp
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, N,
                &alpha, d_Hsrc, 3, d_temp, N, &beta, d_W, 3);

    double h_W[9];
    cudaMemcpy(h_W, d_W, sizeof(double) * 9, cudaMemcpyDeviceToHost);
  //  logMatrixHost(h_W, 3, 3, "logs/W_final_gpu.txt");

    cublasDestroy(handle);
    cudaFree(d_Hsrc);
    cudaFree(d_Htgt);
    cudaFree(d_S);
    cudaFree(d_temp);
    cudaFree(d_W);

    Eigen::MatrixXd W(3, 3);
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            W(i, j) = h_W[i + j * 3];  // convert column-major to Eigen

    return W;
}

void saveMatrixToFile(const Eigen::MatrixXd& mat, const std::string& filename, int precision = 8) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Error: Cannot open file " << filename << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(precision);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j);
            if (j < mat.cols() - 1) file << " ";
        }
        file << "\n";
    }
    file.close();
}


// Compute the optimal transformation using SVD
Eigen::Matrix4d computeOptimalTransformation(const PointCloudT::Ptr& source,
                                             const PointCloudT::Ptr& target,
                                             const std::vector<int>& correspondences) {
    Eigen::Vector3d centroidSource(0, 0, 0), centroidTarget(0, 0, 0);
    size_t n = source->size();
    for (size_t i = 0; i < n; ++i) {
        centroidSource += Eigen::Vector3d(source->points[i].x, source->points[i].y, source->points[i].z);
        centroidTarget += Eigen::Vector3d(target->points[correspondences[i]].x, target->points[correspondences[i]].y, target->points[correspondences[i]].z);
    }
    centroidSource /= n;
    centroidTarget /= n;

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    double sigma = 0.05;

    Eigen::MatrixXd corr_similarity_matrix = Eigen::MatrixXd::Zero(n,n);
    std::cout << "\nCICP new";
    for (size_t i = 0; i < n; ++i)
    {
        Eigen::MatrixXd s(3,1);
        Eigen::MatrixXd t(3,1);
        s(0,0) = source->points[i].x;
        s(1,0) = source->points[i].y;
        s(2,0) = source->points[i].z;
      
        t(0,0) = target->points[correspondences[i]].x;
        t(1,0) = target->points[correspondences[i]].y;
        t(2,0) = target->points[correspondences[i]].z;

        std::cout<<"\n For input point = ["<< s(0,0)<<"," << s(1,0)<<","<< s(2,0)<<"] with idx = "<<i;
        std::cout<<" Target point = " << "idx = " << correspondences[i] << " pt = [" << t(0,0) << "," << t(1,0) << "," << t(2,0) <<"]"; 

     Eigen::MatrixXd corrval=(s-t).transpose()*(s-t);
    //std::cout<<"\nCor_Sigma="<<cor_sigma;
    //double sigma = 0.05;
    double cval=exp(-(corrval(0,0)*corrval(0,0))/(2*sigma*sigma));
    //std::cout <<" cval = " << cval;
    

      //std::cout<<"\nidx[0]="<<index[0]<<" *idx="<<*idx<<" cval="<<cval<<" CMAT="<<CorrMatrix(index[0],*idx);
      
     corr_similarity_matrix(correspondences[i],i)=cval;
     corr_similarity_matrix(i,correspondences[i])=cval;
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3,n);
    Eigen::MatrixXd H_src = Eigen::MatrixXd::Zero(3,n); 
    Eigen::MatrixXd H_tgt = Eigen::MatrixXd::Zero(3,n);  
    for (size_t i = 0; i < n; ++i) {
      //  if(corr_similarity_matrix > 0.9)
        Eigen::Vector3d ps(source->points[i].x - centroidSource.x(),
                           source->points[i].y - centroidSource.y(),
                           source->points[i].z - centroidSource.z());
        Eigen::Vector3d pt(target->points[correspondences[i]].x - centroidTarget.x(),
                           target->points[correspondences[i]].y - centroidTarget.y(),
                           target->points[correspondences[i]].z - centroidTarget.z());


      //  if(corr_similarity_matrix(i,0)>0.9){      
        H_src.col(i) = ps;
        
        H_tgt.col(i) = pt;
        //}
        

        //W += pt * ps.transpose();
    }
    // std::cout << "\nH_tgt=" << H_tgt;
    // std::cout << "\nH_src=" << H_src;
    // std::cout << "\n corr_sim_matrix" <<corr_similarity_matrix;
    // 
  std::ofstream fs;
  std::stringstream ss3;
  ss3<< iteration;
  std::string idx=ss3.str();

 // fs.open("/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res/corr_matrix_"+idx+".txt",std::ios::app);
 
 // fs << corr_similarity_matrix;
   saveMatrixToFile(corr_similarity_matrix, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res/corr_matrix_"+idx+".txt");
 // fs.close();
  fs.open("/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res/src_cicp_new"+idx+".txt",std::ios::app);
  fs << H_src;
  fs.close();
  fs.open("/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res/tgt_cicp_new"+idx+".txt",std::ios::app);
  fs << H_tgt;
  fs.close();
  

    W = (H_src * corr_similarity_matrix.transpose()* H_tgt.transpose());
  //  W = (H_src * H_tgt.transpose());
    fs.open("/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res/W_matrix"+idx+".txt",std::ios::app);
    fs << W;
    fs.close();

     //W = (H_src * corr_similarity_matrix.transpose()* H_tgt.transpose());
    fs.open("/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res/H_src*coormatrix_matrix"+idx+".txt",std::ios::app);
    fs << H_src * corr_similarity_matrix;
    fs.close();
  
  
  

    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3> u = svd.matrixU ();
    Eigen::Matrix<double, 3, 3> v = svd.matrixV ();

  // Compute R = V * U'
  std::cout << "\npU= \n" << u;
  std::cout << "\nV=\n"<< v;
  
  if (u.determinant () * v.determinant () < 0)
  {
    for (int x = 0; x < 3; ++x)
      v (x, 2) *= -1;
  }

  Eigen::Matrix<double, 3, 3> R = v * u.transpose ();
  std::cout << "\nR=\n" << R;


  // Return the correct transformation
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
  transformation.topLeftCorner (3, 3) = R;
  const Eigen::Matrix<double, 3, 1> Rc (R * centroidSource.head (3));
  transformation.block (0, 3, 3, 1) = centroidTarget.head (3) - Rc;
    // Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();

    // if (R.determinant() < 0) {
    //     Eigen::Matrix3d U = svd.matrixU();
    //     U.col(2) *= -1;
    //     R = U * svd.matrixV().transpose();
    // }

    // Eigen::Vector3d T = centroidTarget - R * centroidSource;

    // Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    // transformation.block<3, 3>(0, 0) = R;
    // transformation.block<3, 1>(0, 3) = T;

    return transformation;
}
// Accepts H_src, H_tgt (3xN), S (NxN) as Eigen::MatrixXd and returns W (3x3) as Eigen::MatrixXd


void saveMatrixAsNumpy(const Eigen::Matrix4d &T, const std::string &filename) 
{
    std::ostringstream ss;
    ss << "python3 -c \"import numpy as np; "
       << "T = np.array(" << "[["
       << T(0,0) << "," << T(0,1) << "," << T(0,2) << "," << T(0,3) << "], ["
       << T(1,0) << "," << T(1,1) << "," << T(1,2) << "," << T(1,3) << "], ["
       << T(2,0) << "," << T(2,1) << "," << T(2,2) << "," << T(2,3) << "], ["
       << T(3,0) << "," << T(3,1) << "," << T(3,2) << "," << T(3,3) << "]]); "
       << "np.save('" << filename << "', T)\"";

    std::string py_cmd = ss.str();
    std::cout << "Saving matrix with:\n" << py_cmd << "\n";
    int ret = std::system(py_cmd.c_str());
    std::cout << "Save return code: " << ret << std::endl;
}

// Apply transformation to a point cloud
PointCloudT::Ptr applyTransformation(const PointCloudT::Ptr& cloud, const Eigen::Matrix4d& transformation) {
    PointCloudT::Ptr transformedCloud(new PointCloudT);
    for (const auto& p : cloud->points) {
        Eigen::Vector4d pVec(p.x, p.y, p.z, 1.0);
        Eigen::Vector4d transformedP = transformation * pVec;
        transformedCloud->push_back(pcl::PointNormal(transformedP.x(), transformedP.y(), transformedP.z()));
    }
    return transformedCloud;
}

// Perform one iteration of ICP
PointCloudT::Ptr performOneCICPIteration(const PointCloudT::Ptr& source,
                             const PointCloudT::Ptr& target, Eigen::Matrix4d& totalTransformation) {
    std::vector<int> correspondences = findClosestPoints(source, target);
    Eigen::Matrix4d transformation = computeOptimalTransformation(source, target, correspondences);
    PointCloudT::Ptr alignedCloud_cicp(new PointCloudT);
    alignedCloud_cicp = applyTransformation(source, transformation);
    totalTransformation = transformation * totalTransformation;
    std::cout << "\nICP Step - Updated Transformation:\n" << totalTransformation << std::endl;
    return alignedCloud_cicp;
}

void runPythonRegistration(const std::string& pcd_path, const std::string& transform_path) {
    std::string cmd = "python3 /media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/test_registration.py "
                      "--pcd " + pcd_path +
                      " --transform " + transform_path;
    std::cout << "[Thread] Running: " << cmd << std::endl;
    int result = std::system(cmd.c_str());
    std::cout << "[Thread] Python script exited with code: " << result << std::endl;
}



// ---------------- CUDA Kernels ----------------

void logVector(const double* data, int size, const std::string& name) {
    std::ofstream f(name + ".txt");
    f << std::fixed << std::setprecision(6);
    for (int i = 0; i < size; ++i)
        f << data[i] << "\n";
    f.close();
}

void logIntVector(const int* data, int size, const std::string& name) {
    std::ofstream f(name + ".txt");
    for (int i = 0; i < size; ++i)
        f << data[i] << "\n";
    f.close();
}
__global__ void nearestNeighborKernel(const double* src, const double* tgt, int N, int M, int* indices) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double min_dist = 1e20f;
    int best = -1;

    double sx = src[3*i+0];
    double sy = src[3*i+1];
    double sz = src[3*i+2];

    for (int j = 0; j < M; ++j) {
        double tx = tgt[3*j+0];
        double ty = tgt[3*j+1];
        double tz = tgt[3*j+2];

        double dx = sx - tx;
        double dy = sy - ty;
        double dz = sz - tz;
        double d = (dx*dx + dy*dy + dz*dz);

        if (d < min_dist) {
            min_dist = d;
            best = j;
        }
    }

    indices[i] = best;

}

__global__ void computeCentroidKernel(const double* pts, double* centroid, int N) {
    __shared__ double sum[3];
    if (threadIdx.x == 0) sum[0] = sum[1] = sum[2] = 0;
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&sum[0], pts[3*i+0]);
        atomicAdd(&sum[1], pts[3*i+1]);
        atomicAdd(&sum[2], pts[3*i+2]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        centroid[0] = sum[0] / N;
        centroid[1] = sum[1] / N;
        centroid[2] = sum[2] / N;
    }
}

__global__ void accumulateCentroid(const double* pts, double* accum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(&accum[0], pts[3*i + 0]);
        atomicAdd(&accum[1], pts[3*i + 1]);
        atomicAdd(&accum[2], pts[3*i + 2]);
    }
}

__global__ void subtractCentroidKernel(const double* input, const double* centroid, double* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[3*i + 0] = input[3*i + 0] - centroid[0];
        output[3*i + 1] = input[3*i + 1] - centroid[1];
        output[3*i + 2] = input[3*i + 2] - centroid[2];
    }
}

__global__ void buildMatrixKernel(const double* pts, double* mat, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        mat[i*3 + 0] = pts[3*i + 0];
        mat[i*3 + 1] = pts[3*i + 1];
        mat[i*3 + 2] = pts[3*i + 2];
    }
}
__global__ void computeSimilarityMatrixFull(const double* src, const double* tgt, const int* nn_indices, double* S, int N, double sigma2) {
      
   
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    //int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = nn_indices[i];  // nearest neighbor index for point i

    double dx = src[3*i+0] - tgt[3*j+0];
    double dy = src[3*i+1] - tgt[3*j+1];
    double dz = src[3*i+2] - tgt[3*j+2];

    //   printf("For input point (GPU) [%f, %f, %f] with idx = %d -> Target idx = %d, pt = [%f, %f, %f]\n",
    //        src[3*i+0], src[3*i+1], src[3*i+2], i,
    //        j, tgt[3*j+0], tgt[3*j+1], tgt[3*j+2]);

    //  printf("For input point (GPU) [%f, %f, %f] with idx = %d\n", src[3*i+0], src[3*i+1], src[3*i+2], i);
    //  printf("Target point (GPU) idx = %d, pt = [%f, %f, %f]\n", j, tgt[3*j+0], tgt[3*j+1], tgt[3*j+2]);

    double dist2 = dx*dx + dy*dy + dz*dz;

    double val = expf(-dist2*dist2 / (2.0f * sigma2));
    S[i * N + j] = val;
    S[j * N + i] = val;

    // ✅ DEBUG print for only the first few threads to avoid log spam
    //if (i < 2 && j < 2) {
     //   printf("S[%d,%d] = %.6f (dist2 = %.6f)\n", i, j, val, dist2);
    //}
}

// GPU W = Hsrc * S^T * Htgt^T
void computeW_on_GPU(const double* d_Hsrc, const double* d_Htgt, const double* d_src, const double* d_tgt,
                     const int* d_indices, double* d_W, int N, cublasHandle_t handle, double sigma = 0.05f) {

    double *d_S, *d_temp;
    cudaMalloc(&d_S, sizeof(double) * N * N);
    cudaMemset(d_S, 0, sizeof(double) * N * N); // zero initialize
    cudaMalloc(&d_temp, sizeof(double) * 3 * N);

    computeSimilarityMatrixFull<<<(N + 255)/256, 256>>>(
        d_src, d_tgt, d_indices, d_S, N, sigma * sigma);
    cudaDeviceSynchronize();

    double alpha = 1.0f, beta = 0.0f;

    // temp = S^T * H_tgt^T
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, 3, N, &alpha, d_S, N, d_Htgt, 3, &beta, d_temp, N);

    // W = H_src * temp = (3 x N) * (N x 3)
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, N, &alpha, d_Hsrc, 3, d_temp, N, &beta, d_W, 3);

    cudaFree(d_S);
    cudaFree(d_temp);
}
__global__ void computeCenteredMatrices(
    const double* src, const double* tgt,
    const int* nn_indices,
    const double* cent_src, const double* cent_tgt,
    double* H_src, double* H_tgt,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int j = nn_indices[i];

    // Source - centroid
    H_src[3*i + 0] = src[3*i + 0] - cent_src[0];
    H_src[3*i + 1] = src[3*i + 1] - cent_src[1];
    H_src[3*i + 2] = src[3*i + 2] - cent_src[2];

    // Target - centroid
    H_tgt[3*i + 0] = tgt[3*j + 0] - cent_tgt[0];
    H_tgt[3*i + 1] = tgt[3*j + 1] - cent_tgt[1];
    H_tgt[3*i + 2] = tgt[3*j + 2] - cent_tgt[2];
}

// ---------------- Host Function ----------------

PointCloudT::Ptr performOneCICPIteration_GPU(const PointCloudT::Ptr& source,
                                              const PointCloudT::Ptr& target,
                                              Eigen::Matrix4d& T_total,
                                              double sigma = 0.05f) {
   int N = source->size();
    int M = target->size();

    std::vector<double> src_flat(3*N), tgt_flat(3*M);
    for (int i = 0; i < N; ++i) {
        src_flat[3*i+0] = source->points[i].x;
        src_flat[3*i+1] = source->points[i].y;
        src_flat[3*i+2] = source->points[i].z;
    }
    for (int i = 0; i < M; ++i) {
        tgt_flat[3*i+0] = target->points[i].x;
        tgt_flat[3*i+1] = target->points[i].y;
        tgt_flat[3*i+2] = target->points[i].z;
    }

    double *d_src, *d_tgt, *d_cent_src, *d_cent_tgt, *d_Hsrc, *d_Htgt, *d_S, *d_temp, *d_W;
    int *d_indices;

    cudaMalloc(&d_src, 3*N*sizeof(double));
    cudaMalloc(&d_tgt, 3*M*sizeof(double));
    cudaMemcpy(d_src, src_flat.data(), 3*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt, tgt_flat.data(), 3*M*sizeof(double), cudaMemcpyHostToDevice);

    // std::vector<double> h_src(3*N), h_tgt(3*M);
    // cudaMemcpy(h_src.data(), d_src, 3*N*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_tgt.data(), d_tgt, 3*M*sizeof(double), cudaMemcpyDeviceToHost);

    // logMatrix(h_src.data(), N, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/src_points");
    // logMatrix(h_tgt.data(), M, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/tgt_points");

    
    cudaMalloc(&d_cent_src, 3*sizeof(double));
    cudaMemset(d_cent_src, 0, 3*sizeof(double));
  
    cudaMalloc(&d_cent_tgt, 3*sizeof(double));
    cudaMemset(d_cent_tgt, 0, 3*sizeof(double));
    // std::vector<double> h_cent_src(3), h_cent_tgt(3);
    // cudaMemcpy(h_cent_src.data(), d_cent_src, 3*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_cent_tgt.data(), d_cent_tgt, 3*sizeof(double), cudaMemcpyDeviceToHost);
    // logMatrix(h_cent_src.data(), 1, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/cent_src_points");
    // logMatrix(h_cent_tgt.data(), 1, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/cent_tgt_points");


    
    cudaMalloc(&d_Hsrc, sizeof(double)*3*N);
    cudaMalloc(&d_Htgt, sizeof(double)*3*N);

    cudaMalloc(&d_S, sizeof(double)*N*N);
    cudaMemset(d_S, 0, sizeof(double) * N * N);
    cudaMalloc(&d_temp, sizeof(double)*3*N);
    cudaMalloc(&d_W, sizeof(double)*9);
    cudaMalloc(&d_indices, sizeof(int)*N);

    nearestNeighborKernel<<<(N+255)/256, 256>>>(d_src, d_tgt, N, M, d_indices);
    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();  
    if (err != cudaSuccess) {std::cerr << "NN->CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;}
    // std::vector<int> h_indices(N);
    // cudaMemcpy(h_indices.data(), d_indices, N * sizeof(int), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < N; i++){
    // std::cout<<"\n(GPU) For input point = ["<< source->points[i].x <<"," << source->points[i].y<<","<< source->points[i].z<<"] with idx = "<<i;
    // std::cout<<" Target point = " << "idx = " << h_indices[i] << " pt = [" << target->points[h_indices[i]].x << "," << target->points[h_indices[i]].y << "," << target->points[h_indices[i]].z <<"]";
    // }
    
    
    assert(d_src != nullptr);
    assert(d_cent_src != nullptr);
    computeCentroidKernel<<<(N+255)/256, 256>>>(d_src, d_cent_src, N);
    err = cudaGetLastError();
    cudaDeviceSynchronize();  
     if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
       }
    computeCentroidKernel<<<(M+255)/256, 256>>>(d_tgt, d_cent_tgt, M);
    err = cudaGetLastError();
    cudaDeviceSynchronize();  
     if (err != cudaSuccess) {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
       }
    
    double h_centroid[3];
    cudaMemcpy(h_centroid, d_cent_src, 3*sizeof(double), cudaMemcpyDeviceToHost);
    logVector(h_centroid, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/centroid_src");
    
    cudaMemcpy(h_centroid, d_cent_tgt, 3*sizeof(double), cudaMemcpyDeviceToHost);
    logVector(h_centroid, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/centroid_tgt");
    
       
    double *d_src_centered, *d_tgt_centered;
    cudaMalloc(&d_src_centered, sizeof(double) * 3 * N);
    cudaMalloc(&d_tgt_centered, sizeof(double) * 3 * N);
    subtractCentroidKernel<<<(N+255)/256, 256>>>(d_src, d_cent_src, d_src_centered, N);
    subtractCentroidKernel<<<(M+255)/256, 256>>>(d_tgt, d_cent_tgt, d_tgt_centered, M);

    // std::vector<double> h_diff_src(3*N), h_diff_tgt(3*M);
    // cudaMemcpy(h_diff_src.data(), d_src_centered, 3*N*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_diff_tgt.data(), d_tgt_centered, 3*M*sizeof(double), cudaMemcpyDeviceToHost);

    // logMatrix(h_diff_src.data(), N, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/src_diff_points");
    // logMatrix(h_diff_tgt.data(), M, 3, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/tgt_diff_points");

    

    // buildMatrixKernel<<<(N+255)/256, 256>>>(d_src, d_Hsrc, N);
    // buildMatrixKernel<<<(N+255)/256, 256>>>(d_tgt, d_Htgt, N);


    dim3 blockDim(16, 16);
    dim3 gridDim((N+15)/16, (N+15)/16);
    computeSimilarityMatrixFull<<<gridDim, blockDim>>>(d_src, d_tgt, d_indices, d_S, N, sigma * sigma);

    std::vector<double> h_S(N * N);
    cudaMemcpy(h_S.data(), d_S, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
     
    logMatrix(h_S.data(), N, N, "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res_gpu/SM_mat");

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    computeCenteredMatrices<<<blocks, threads>>>(
    d_src, d_tgt, d_indices, d_cent_src, d_cent_tgt, d_Hsrc, d_Htgt, N);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0f, beta = 0.0f;

    // temp = S^T * H_tgt^T
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, 3, N, &alpha, d_S, N, d_Htgt, 3, &beta, d_temp, N);

    // W = H_src * temp = 3x3
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, N, &alpha, d_Hsrc, 3, d_temp, N, &beta, d_W, 3);
    cublasDestroy(handle);

    double h_W[9];
    cudaMemcpy(h_W, d_W, sizeof(double)*9, cudaMemcpyDeviceToHost);
    Eigen::Matrix3d Wf = Eigen::Map<Eigen::Matrix3d>(h_W);
    Eigen::Matrix3d W = Wf.cast<double>();
    std::cout << "\n[DEBUG] Covariance matrix W:\n" << W << "\n";

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    Eigen::Matrix3d R = V * U.transpose();
    std::cout << "[DEBUG] Rotation matrix R:\n" << R << "\n";

    double h_csrc[3], h_ctgt[3];
    cudaMemcpy(h_csrc, d_cent_src, 3*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ctgt, d_cent_tgt, 3*sizeof(double), cudaMemcpyDeviceToHost);
    Eigen::Vector3d c_src(h_csrc[0], h_csrc[1], h_csrc[2]);
    Eigen::Vector3d c_tgt(h_ctgt[0], h_ctgt[1], h_ctgt[2]);
    Eigen::Vector3d t = c_tgt - R * c_src;
    std::cout << "[DEBUG] Translation vector t: " << t.transpose() << "\n";

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
    T_total = T * T_total;

    PointCloudT::Ptr out(new PointCloudT);
    for (int i = 0; i < N; ++i) {
        Eigen::Vector4d p(src_flat[3*i+0], src_flat[3*i+1], src_flat[3*i+2], 1.0);
        Eigen::Vector4d tp = T * p;
        out->push_back(PointT(tp[0], tp[1], tp[2]));
    }

    cudaFree(d_src); cudaFree(d_tgt); cudaFree(d_cent_src); cudaFree(d_cent_tgt);
    cudaFree(d_Hsrc); cudaFree(d_Htgt); cudaFree(d_S); cudaFree(d_temp); cudaFree(d_W); cudaFree(d_indices);
    cudaFree(d_src_centered);
    cudaFree(d_tgt_centered);

    return out;
}



int main(int argc, char** argv) {
    // Leaf size and registration method from user input
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <source.pcd> <method> <leaf_size> <outlier_percent>" << std::endl;
        return -1;
    }
    cudaSetDevice(1); 

    PointCloudT::Ptr cloud(new PointCloudT);
    pcl::io::loadPCDFile(argv[1], *cloud);

    double leaf_size = std::stof(argv[3]);
    PointCloudT::Ptr source = downsample(cloud, leaf_size);
    PointCloudT::Ptr target(new PointCloudT);
    Eigen::Matrix4d tr = generateRandomTransformation();
     std::string matrix_file = "../transform.npy";
    saveMatrixAsNumpy(tr, matrix_file); 

    pcl::transformPointCloud(*source, *target, tr);
    corruptRandomSubsetWithNonGaussianNoise(source, std::stof(argv[4]), 0.05f, 5.0f);

    estimateNormals(source);
    estimateNormals(target);
    Eigen::Matrix4d totalTransformation = Eigen::Matrix4d::Identity(); 
    Eigen::Matrix4d totalTransformation_gpu = Eigen::Matrix4d::Identity(); 

  
    // Correntropy ICP
    // pcl::IterativeClosestPoint<PointT, PointT> cicp;
    // auto correntropy = pcl::make_shared<pcl::registration::TransformationEstimationCorrentropySVD<PointT, PointT>>();
    // cicp.setTransformationEstimation(correntropy);
    // cicp.setInputSource(source);
    // cicp.setInputTarget(target);
    // cicp.setMaximumIterations(1);

    PointCloudT::Ptr aligned_cicp(new PointCloudT(*source));
    PointCloudT::Ptr aligned_cicp_gpu(new PointCloudT(*source));

    // User ICP
    auto user_icp = get_user_icp(argv[2]);
    user_icp->setInputSource(source);
    user_icp->setInputTarget(target);
    user_icp->setMaximumIterations(1);
    PointCloudT::Ptr aligned_user(new PointCloudT(*source));

    // Visualization
    pcl::visualization::PCLVisualizer viewer("CoSM ICP Viewer");
    viewer.registerKeyboardCallback(keyboardEventOccurred, nullptr);
    int v1, v2;
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer.setBackgroundColor(1, 1, 1, v1);
    viewer.setBackgroundColor(1, 1, 1, v2);

    // Add text labels
    viewer.addText("Source (Black) Target (Dark Green) Aligned (Red)", 10, 10, 14, 0, 0, 0, "info_text", v1);
    viewer.addText("Source (Black) Target (Dark Green) Aligned (Red)", 10, 10, 14, 0, 0, 0, "info_text2", v2);
    viewer.addText("CosmICP", 10, 180, 14, 0, 0, 0, "info_text1", v1);
    
    

    // Display random transformation
    std::ostringstream tf_stream;
    tf_stream << argv[2] << "\n \n";
    Eigen::Matrix4d tf = generateRandomTransformation();
    tf_stream << "Random Transformation:\n" << tf;
    viewer.addText(tf_stream.str(), 10, 80, 12, 0, 0, 0, "tf_text", v1);
    tf_stream << "\n \n" << argv[2];
    std::cout<< "\ntf_stream = " << tf_stream.str();
     std::cout<< "\nargcv[2] = " << argv[2];
    viewer.addText(tf_stream.str(), 10, 180, 12, 0, 0, 0, "info_text3", v2);
    // Add text labels
    viewer.addText("Source (Black) Target (Dark Green) Aligned (Red)", 10, 10, 14, 0, 0, 0, "info_text", v1);
    viewer.addText("Source (Black) Target (Dark Green) Aligned (Red)", 10, 10, 14, 0, 0, 0, "info_text2", v2);

    // // Display random transformation
    // std::ostringstream tf_stream;
    // Eigen::Matrix4d tf = generateRandomTransformation();
    // tf_stream << "Random Transformation:" << tf;
    // viewer.addText(tf_stream.str(), 10, 80, 12, 0, 0, 0, "tf_text", v1);

    // Color handlers
    auto black = pcl::visualization::PointCloudColorHandlerCustom<PointT>(source, 0, 0, 0);
    auto darkgreen = pcl::visualization::PointCloudColorHandlerCustom<PointT>(target, 0, 100, 0);
    auto red1 = pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_cicp, 255, 0, 0);
    auto red2 = pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_user, 255, 0, 0);

    viewer.addPointCloud(source, black, "src1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "src1", v1);
    viewer.addPointCloud(target, darkgreen, "tgt1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "tgt1", v1);
    viewer.addPointCloud(aligned_cicp, red1, "aligned1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned1", v1);

    viewer.addPointCloud(source, black, "src2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "src2", v2);
    viewer.addPointCloud(target, darkgreen, "tgt2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "tgt2", v2);
    if (std::string(argv[2]) != "goicp"){
    viewer.addPointCloud(aligned_user, red2, "aligned2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned2", v2);
    }
    // Optional Go-ICP integration if selected
    if (std::string(argv[2]) == "goicp") {
        saveAsXYZ(source, "source.xyz");
        saveAsXYZ(target, "target.xyz");
        writeGoICPConfig("source.xyz", "target.xyz");
        runGoICP();
        Eigen::Matrix4d goicp_tf = loadGoICPResult("goicpoutput.txt");

        PointCloudT::Ptr aligned_goicp(new PointCloudT);
        pcl::transformPointCloud(*source, *aligned_goicp, goicp_tf);

        auto blue = pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_goicp, 0,0,255);
        viewer.addPointCloud(aligned_goicp, blue, "aligned_goicp", v2);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned_goicp", v2);
    }

    // Call the Python script
    std::string pcd_path = argv[1];
    std::string transform_path = "/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/transform.npy";
   // std::thread py_thread(runPythonRegistration, pcd_path, transform_path);
    
    
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
        if (step_icp) {
            PointCloudT::Ptr temp1(new PointCloudT);

              ///////////////// Run on GPU ///////////////////////////////////////////////////////////

                std::vector<double> src_flat, tgt_flat;
                for (const auto& pt : *aligned_cicp_gpu) {
                src_flat.push_back(pt.x);
                src_flat.push_back(pt.y);
                src_flat.push_back(pt.z);
                }
                for (const auto& pt : *target) {
                tgt_flat.push_back(pt.x);
                tgt_flat.push_back(pt.y);
                tgt_flat.push_back(pt.z);
                }

                //runICP_GPU(src_flat, tgt_flat, source->size(), target->size(),totalTransformation_gpu);
          //  aligned_cicp_gpu = performOneCICPIteration_GPU(aligned_cicp_gpu, target,totalTransformation_gpu);
            /////////////////////////////////////////////////////////////////////////////////////////////////////
            
            aligned_cicp = performOneCICPIteration(aligned_cicp, target,totalTransformation);
            // cicp.setInputSource(aligned_cicp);
            // cicp.align(*temp1);
            //*aligned_cicp =;

            PointCloudT::Ptr temp2(new PointCloudT);
            user_icp->setInputSource(aligned_user);
            user_icp->align(*temp2);
            *aligned_user = *temp2;

            double rmse_correntropy = computeRMSE(aligned_cicp, target);
            double rmse_user = computeRMSE(aligned_user, target);
            std::cout << "[Correntropy ICP] RMSE: " << rmse_correntropy
                      << " | [" << argv[2] << "] RMSE: " << rmse_user << std::endl;
            
            std::cout <<"\nIteration_number: " << ++iteration;

            viewer.updatePointCloud(aligned_cicp, pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_cicp, 255, 0, 0), "aligned1");
            if (std::string(argv[2]) != "goicp") {
            viewer.updatePointCloud(aligned_user, pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_user, 255, 0, 0), "aligned2");
            }
            step_icp = false;
        }
    }
    return 0;
}
