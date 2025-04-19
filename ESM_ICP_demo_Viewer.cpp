// CoSM ICP Viewer - With Iterative Alignment Visualization

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


using PointT = pcl::PointNormal;
using PointN = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;
int iteration = 0;
bool perform_alignment = false;
bool run_goicp = true;
bool run_deeplearningmodel = true;

Eigen::Matrix4f generateRandomTransformation() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Angles in radians: ±π
    std::uniform_real_distribution<float> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<float> trans_dist(-10.01f, 10.01f);  // No translation

    // Generate Euler angles in radians
    float roll  = angle_dist(gen);   // rotation around X
    float pitch = angle_dist(gen);   // rotation around Y
    float yaw   = angle_dist(gen);   // rotation around Z

    // Rotation matrices
    Eigen::Matrix3f rot_x = Eigen::AngleAxisf(roll,  Eigen::Vector3f::UnitX()).toRotationMatrix();
    Eigen::Matrix3f rot_y = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()).toRotationMatrix();
    Eigen::Matrix3f rot_z = Eigen::AngleAxisf(yaw,   Eigen::Vector3f::UnitZ()).toRotationMatrix();

    // Combined rotation (ZYX convention)
    Eigen::Matrix3f rotation = rot_z * rot_y * rot_x;

    // Translation vector (zero in this case)
    Eigen::Vector3f translation(
        trans_dist(gen),
        trans_dist(gen),
        trans_dist(gen)
    );

    // Final 4x4 transformation
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0) = rotation;
    transform.block<3,1>(0,3) = translation;

     // Convert back to roll-pitch-yaw (ZYX order)
    Eigen::Vector3f euler_angles = rotation.eulerAngles(2, 1, 0);  // ZYX: yaw, pitch, roll

    std::cout << "Generated transformation: [roll, pitch, yaw, tx, ty, tz] = ["
              << euler_angles[2] << ", "  // roll
              << euler_angles[1] << ", "  // pitch
              << euler_angles[0] << ", "  // yaw
              << translation[0] << ", "
              << translation[1] << ", "
              << translation[2] << "]" << std::endl;


    return transform;
}
void computeTransformationErrors(const Eigen::Matrix4f& predicted,
                                 const Eigen::Matrix4f& ground_truth,
                                 const std::string& method_name) {
    // Extract rotation and translation
    Eigen::Matrix3f R_pred = predicted.block<3,3>(0,0);
    Eigen::Vector3f t_pred = predicted.block<3,1>(0,3);

    Eigen::Matrix3f R_gt = ground_truth.block<3,3>(0,0);
    Eigen::Vector3f t_gt = ground_truth.block<3,1>(0,3);

    // Rotation error (as rotation matrix difference)
    Eigen::Matrix3f R_diff = R_pred - R_gt;
    float mse_rot = R_diff.squaredNorm() / 9.0f;
    float rmse_rot = std::sqrt(mse_rot);
    float mae_rot = (R_diff.cwiseAbs()).sum() / 9.0f;

    // Translation error
    Eigen::Vector3f t_diff = t_pred - t_gt;
    float mse_trans = t_diff.squaredNorm() / 3.0f;
    float rmse_trans = std::sqrt(mse_trans);
    float mae_trans = t_diff.cwiseAbs().sum() / 3.0f;

    // Output
    std::cout << "\n📌 [" << method_name << "] Transformation Error Metrics:\n";

    std::cout << "🔍 Rotation Error:\n"
              << "    - MSE  = " << mse_rot << "\n"
              << "    - RMSE = " << rmse_rot << "\n"
              << "    - MAE  = " << mae_rot << std::endl;

    std::cout << "📦 Translation Error:\n"
              << "    - MSE  = " << mse_trans << "\n"
              << "    - RMSE = " << rmse_trans << "\n"
              << "    - MAE  = " << mae_trans << std::endl;
}

PointCloudT::Ptr downsample(const PointCloudT::Ptr& cloud, float leaf_size) {
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

float computeRMSE(const PointCloudT::Ptr& src, const PointCloudT::Ptr& tgt) {
    float error = 0.0;
    for (size_t i = 0; i < src->size(); ++i) {
        float dx = src->points[i].x - tgt->points[i].x;
        float dy = src->points[i].y - tgt->points[i].y;
        float dz = src->points[i].z - tgt->points[i].z;
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
                auto ndt = pcl::NormalDistributionsTransform<PointT, PointT>::Ptr(
                new pcl::NormalDistributionsTransform<PointT, PointT>()
            );

            // Set typical NDT parameters (you can tune these)
            ndt->setResolution(1.0);
            ndt->setStepSize(0.1);
            ndt->setTransformationEpsilon(0.01);
            ndt->setMaximumIterations(35);
            return ndt;
        //return pcl::NormalDistributionsTransform<PointT, PointT>::Ptr(new pcl::NormalDistributionsTransform<PointT, PointT>());
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
    float corruption_ratio,           // e.g., 0.1 = corrupt 10% of points
    float uniform_noise_amplitude,    // e.g., ±0.05 (5 cm)
    float outlier_range               // e.g., ±10.0 (meters)
) {
    int total_points = cloud->size();
    int num_to_corrupt = static_cast<int>(corruption_ratio * total_points);

    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> index_dist(0, total_points - 1);
    std::uniform_real_distribution<float> uniform_noise(-uniform_noise_amplitude, uniform_noise_amplitude);
    std::uniform_real_distribution<float> outlier_dist(-outlier_range, outlier_range);
    std::uniform_real_distribution<float> prob_dist(0.0f, 0.09f);

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
    return std::system("../Go-ICP/build/./GoICP target.xyz source.xyz 0 ../Go-ICP/demo/config.txt goicpoutput.txt");
}

// Load result transformation
Eigen::Matrix4f loadGoICPResult(const std::string& result_file) {
      Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
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
    float R[9], t[3];
    for (int i = 0; i < 9; ++i) file >> R[i];
    for (int i = 0; i < 3; ++i) file >> t[i];
    tf.block<3, 3>(0, 0) = Eigen::Map<Eigen::Matrix3f>(R).transpose();
    tf(0, 3) = t[0]; tf(1, 3) = t[1]; tf(2, 3) = t[2];
    std::cout <<"\ntf= \n" <<tf;
    return tf;
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void* viewer_void) {
    if (event.keyDown() && event.getKeySym() == "space") {
        perform_alignment = true;
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

// Compute the optimal transformation using SVD
Eigen::Matrix4d computeOptimalTransformation(const PointCloudT::Ptr& source,
                                             const PointCloudT::Ptr& target,
                                             const std::vector<int>& correspondences) {
    Eigen::Vector3d centroidSource(0, 0, 0), centroidTarget(0, 0, 0);
    // size_t n =0;
    // if(source.size() <= target.size() ) n = source.size();
    // else n = target.size();
    size_t n = source->size();
    for (size_t i = 0; i < n; ++i) 
    {
        centroidSource += Eigen::Vector3d(source->points[i].x, source->points[i].y, source->points[i].z);
        centroidTarget += Eigen::Vector3d(target->points[correspondences[i]].x, target->points[correspondences[i]].y, target->points[correspondences[i]].z);
    }
    centroidSource /= n;
    centroidTarget /= n;

    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    double sigma = 1.0;

    Eigen::MatrixXd corr_similarity_matrix = Eigen::MatrixXd::Zero(n,n);
    std::cout << "\nESM_ICP new";
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

    
        Eigen::MatrixXd corrval=(s-t).transpose()*(s-t);
        double cval=exp(-std::pow(corrval(0,0),1)/(2*sigma*sigma));  
        corr_similarity_matrix(correspondences[i],i)=cval;
        corr_similarity_matrix(i,correspondences[i])=cval;
    }

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3,n);
    Eigen::MatrixXd H_src = Eigen::MatrixXd::Zero(3,n); 
    Eigen::MatrixXd H_tgt = Eigen::MatrixXd::Zero(3,n);  
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3d ps(source->points[i].x - centroidSource.x(),
                           source->points[i].y - centroidSource.y(),
                           source->points[i].z - centroidSource.z());
        Eigen::Vector3d pt(target->points[correspondences[i]].x - centroidTarget.x(),
                           target->points[correspondences[i]].y - centroidTarget.y(),
                           target->points[correspondences[i]].z - centroidTarget.z());


        H_src.col(i) = ps;
        
        H_tgt.col(i) = pt;
    }
    
  
    

    W = (H_src * corr_similarity_matrix.transpose()* H_tgt.transpose());
    
     
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 3, 3> u = svd.matrixU ();
    Eigen::Matrix<double, 3, 3> v = svd.matrixV ();
    Eigen::Matrix3d D = svd.singularValues().asDiagonal();

  
     
  
    if (u.determinant () * v.determinant () < 0)
    {
      for (int x = 0; x < 3; ++x)
      v (x, 2) *= -1;
    }

  Eigen::Matrix<double, 3, 3> R = v * u.transpose ();
 
  // Return the correct transformation
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
  transformation.topLeftCorner (3, 3) = R;
  const Eigen::Matrix<double, 3, 1> Rc (R * centroidSource.head (3));
  transformation.block (0, 3, 3, 1) = centroidTarget.head (3) - Rc;
 
    return transformation;
}

void saveMatrixAsNumpy(const Eigen::Matrix4f &T, const std::string &filename) 
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
PointCloudT::Ptr performOneESM_ICPIteration(const PointCloudT::Ptr& source,
                             const PointCloudT::Ptr& target, Eigen::Matrix4d& totalTransformation) {
    std::vector<int> correspondences = findClosestPoints(source, target);
    Eigen::Matrix4d transformation = computeOptimalTransformation(source, target, correspondences);
    PointCloudT::Ptr alignedCloud_ESM_ICP(new PointCloudT);
    alignedCloud_ESM_ICP = applyTransformation(source, transformation);
    totalTransformation = transformation * totalTransformation;
    std::cout << "ICP Step - Updated Transformation:\n" << totalTransformation << std::endl;
    return alignedCloud_ESM_ICP;
}
void runPythonRegistration(const std::string& pcd_path,
                           const std::string& transform_path,
                           const std::string& model_name,
                           const std::string& output_pcd_path) {
    // Path to Python script
    const std::string python_script = "../DL_registration.py";

    // Start building command
    std::string cmd = "python3 \"" + python_script + "\" "
                      "--pcd \"" + pcd_path + "\" "
                      "--transform \"" + transform_path + "\" "
                      "--model " + model_name + " "
                      "--save_path \"" + output_pcd_path + "\"";

    // // Add pretrained model path if provided
    // if (!pretrained_model_path.empty()) {
    //     cmd += " --pretrained \"" + pretrained_model_path + "\"";
    // }

    std::cout << "[Python] Running registration command:\n" << cmd << std::endl;
    int result = std::system(cmd.c_str());
    if (result == 0) {
        std::cout << "[Python] ✅ Script finished successfully.\n";
    } else {
        std::cerr << "[Python] ❌ Script failed with exit code: " << result << std::endl;
    }
}
Eigen::Matrix4f loadTransformationFromFile(const std::string& filename) {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open transformation file: " << filename << std::endl;
        return transform;
    }
    for (int i = 0; i < 4 && file; ++i)
        for (int j = 0; j < 4 && file; ++j)
            file >> transform(i, j);

    file.close();
    std::cout << "📥 Loaded transformation from " << filename << ":\n" << transform << std::endl;
    return transform;
}
bool fileExists(const std::string& filename) {
    std::ifstream infile(filename);
    return infile.good();
}
void saveTransformationToFile(const Eigen::Matrix4f& transform, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open file for saving transformation: " << filename << std::endl;
        return;
    }

    // Write matrix in row-major order
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            file << transform(i, j);
            if (j < 3) file << " ";
        }
        file << "\n";
    }

    file.close();
    std::cout << "✅ Transformation saved to " << filename << std::endl;
}
Eigen::Matrix4d loadTransformationFromTxt_dlm(const std::string& filename) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open predicted transformation file: " << filename << std::endl;
        return transform;
    }
    for (int i = 0; i < 4 && file; ++i)
        for (int j = 0; j < 4 && file; ++j)
            file >> transform(i, j);
    file.close();
    std::cout << "📥 Loaded deep model predicted transform:\n" << transform << std::endl;
    return transform;
}

int main(int argc, char** argv) {
    // Leaf size and registration method from user input
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <source.pcd> <method> <leaf_size> <outlier_percent>" << std::endl;
        return -1;
    }

    PointCloudT::Ptr cloud(new PointCloudT);
    pcl::io::loadPCDFile(argv[1], *cloud);
    Eigen::Matrix4f tr;
    //Eigen::Matrix4f tr_src = generateRandomTransformation();
   // pcl::transformPointCloud(*cloud, *cloud, tr_src);
    bool use_saved_transform = (argc > 5 && std::string(argv[5]) == "--use_saved_transform");
    std::string transform_file_txt = "../last_transform.txt";
    if (use_saved_transform && fileExists(transform_file_txt)) {
        tr = loadTransformationFromFile(transform_file_txt);
    } else 
    {
        tr = generateRandomTransformation();
        saveTransformationToFile(tr, transform_file_txt);
    }
    float leaf_size = std::stof(argv[3]);
    PointCloudT::Ptr source = downsample(cloud, leaf_size);
    PointCloudT::Ptr target(new PointCloudT);
    
     std::string matrix_file = "../transform.npy";
    saveMatrixAsNumpy(tr, matrix_file); 
    
    std::cout<< "\n random transfomration applied=\n" <<  tr <<"\n";
    pcl::transformPointCloud(*source, *target, tr);
    corruptRandomSubsetWithNonGaussianNoise(source, std::stof(argv[4]), 100.05f, 5.0f);
    //source = downsample(cloud, 0.009);
    estimateNormals(source);
    estimateNormals(target);
    Eigen::Matrix4d totalTransformation = Eigen::Matrix4d::Identity(); 
    
    Eigen::Matrix4d T_icp_total = Eigen::Matrix4d::Identity(); 
    Eigen::Matrix4d T_icp_nl_total = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_icp_p2pl_total = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_goicp_total = Eigen::Matrix4d::Identity();  // Already exists if using

    // Correntropy ICP
    // pcl::IterativeClosestPoint<PointT, PointT> ESM_ICP;
    // auto correntropy = pcl::make_shared<pcl::registration::TransformationEstimationCorrentropySVD<PointT, PointT>>();
    // ESM_ICP.setTransformationEstimation(correntropy);
    // ESM_ICP.setInputSource(source);
    // ESM_ICP.setInputTarget(target);
    // ESM_ICP.setMaximumIterations(1);

    PointCloudT::Ptr aligned_ESM_ICP(new PointCloudT(*source));

    // User ICP
    auto user_icp = get_user_icp(argv[2]);
   // if(std::string(argv[2]) = "ndt")
    user_icp->setInputSource(source);
    user_icp->setInputTarget(target);
    user_icp->setMaximumIterations(1);
    
// If method is NDT, cast and configure it properly 
 auto ndt_ptr = std::dynamic_pointer_cast<pcl::NormalDistributionsTransform<PointT, PointT>>(user_icp);
if (std::string(argv[2]) == "ndt") {
   
    if (ndt_ptr) {
        ndt_ptr->setResolution(200.0);
        ndt_ptr->setStepSize(0.1);
        ndt_ptr->setTransformationEpsilon(0.01);
        ndt_ptr->setMaximumIterations(35);  // ✅ Set correctly on real NDT object
        ndt_ptr->setInputSource(source);
        ndt_ptr->setInputTarget(target);
        // Required fix
        auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
        ndt_ptr->setSearchMethodTarget(tree, true);
    } else {
        std::cerr << "[ERROR] Failed to cast to NDT!" << std::endl;
        return -1;
    }
   } 
   else 
   {
    user_icp->setMaximumIterations(1);  // For ICP variants
   }
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
    viewer.addText("ESM_ICP", 10, 180, 14, 0, 0, 0, "info_text1", v1);
    
    

    // Display random transformation
    std::ostringstream tf_stream;
    tf_stream << argv[2] << "\n \n";
    //Eigen::Matrix4f tf = generateRandomTransformation();
    tf_stream << "Random Transformation:\n" << tr;
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
    // Eigen::Matrix4f tf = generateRandomTransformation();
    // tf_stream << "Random Transformation:" << tf;
    // viewer.addText(tf_stream.str(), 10, 80, 12, 0, 0, 0, "tf_text", v1);

    // Color handlers
    auto red = pcl::visualization::PointCloudColorHandlerCustom<PointT>(source, 255, 0, 0);
    auto darkgreen = pcl::visualization::PointCloudColorHandlerCustom<PointT>(target, 0, 100, 0);
    auto red1 = pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_ESM_ICP, 255, 0, 0);
    auto red2 = pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_user, 255, 0, 0);

    viewer.addPointCloud(source, red, "src1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "src1", v1);
    viewer.addPointCloud(target, darkgreen, "tgt1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "tgt1", v1);
    viewer.addPointCloud(aligned_ESM_ICP, red1, "aligned1", v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "aligned1", v1);

    viewer.addPointCloud(source, red, "src2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "src2", v2);
    viewer.addPointCloud(target, darkgreen, "tgt2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "tgt2", v2);
    if (std::string(argv[2]) == "icp" || std::string(argv[2]) == "point2plane" || std::string(argv[2]) == "icp_nl" || std::string(argv[2]) == "ndt"   ){
    viewer.addPointCloud(aligned_user, red2, "aligned2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "aligned2", v2);
    }
    

    // Call the Python script
    std::string pcd_path = argv[1];
    std::string transform_path = "../transform.npy";
   // std::thread py_thread(runPythonRegistration, pcd_path, transform_path);
    pcl::io::savePCDFileASCII("../saved_pcds/aligned_ESM_ICP_"+ std::to_string(iteration) +".pcd", *aligned_ESM_ICP);
    pcl::io::savePCDFileASCII("../saved_pcds/target.pcd", *target);

    
    while (!viewer.wasStopped()) {
        
        viewer.spinOnce(100);
        
        
        if (perform_alignment) {
            
             if ((std::string(argv[2]) == "dcp" || std::string(argv[2]) == "deepgmr" || std::string(argv[2]) == "pointnetlk" 
                  || std::string(argv[2]) == "rpmnet") and run_deeplearningmodel == true) {
                
               runPythonRegistration("../saved_pcds/aligned_ESM_ICP_"+ std::to_string(iteration) +".pcd",
                      "../transform.npy",
                      std::string(argv[2]),
                      "../saved_pcds/aligned_"+ std::string(argv[2])+ ".pcd");
                
                
                   PointCloudT::Ptr deep_cloud(new PointCloudT);
                    if (pcl::io::loadPCDFile<PointT>("../saved_pcds/aligned_"+ std::string(argv[2])+ ".pcd", *deep_cloud) == -1) {
                        PCL_ERROR("❌ Couldn't read file %s\n", "../saved_pcds/aligned_"+ std::string(argv[2])+ ".pcd");
                       // return nullptr;
                    } else {
                        std::cout << "✅ Loaded " << cloud->size() << " points from " << "../saved_pcds/aligned_"+ std::string(argv[2])+ ".pcd" << std::endl;
                    }
                Eigen::Matrix4d T_deep_total = loadTransformationFromTxt_dlm("predicted_transform.txt");
                computeTransformationErrors(T_deep_total.cast<float>(), tr, std::string(argv[2]));     
                viewer.removePointCloud("src2", v2);
                auto red = pcl::visualization::PointCloudColorHandlerCustom<PointT>(deep_cloud, 255,0,0);
                viewer.addPointCloud(deep_cloud, red, "aligned_"+ std::string(argv[2]), v2);
                viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "aligned_"+ std::string(argv[2]), v2);
                run_deeplearningmodel = false;
                
            }
            for(int i = 0; i < 40; i++) 
            {
             viewer.spinOnce(100);
            PointCloudT::Ptr temp1(new PointCloudT);
            aligned_ESM_ICP = performOneESM_ICPIteration(aligned_ESM_ICP, target,totalTransformation);
            // ESM_ICP.setInputSource(aligned_ESM_ICP);
            // ESM_ICP.align(*temp1);
            //*aligned_ESM_ICP =;
            
            pcl::io::savePCDFileASCII("../saved_pcds/aligned_ESM_ICP_"+ std::to_string(iteration+1) +".pcd", *aligned_ESM_ICP);
           // pcl::io::savePCDFileASCII("/media/ashu/09FDAC46654EECDA/registration_methods/ESM_ICP/saved_pcds/target.pcd", *target);
            //if (std::string(argv[2]) != "goicp") {
 
            PointCloudT::Ptr temp2(new PointCloudT);
            if (std::string(argv[2]) == "gicp") {
                auto gicp = pcl::make_shared<pcl::GeneralizedIterativeClosestPoint<PointT, PointT>>();
                gicp->setMaximumIterations(100);
                gicp->setInputSource(aligned_user);
                gicp->setInputTarget(target);
                gicp->align(*temp2);
                *aligned_user = *temp2;
                if (gicp->hasConverged()) {
               std::cout << "[GICP] Converged. Score: " << gicp->getFitnessScore() << std::endl;
               } else {
                 std::cerr << "[GICP] Warning: did NOT converge." << std::endl;
                }

               *aligned_user = *temp2;
            } else if (std::string(argv[2]) == "ndt") {
                ndt_ptr->align(*temp2);
                *aligned_user = *temp2;

                if (ndt_ptr->hasConverged()) {
                    std::cout << "✅ NDT converged with score: " << ndt_ptr->getFitnessScore() << std::endl;
                    std::cout << "Final transform:\n" << ndt_ptr->getFinalTransformation() << std::endl;
                } else {
                    std::cerr << "❌ NDT did not converge." << std::endl;
                }

            }
            
            else
            {
                user_icp->setInputSource(aligned_user);
                user_icp->setInputTarget(target);
                user_icp->align(*temp2);
                *aligned_user = *temp2;
                if(std::string(argv[2]) == "icp_nl")
                {
                   Eigen::Matrix4f T_icp_nl_current = user_icp->getFinalTransformation(); 
                   T_icp_nl_total = T_icp_nl_current.cast<double>() * T_icp_total;
                   computeTransformationErrors(T_icp_nl_total.cast<float>(), tr, "ICP_NL");
                   //T_icp_nl_total = user_icp->getFinalTransformation()          
                }
                if(std::string(argv[2]) == "point2plane")
                {
                   Eigen::Matrix4f T_icp_p2pl_current = user_icp->getFinalTransformation(); 
                   T_icp_p2pl_total = T_icp_p2pl_current.cast<double>() * T_icp_p2pl_total;
                   computeTransformationErrors(T_icp_p2pl_total.cast<float>(), tr, "ICP_P2PL");
                   //T_icp_nl_total = user_icp->getFinalTransformation()          
                }
                if(std::string(argv[2]) == "icp")
                {
                   Eigen::Matrix4f T_icp_current = user_icp->getFinalTransformation(); 
                   T_icp_total = T_icp_current.cast<double>() * T_icp_total;
                   computeTransformationErrors(T_icp_total.cast<float>(), tr, "ICP");
                   //T_icp_nl_total = user_icp->getFinalTransformation()          
                }
            }
            
           
            


            float rmse_correntropy = computeRMSE(aligned_ESM_ICP, target);
            float rmse_user = computeRMSE(aligned_user, target);
            computeTransformationErrors(totalTransformation.cast<float>(), tr, "ESM_ICP");

        //     std::cout << "[Correntropy ICP] RMSE: " << rmse_correntropy
        //               << " | [" << argv[2] << "] RMSE: " << rmse_user << std::endl;
        //   //  }            

            std::cout <<"\nIteration_number: " << ++iteration;

            
            
            viewer.updatePointCloud(aligned_ESM_ICP, pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_ESM_ICP, 255, 0, 0), "aligned1");
            pcl::io::savePCDFileASCII("../saved_pcds/aligned_"+ std::string(argv[2]) + "_" + std::to_string(iteration+1) +".pcd", *aligned_user);
            
            if (std::string(argv[2]) == "icp" || std::string(argv[2]) == "point2plane" || std::string(argv[2]) == "icp_nl" || std::string(argv[2]) == "ndt"  ) {
            viewer.updatePointCloud(aligned_user, pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_user, 255, 0, 0), "aligned2");
            }
            // Optional Go-ICP integration if selected
            if (std::string(argv[2]) == "goicp" and run_goicp == true) {
                saveAsXYZ(source, "source.xyz");
                saveAsXYZ(target, "target.xyz");
                writeGoICPConfig("source.xyz", "target.xyz");
                runGoICP();
                Eigen::Matrix4f goicp_tf = loadGoICPResult("goicpoutput.txt");
                std::cout<<"\nGoicp returned transfrom=\n" << goicp_tf <<"\n";


                PointCloudT::Ptr aligned_goicp(new PointCloudT);
                pcl::transformPointCloud(*source, *aligned_goicp, goicp_tf);
                computeTransformationErrors(goicp_tf.cast<float>(), tr, "GO_ICP");
                 pcl::io::savePCDFileASCII("../saved_pcds/aligned_goicp_"+ std::to_string(iteration+1) +".pcd", *aligned_goicp);
                viewer.removePointCloud("src2", v2);
                //auto red = pcl::visualization::PointCloudColorHandlerCustom<PointT>(aligned_goicp, 255,0,0);
                viewer.addPointCloud(aligned_goicp, red, "aligned_goicp", v2);
                viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "aligned_goicp", v2);
                run_goicp = false;
                
            }
            viewer.removePointCloud("src1", v1);
            if (std::string(argv[2]) == "icp" || std::string(argv[2]) == "point2plane" || std::string(argv[2]) == "icp_nl" || std::string(argv[2]) == "ndt"   ) viewer.removePointCloud("src2", v2);
            
           }
            perform_alignment = false;
        }
    }
    return 0;
}
