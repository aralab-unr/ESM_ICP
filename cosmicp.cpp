#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Dense>
#include <random>
#include <pcl/registration/registration.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_correntropy_svd.h>
#include <random>
#include <unordered_set>
// Global PCL point cloud objects
using namespace pcl;
pcl::PointCloud<pcl::PointXYZ>::Ptr sourceCloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud_cicp(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr alignedCloud_icp(new pcl::PointCloud<pcl::PointXYZ>);
Eigen::Matrix4d transformation_matrix_cor_svd=Eigen::Matrix4d::Identity ();

pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>::Ptr icp_cor ( new pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> () );
pcl::registration::TransformationEstimationCorrentropySVD<pcl::PointXYZ, PointXYZ>::Ptr trans_cor_svd (new pcl::registration::TransformationEstimationCorrentropySVD<PointXYZ, PointXYZ>);

int ctr=0;
bool performICP = false; // Flag to trigger ICP iteration
Eigen::Matrix4d totalTransformation = Eigen::Matrix4d::Identity(); // Track cumulative transformation

// Downsample point cloud using Voxel Grid Filter
pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leafSize) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxelFilter;
    voxelFilter.setInputCloud(cloud);
    voxelFilter.setLeafSize(leafSize, leafSize, leafSize);
    voxelFilter.filter(*cloudFiltered);
    return cloudFiltered;
}

Eigen::Matrix4d generateRandomTransformation() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Random rotation angle between -π and π
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);

    // Random axis components between -1 and 1
    std::uniform_real_distribution<double> axis_dist(-1.0, 1.0);

    // Generate random axis
    Eigen::Vector3d axis(axis_dist(gen), axis_dist(gen), axis_dist(gen));
    axis.normalize();  // Make it a unit vector

    double angle = 2.14;

    // Create rotation matrix from angle-axis
    Eigen::AngleAxisd angleAxis(angle, axis);
    Eigen::Matrix3d rotation = angleAxis.toRotationMatrix();

    // Random translation
    std::uniform_real_distribution<double> trans_dist(-0.1, 0.1);
    Eigen::Vector3d translation(0.005, 0.005, 0.005);

    // Construct 4x4 transformation matrix
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3,3>(0,0) = rotation;
    transform.block<3,1>(0,3) = translation;
    

    return transform;
}


// Apply transformation to a point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr applyTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Matrix4d& transformation) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& p : cloud->points) {
        Eigen::Vector4d pVec(p.x, p.y, p.z, 1.0);
        Eigen::Vector4d transformedP = transformation * pVec;
        transformedCloud->push_back(pcl::PointXYZ(transformedP.x(), transformedP.y(), transformedP.z()));
    }
    return transformedCloud;
}

// Find the closest point in target for each point in source
std::vector<int> findClosestPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source, const pcl::PointCloud<pcl::PointXYZ>::Ptr& target) {
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
Eigen::Matrix4d computeOptimalTransformation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
                                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
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
    double sigma = 10.0;

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

    //     std::cout<<"\n For input point = ["<< s(0,0)<<"," << s(1,0)<<","<< s(2,0)<<"] with idx = "<<i;
    //   std::cout<<" Target point = " << "idx = " << correspondences[i] << " pt = [" << t(0,0) << "," << t(1,0) << "," << t(2,0) <<"]"; 

     Eigen::MatrixXd corrval=(s-t).transpose()*(s-t);
    //std::cout<<"\nCor_Sigma="<<cor_sigma;
    //double sigma = 0.05;
    double cval=exp(-(corrval(0,0))/(2*sigma*sigma));
//    / std::cout <<" cval = " << cval;
    

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
  ss3<<ctr++;
  std::string idx=ss3.str();

  fs.open("/media/ashu/09FDAC46654EECDA/registration_methods/cosmIcp/res/corr_matrix_"+idx+".txt",std::ios::app);
  fs << corr_similarity_matrix;
  fs.close();
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
  std::cout << "\npU=" << u;
  std::cout << "\nV="<< v;
  
  if (u.determinant () * v.determinant () < 0)
  {
    for (int x = 0; x < 3; ++x)
      v (x, 2) *= -1;
  }

  Eigen::Matrix<double, 3, 3> R = v * u.transpose ();
  std::cout << "\nR=" << R;


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

// Perform one iteration of ICP
void performOneCICPIteration() {
    std::vector<int> correspondences = findClosestPoints(alignedCloud_cicp, targetCloud);
    Eigen::Matrix4d transformation = computeOptimalTransformation(alignedCloud_cicp, targetCloud, correspondences);
    alignedCloud_cicp = applyTransformation(alignedCloud_cicp, transformation);
    totalTransformation = transformation * totalTransformation;
    std::cout << "\nICP Step - Updated Transformation:\n" << totalTransformation << std::endl;
}

void performOneICPIteration() {

     icp_cor->setInputSource (alignedCloud_icp);
    icp_cor->align (*alignedCloud_icp);
    Eigen::Matrix4d transformation = icp_cor->getFinalTransformation ().inverse().cast<double>();
   // alignedCloud_icp = applyTransformation(alignedCloud_icp, transformation); 
    
    transformation_matrix_cor_svd *= icp_cor->getFinalTransformation ().inverse().cast<double>(); 
    std::cout << "\nworking CICP Step - Updated Transformation:\n" << transformation_matrix_cor_svd << std::endl;
}



// Keyboard callback function
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event, void*) {
    if (event.getKeySym() == "space" && event.keyDown()) {
        performICP = true;
    }
}

void corruptRandomSubsetWithNonGaussianNoise(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
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

// Main function
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.pcd>" << std::endl;
        return -1;
    }

    // Load input point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud);

    // Downsample the point cloud
    float leafSize = 0.04f;
    sourceCloud = downsamplePointCloud(cloud, leafSize);
    
    //sourceCloud = cloud;
    Eigen::Matrix4d randomTransform = generateRandomTransformation();
    std::cout << "\nTrnasformation matrix = \n" << randomTransform;
    targetCloud = applyTransformation(sourceCloud, randomTransform);
   // corruptRandomSubsetWithNonGaussianNoise(sourceCloud, 0.15f, 0.05f, 5.0f);
    alignedCloud_cicp = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(*sourceCloud));
    alignedCloud_icp = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>(*sourceCloud));
    

    // Create viewer
    pcl::visualization::PCLVisualizer viewer("ICP Viewer");
    viewer.setBackgroundColor(1.0, 1.0, 1.0);
    int v1 (0); int v2 (1);

    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);  // xmin, ymin, xmax, ymax
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    viewer.setBackgroundColor(1.0, 1.0, 1.0, v1); // White for left viewport
    viewer.setBackgroundColor(1.0, 1.0, 1.0, v2); // White for right viewport

    viewer.addPointCloud(sourceCloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sourceCloud, 255, 0, 0), "source_cicp",v1);
    viewer.addPointCloud(targetCloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(targetCloud, 0, 100, 0), "target_cicp",v1);
    viewer.addPointCloud(alignedCloud_cicp, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(alignedCloud_cicp, 255, 0, 0), "aligned_cicp",v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source_cicp",v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_cicp",v1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned_cicp",v1);
   
    viewer.addPointCloud(sourceCloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(sourceCloud, 255, 0, 0), "source_icp",v2);
    viewer.addPointCloud(targetCloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(targetCloud, 0, 100, 0), "target_icp",v2);
    viewer.addPointCloud(alignedCloud_icp, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(alignedCloud_icp, 255, 0, 0), "aligned_icp",v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source_icp",v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "target_icp",v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned_icp",v2);
   

    viewer.registerKeyboardCallback(keyboardEventOccurred);





// ///////////////////////ICP CORRENTROPY SVD//////////////////////////////////////////////////////////////////
  
    icp_cor->setMaximumIterations (100);
  icp_cor->setTransformationEstimation (trans_cor_svd);
  icp_cor->setInputSource (alignedCloud_icp);
  icp_cor->setInputTarget (targetCloud);
  icp_cor->setMaximumIterations(1);
  
// //////////////////////////////////////////////////////////////////////////////////////////////////////////

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
        if (performICP) {
            performOneCICPIteration();
            performOneICPIteration();
            //viewer.updatePointCloud(alignedCloud, "aligned");
            viewer.removePointCloud("aligned_cicp", v1);
            viewer.removePointCloud("aligned_icp", v2);
            viewer.removePointCloud("source_cicp", v1);
            viewer.removePointCloud("source_icp", v2);
            
            auto newColor_cicp = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(alignedCloud_cicp, 255, 0, 0); // Blue
            auto newColor_icp = pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(alignedCloud_icp, 255, 0, 0); // Blue
            
            viewer.addPointCloud<pcl::PointXYZ>(alignedCloud_cicp, newColor_cicp, "aligned_cicp", v1);
            viewer.addPointCloud<pcl::PointXYZ>(alignedCloud_icp, newColor_icp, "aligned_icp", v2);
            // Optionally update point size or other properties
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned_cicp", v1);
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "aligned_icp", v2);
            performICP = false;
        }
    }

    return 0;
}
