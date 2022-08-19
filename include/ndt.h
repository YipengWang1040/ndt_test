#ifndef MY_NDT_H
#define MY_NDT_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


using PointT=pcl::PointXYZ;
using Ptr=pcl::PointCloud<PointT>::Ptr;
using ConstPtr=pcl::PointCloud<PointT>::ConstPtr;

namespace my_ndt{

struct Config{
    double epsilon=0.01;
    double step_size=0.1;
    double resolution=1.0;
    double outlier_ratio=0.5;
    double leaf_size=0.4;
    size_t iteration=50;
};

struct Result{
    Eigen::Matrix4d transform;
    int iteration;
};

Result ndt_matching(ConstPtr target_cloud, ConstPtr input_cloud, const Eigen::Matrix4d& initial_guess, const Config& config);

}





#endif //MY_NDT_H
