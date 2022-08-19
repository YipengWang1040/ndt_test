#include "ndt.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/voxel_grid_covariance.h>

#include <unordered_map>
#include <cmath>

namespace my_ndt {

struct rgb_statistics{
    Eigen::Vector3d mean_;
    Eigen::Matrix3d cov_;
    Eigen::Matrix3d icov_;
    Eigen::Matrix3d evecs_;
    Eigen::Vector3d evals_;
};

Eigen::Matrix3d euler2matrix(const Eigen::Vector3d& vec){
    const double x=vec.x();
    const double y=vec.y();
    const double z=vec.z();
    using std::cos;
    using std::sin;
    Eigen::Matrix3d ret;
    ret<<   cos(y)*cos(z), -cos(y)*sin(z), sin(y),
            cos(x)*sin(z)+sin(x)*sin(y)*sin(z), cos(x)*cos(z)-sin(x)*sin(y)*sin(z), -sin(x)*cos(y),
            sin(x)*sin(z)-cos(x)*sin(y)*cos(z), cos(x)*sin(y)*sin(z)+sin(x)*cos(z), cos(x)*cos(y);
}

Eigen::Matrix<double, 9,3> prepare_dr_dx(const Eigen::Vector3d& euler_angles){
    const double x=euler_angles.x();
    const double y=euler_angles.y();
    const double z=euler_angles.z();
    using std::cos;
    using std::sin;
    Eigen::Matrix<double, 9,3> ret;
    const double sx=sin(x);
    const double sy=sin(y);
    const double sz=sin(z);
    const double cx=cos(x);
    const double cy=cos(y);
    const double cz=cos(z);
    // eq. 6.19
    ret<<   0,               0,               0,
            -sx*sz+cx*sy*cz, -sx*cz-cx*sy*sz, -cx*cy,  // a
            cx*sz+sx*sy*cz,  -sx*sy*sz+cx*cz, -sx*cy,  // b
            -sy*cz,          sy*sz,           cy,      // c
            sx*cy*cz,        -sx*cy*sz,       sx*sy,   // d
            -cx*cy*cz,       cx*cy*sz,        -cx*sy,  // e
            -cy*sz,          -cy*cz,          0,       // f
            cx*cz-sx*sy*sz,  -cx*sz-sx*sy*cz, 0,       // g
            sx*cz+cx*cy*sz,  cx*sy*cz-sx*sz,  0;       // h

    return ret;

}

// verticaly are 3 by 1 vectors a~f after multiplication, according to eq. 6.21
Eigen::Matrix<double, 18,3> prepare_d2r_dx2(const Eigen::Vector3d& euler_angles){
    const double x=euler_angles.x();
    const double y=euler_angles.y();
    const double z=euler_angles.z();
    using std::cos;
    using std::sin;
    Eigen::Matrix<double, 18,3> ret;
    const double sx=sin(x);
    const double sy=sin(y);
    const double sz=sin(z);
    const double cx=cos(x);
    const double cy=cos(y);
    const double cz=cos(z);
    // eq. 6.21
    ret<<   0,               0,               0,
            -cx*sz-sx*sy*cz, -cx*cz+sx*sy*sz, sx*cy,
            -sx*sz+cx*sy*cz, -cx*sy*sz-sx*cz, -cx*cy,   // a

            0,               0,               0,
            cx*cy*cz,        -cx*cy*sz,       cx*sy,
            sx*cy*cz,        -sx*cy*sz,       sx*sy,    // b

            0,               0,               0,
            -sx*cz-cx*sy*sz, -sx*sz-cx*sy*cz, 0,
            cx*cz-sx*sy*sz,  -sx*sy*cz-cx*sz, 0,        // c

            -cy*cz,          cy*sz,           -sy,
            -sx*sy*cz,       sx*sy*sz,        sx*cy,
            cx*sy*cz,        -cx*sy*sz,       -cx*cy,   // d

            sy*sz,           sy*cz,           0,
            -sx*cy*sz,       -sx*cy*cz,       0,
            cx*cy*cz,        cx*cy*cz,        0,        // e

            -cy*cz,          cy*sz,           0,
            -cx*sz-sx*sy*cz, -cx*cz+sx*sy*sz, 0,
            -sx*sz+cx*sy*cz, -cx*sy*sz-sx*cz, 0;        // f




    return ret;
}

Result ndt_matching(ConstPtr target_cloud, ConstPtr input_cloud, const Eigen::Matrix4d& initial_guess, const Config& config){
    Result ret;

    // initialization

    // assign grids for input cloud, compute the statistics
    Ptr filtered_cloud(new pcl::PointCloud<PointT>);
    pcl::VoxelGridCovariance<PointT> voxel_grid_convariance;
    voxel_grid_convariance.setLeafSize(config.leaf_size,config.leaf_size,config.leaf_size);
    voxel_grid_convariance.setInputCloud(input_cloud);
    voxel_grid_convariance.filter(true);

    // todo:
    // calculate rgb statistics for each grid
//    std::unordered_map<pcl::VoxelGridCovariance<PointT>::Leaf*, rgb_statistics> rgb_information;
//    std::unordered_map<pcl::VoxelGridCovariance<PointT>::Leaf*, std::vector<PointT>> rgb_buffer;

//    for(auto&& pt:input_cloud->points){

//    }

    const double c1=10*(1-config.outlier_ratio);
    const double c2=config.outlier_ratio/std::pow(config.resolution,3);
    const double d3=-std::log(c2);
    const double d1=-std::log(c1+c2)-d3;
    const double d2=-2*std::log((-std::log(c1*std::exp(-0.5)+c2)-d3)/d1);


    bool converged=false;

    // pack initial p vector
    Eigen::Matrix<double,6,1> p;
    p.head(3)=initial_guess.block<3,1>(0,3);
    p.tail(3)=initial_guess.block<3,3>(0,0).eulerAngles(0,1,2);

    double score=0;
    size_t iteration=0;

    while(!converged){
        score=0;
        // Corresponds to eq. 6.18 and eq. 6.20
        Eigen::Matrix<double,3,6> J_E=Eigen::Matrix<double,3,6>::Zero();
        J_E.block<3,3>(0,0)=Eigen::Matrix3d::Identity();
        Eigen::Matrix<double,18,6> H_E=Eigen::Matrix<double,18,6>::Zero();
        // According to eq. 6.19 and eq. 6.21, right half of J_E can be written as x multiplied by a matrix not relevent to x, so we pre-calculate the matrix
        Eigen::Matrix<double, 9,3> dr_dx=prepare_dr_dx(p.tail(3));
        // So does matrix H_E
        Eigen::Matrix<double,18,3> d2r_dx2=prepare_d2r_dx2(p.tail(3));

        Eigen::Matrix4d transform=Eigen::Matrix4d::Identity();
        transform.block<3,3>(0,0)=euler2matrix(p.tail(3));
        transform.block<3,1>(0,3)=p.head(3);

        Eigen::Matrix<double,6,1> g=Eigen::Matrix<double,6,1>::Zero();
        Eigen::Matrix<double,6,6> H=Eigen::Matrix<double,6,6>::Zero();

        for(auto&& pt:target_cloud->points){
            // find cell b_i
            Eigen::Vector3d point=pt.getArray3fMap().cast<double>();
            point=transform.block<3,3>(0,0)*point+transform.block<3,1>(0,3);
            Eigen::Vector3f pointf=point.cast<float>();
            pcl::VoxelGridCovariance<PointT>::LeafConstPtr leaf=voxel_grid_convariance.getLeaf(pointf);

            // score=score+p
            Eigen::Vector3d x_prime=point-leaf->getMean();
            double exp_base=std::exp(-d2/2*x_prime.transpose()*leaf->getInverseCov()*x_prime);
            score+=-d1*exp_base;

            // update g and H
            // column vector contains 0,a,b,...,h in eq.6.18
            Eigen::Matrix<double,9,1> col_je=dr_dx*x_prime;
            J_E.block<3,3>(0,3)=Eigen::Map<Eigen::Matrix3d>(col_je.data());

            // column vector contains a,b,...,f in eq.6.20
            Eigen::Matrix<double,18,1> col_he=d2r_dx2*x_prime;
            H_E.block<3,1>(9,3)=col_he.block<3,1>(0,0);   // a
            H_E.block<3,1>(9,4)=col_he.block<3,1>(3,0);   // b
            H_E.block<3,1>(9,5)=col_he.block<3,1>(6,0);   // c
            H_E.block<3,1>(12,3)=col_he.block<3,1>(3,0);  // b
            H_E.block<3,1>(12,4)=col_he.block<3,1>(9,0);  // d
            H_E.block<3,1>(12,5)=col_he.block<3,1>(12,0); // e
            H_E.block<3,1>(15,3)=col_he.block<3,1>(6,0);  // c
            H_E.block<3,1>(15,4)=col_he.block<3,1>(12,0); // e
            H_E.block<3,1>(15,5)=col_he.block<3,1>(15,0); // f

            for(long i=0;i<6;++i){
                g(i)+=d1*d2*x_prime.transpose().dot(leaf->getInverseCov()*J_E.col(i))*exp_base;
                for(long j=0;j<6;++j){
                    H(i,j)=d1*d2*exp_base*(-d2*(x_prime.transpose().dot(leaf->getInverseCov()*J_E.col(i)))
                                           *(x_prime.transpose().dot(leaf->getInverseCov()*J_E.col(j)))
                                           +x_prime.transpose().dot(leaf->getInverseCov()*H_E.block<3,1>(3*i,j))
                                           +J_E.col(j).transpose().dot(leaf->getInverseCov()*J_E.col(i)));
                }
            }
        }

        // solve H*delta_p=-g
        Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(H,Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix<double, 6, 1> delta_p = sv.solve(-g);
        p=p+delta_p;

        iteration+=1;
        if(iteration>config.iteration)
            converged=true;

        double cos_angle=(euler2matrix(p.tail(3)).trace()-1)/2;
        double squared_dist=p.head(3).squaredNorm();
        if(config.epsilon>cos_angle && config.epsilon>squared_dist)
            converged=true;

    }

    ret.transform=Eigen::Matrix4d::Identity();
    ret.transform.block<3,3>(0,0)=euler2matrix(p.tail(3));
    ret.transform.block<3,1>(0,3)=p.head(3);
    ret.iteration=iteration;

    return ret;
}


} // namespace my_ndt
