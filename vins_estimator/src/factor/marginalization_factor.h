/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

// 参与边缘化的因子项
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks; // 优化变量数据
    std::vector<int> drop_set; // 待marg的优化变量ID

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals; // 残差，IMU：15X1；视觉：2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// 起到边缘化管家的作用
class MarginalizationInfo
{
  public:
    MarginalizationInfo(){valid = true;};
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info); // 添加残差块的相关信息（优化变量、待marg的变量）
    void preMarginalize(); // 得到每次IMU和视觉观测(cost_function)对应的参数块(parameter_blocks)，雅克比矩阵(Jacobins)，残差值(residuals)
    void marginalize(); // 构建Hessian矩阵，Schur掉需要marg的变量，得到剩余变量的约束，即为边缘化约束（先验约束）
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; // 和需要边缘化的帧所有相关的观测值：包括视觉、IMU、外参以及上一个先验等
    int m, n;// m：需要边缘化掉的变量的总维度；// n：需要保留的变量的总维度
    std::unordered_map<long, int> parameter_block_size; //global size：各个优化变量的维度
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size：各个优化变量在H矩阵中的ID
    std::unordered_map<long, double *> parameter_block_data; // 各个优化变量在H矩阵中的数据

    std::vector<int> keep_block_size; //global size ：边缘化之后，保留下来的各个优化变量的维度
    std::vector<int> keep_block_idx;  //local size ：边缘化之后，保留下来的各个优化变量在H矩阵中的ID
    std::vector<double *> keep_block_data; // 边缘化之后，保留下来的各个优化变量在H矩阵中的数据

    Eigen::MatrixXd linearized_jacobians; // 边缘化之后从信息矩阵恢复出来的雅克比矩阵
    Eigen::VectorXd linearized_residuals; // 边缘化之后从信息矩阵恢复出来的残差向量
    const double eps = 1e-8;
    bool valid;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
