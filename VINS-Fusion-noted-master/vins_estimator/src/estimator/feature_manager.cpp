/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {
        it.used_num = it.feature_per_frame.size();
        if (it.used_num >= 4)
        {
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief   把特征点放入feature的list容器中，即建立特征点和观测帧之间的联系;
 *          计算每一个点跟踪次数和它在次新帧和次次新帧间的视差，返回次新帧是否是关键帧
 * @param[in]   frame_count 窗口内帧的个数
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   td IMU和cam同步时间差
 * @return  bool true：次新帧是关键帧;false：非关键帧
**/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;
    /* Step 1、遍历当前帧中所有的特征点
                如果是第一次观测到的特征点，加入到feature大家庭中
                如果是已知点，只需要加入新的观测帧即可
    */         
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 左目的结果
        assert(id_pts.second[0].first == 0); // 检测该帧是不是左目ID == 0
        // 如果右目也有追踪结果
        if(id_pts.second.size() == 2)
        {
            // 加入右侧的观测结果
            f_per_fra.rightObservation(id_pts.second[1].second);
            assert(id_pts.second[1].first == 1);
        }

        int feature_id = id_pts.first; // 当前特征点的ID
        // 看看这个角点是不是新的角点
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });
        // 如果他是新的角点，即返回得是end()，没有找到一样的角点
        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));
            feature.back().feature_per_frame.push_back(f_per_fra);
            new_feature_num++;
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
            if( it-> feature_per_frame.size() >= 4)
                long_track_num++;
        }
    }

    // Step 2、通过视差或者特征点跟踪次数等条件判断次新帧是否是关键帧
    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    /*
        frame_count < 2：滑窗中仅一帧
        last_track_num < 20：在当前帧中追踪到之前的大于20个特征点
        long_track_num < 40：观测次数超过4次的特征点数量 < 40
        new_feature_num > 0.5 * last_track_num:新提取的特征点数量超过在当前帧中被跟踪到的旧特征点的一半
    */
    // 满足上述任一条件，即认为次新帧为关键帧
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;

    // 如果上述条件都不满足，那么就计算次新帧和次次新帧之间的视差
    for (auto &it_per_id : feature)
    {
        /*
            it_per_id.start_frame <= frame_count - 2 : 观测到该点的第一帧至少是当前帧的前两帧
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1：目前最后一次观测到该点的帧ID >= 当前帧的上一帧
            这样保证了该点至少有两帧观测次数，并且次新帧一定观测到了该特征点
        */
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {   
            // 实际就是特征点在不同归一化相机坐标系下的像素距离
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // 说明所有点都是第一次观测到，认为该帧是关键帧
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        last_average_parallax = parallax_sum / parallax_num * FOCAL_LENGTH;
        // 如果次新帧和次次新帧之间的平均视差超过了规定阈值，认为次新帧是关键帧，在滑窗中保留
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

/**
 * @brief 找到两帧之间的共视点
 * @param[in] frame_count_l
 * @param[in] frame_count_r 输入需要搜寻的帧
 * @param[out] corres 共视点对
 **/
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        // 判断输入帧数是否在观测到该点的帧数范围内
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;     // 与观测到的第一帧的间隔帧数
            int idx_r = frame_count_r - it.start_frame;

            // it.start_frame记录第一次观测到该点的帧ID，
            // it.feature_per_frame记录是之后所有观测到该点的帧ID
            // idx_l = frame_count_l - it.start_frame是计算frame_count_l距离第一次观测到该点的帧的间隔，从而定位到该点在feature_per_frame中的其他观测值
            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

// 将长期跟踪但是深度仍未成功恢复的点删除
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth()
{
    for (auto &it_per_id : feature)
        it_per_id.estimated_depth = -1;
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}


void FeatureManager::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                        Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool FeatureManager::solvePoseByPnP(Eigen::Matrix3d &R, Eigen::Vector3d &P, 
                                      vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D)
{
    Eigen::Matrix3d R_initial;
    Eigen::Vector3d P_initial;

    // w_T_cam ---> cam_T_w 
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    //printf("pnp size %d \n",(int)pts2D.size() );
    if (int(pts2D.size()) < 4)
    {
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, 1);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ)
    {
        printf("pnp failed ! \n");
        return false;
    }
    cv::Rodrigues(rvec, r);
    //cout << "r " << endl << r << endl;
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);

    return true;
}

/**
 * @brief 通过上一帧的Ps[frameCnt-1]、Rs[frameCnt-1]以及当前帧看到了所有恢复了深度的2D-3D，通过PnP确定Ps[frameCnt]、Rs[frameCnt]
 * 
 * @param frameCnt 当前滑窗中的帧ID
 * @param Ps 滑窗中Ps_w_bk
 * @param Rs 滑窗中Rs_w_bk
 * @param tic 外参：R_b_c
 * @param ric 外参：t_b_c
 */
void FeatureManager::initFramePoseByPnP(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    if(frameCnt > 0)
    {
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        // 遍历当前所有的特征点
        for (auto &it_per_id : feature)
        {
            // 由于至少是双目，因此有一些点可以根据双目来进行三角化
            if (it_per_id.estimated_depth > 0)
            {
                int index = frameCnt - it_per_id.start_frame;   // 计算在数组中的索引值
                // pnp为了求当前的位姿，因此得保障当前这个特征点能被当前帧所看到
                if((int)it_per_id.feature_per_frame.size() >= index + 1)
                {
                    // 将相机坐标系转到imu坐标系
                    Vector3d ptsInCam = ric[0] * (it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth) + tic[0];
                    // 将imu坐标系转到世界坐标系
                    Vector3d ptsInWorld = Rs[it_per_id.start_frame] * ptsInCam + Ps[it_per_id.start_frame];
                    // 得到3d->2d这样的对应关系
                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    cv::Point2f point2d(it_per_id.feature_per_frame[index].point.x(), it_per_id.feature_per_frame[index].point.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d); 
                }
            }
        }
        Eigen::Matrix3d RCam;
        Eigen::Vector3d PCam;
        // trans to w_T_cam
        // 上一帧的相机坐标系坐标，作为初值
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];
        // 解pnp
        if(solvePoseByPnP(RCam, PCam, pts2D, pts3D))
        {
            // trans to w_T_imu
            // 从相机坐标系转到imu坐标系下
            Rs[frameCnt] = RCam * ric[0].transpose(); 
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;

            Eigen::Quaterniond Q(Rs[frameCnt]);
            //cout << "frameCnt: " << frameCnt <<  " pnp Q " << Q.w() << " " << Q.vec().transpose() << endl;
            //cout << "frameCnt: " << frameCnt << " pnp P " << Ps[frameCnt].transpose() << endl;
        }
    }
}

/**
 * @brief 恢复当前所有特征点的深度
 * 
 * @param frameCnt 帧数
 * @param Ps 每帧的位置
 * @param Rs 每帧的姿态
 * @param tic 外参-平移
 * @param ric 外参-旋转
 */
void FeatureManager::triangulate(int frameCnt, Vector3d Ps[], Matrix3d Rs[], Vector3d tic[], Matrix3d ric[])
{
    // 遍历每个特征点
    for (auto &it_per_id : feature)
    {
        // 如果该特征点深度>0，即该特征点的状态认为是好的，那么就跳过
        if (it_per_id.estimated_depth > 0)
            continue;
        // 如果是双目并且两个相机都能看到同一个点，那就根据两个相机的外参去进行三角化
        if(STEREO && it_per_id.feature_per_frame[0].is_stereo)
        {
            int imu_i = it_per_id.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            // 把位姿从imu坐标系转换到相机坐标系
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            // Twc -> Tcw，左目的pose
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;
            //cout << "left pose " << leftPose << endl;
            // 同理得到右目的pose
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[1];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[1];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;
            //cout << "right pose " << rightPose << endl;

            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[0].pointRight.head(2);
            //cout << "point0 " << point0.transpose() << endl;
            //cout << "point1 " << point1.transpose() << endl;
            // 三角化出点的深度
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            // 在左目相机坐标系下的坐标
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH; // 代表三角化过，但是失败了
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        // 如果是单目或者双目只有一个相机看到，那就需要同一个相机不同位姿的观测来三角化
        else if(it_per_id.feature_per_frame.size() > 1)
        {
            int imu_i = it_per_id.start_frame;
            Eigen::Matrix<double, 3, 4> leftPose;
            // 取出起始帧的位姿
            Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
            leftPose.leftCols<3>() = R0.transpose();
            leftPose.rightCols<1>() = -R0.transpose() * t0;

            imu_i++; // 就直接取起始帧下一帧的位姿
            Eigen::Matrix<double, 3, 4> rightPose;
            Eigen::Vector3d t1 = Ps[imu_i] + Rs[imu_i] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_i] * ric[0];
            rightPose.leftCols<3>() = R1.transpose();
            rightPose.rightCols<1>() = -R1.transpose() * t1;
            // 下面就是三角化出位姿
            Eigen::Vector2d point0, point1;
            Eigen::Vector3d point3d;
            point0 = it_per_id.feature_per_frame[0].point.head(2);
            point1 = it_per_id.feature_per_frame[1].point.head(2);
            triangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Eigen::Vector3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                it_per_id.estimated_depth = depth;
            else
                it_per_id.estimated_depth = INIT_DEPTH;
            /*
            Vector3d ptsGt = pts_gt[it_per_id.feature_id];
            printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                            ptsGt.x(), ptsGt.y(), ptsGt.z());
            */
            continue;
        }
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        // 貌似下面就进不来了吧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

/**
 * @brief 移除外点
 * 
 * @param outlierIndex 外点索引
 * @return ** void 
 */
void FeatureManager::removeOutlier(set<int> &outlierIndex)
{
    std::set<int>::iterator itSet;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end())
        {
            feature.erase(it);
            //printf("remove outlier %d \n", index);
        }
    }
}

/**
 * @brief 遍历当前所有特征点
 *        Step 1、由于边缘化最旧帧，所以第一帧观测帧非第一帧的 start_frame 减一
 *        Step 2、如果以第一帧观测帧是最旧帧：该点性能不好，直接删除；该点性能好，删除第一次观测帧，并更新深度
 * 
 * @param marg_R 
 * @param marg_P 
 * @param new_R 
 * @param new_P 
 * @return ** void 
 */
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    // 遍历当前所有特征点
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        // Step 1、由于边缘化最旧帧，所以第一帧观测帧非第一帧的 start_frame 减一
        if (it->start_frame != 0)
            it->start_frame--;
        // Step 2、如果以第一帧观测帧是最旧帧
        else
        {
            // uv_i是在旧帧上的该点基于第一次观测帧上的归一化相机坐标
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            // 将第一次观测帧删除
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            // 如果该点的观测次数 < 2，直接删除该点
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            // 否则就更新该点的深度
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

// 在初始化阶段使用
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

// 全阶段使用
void FeatureManager::removeFront(int frame_count)
{
    // 遍历所有特征点
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        // 如果该点第一次观测帧为最新帧，因为最新帧要前移，所以start_frame减一
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        // 如果该点第一次观测帧不是最新帧
        else
        {
            // j是次新帧离start_frame的帧数
            int j = WINDOW_SIZE - 1 - it->start_frame;
            // 如果 该点的所有观测帧数 < j 说明该点的观测帧不包含次新帧
            if (it->endFrame() < frame_count - 1)
                continue;
            // 如果包含就删除次新帧的观测
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * @brief 计算某特征点在观测到它的次新帧和次次新帧上的距离（归一化相机平面上）
 * 
 * @param it_per_id 
 * @param frame_count 
 * @return ** double 
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    // frame_i：次新帧；frame_j：次次新帧
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    // 疑惑：depth肯定为1，除和不除的区别是什么？？？
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}