/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    // 如果是多线程模式，就是一个线程做光流，一个线程做后端优化
    // 否则，就是一个做完光流之后在做线程优化,串行处理
    // 在vinsfusion有两种获取数据的方式，一种是euroc数据集一样使用rosbag包来获取，通常这种情况下对实时性要求比较高
    // 另一种是kitti数据集一样读取离线数据，此时对实时性要求就比较低
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}


/**
 * @brief   在该函数中集成了预积分、跟踪、后端优化等重要函数
 * @param[in] t 时间戳 
 * @param[in] _img 左目帧
 * @param[in] _img 右目帧
 * */
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    // Step 1、帧总数+1，并对新帧进行特征点的光流跟踪以及新特征点的提取
    inputImageCnt++; // 输入帧总数加1
    /**
    *@brief 用map构建featureFrame，用于记录该帧中提取到的特征点信息
    *从左到右，依次对变量含义进行解释
    *int   特征点ID
    *vector<pair<int, Eigen::Matrix<double, 7, 1>>> int：0：左目/1：右目；
    *Eigen::Matrix<double, 7, 1> 
    **/
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    // 右目图像为空，即单目
    if(_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img);
    else // 双目
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }
    // 如果是多线程模式就做一下降采样
    if(MULTIPLE_THREAD)  
    {     
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
}

// 两个作用，一个是送入把imu数据送入buffer，另一个是输出高频率里程计
void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if(!MULTIPLE_THREAD)
        processMeasurements();
}

/**
 * @brief 
 * @param[in] t0 上一图像帧对应的时间戳
 * @param[in] t1 当前图像帧对应的时间戳
 * @param[in&out] accVector、gyrVector：记录两幅图像帧之间的IMU数据（线加速度、角加速度）
*/
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)                                       // 如果当前帧时间t1 <= 加速度容器中最后一个数据的时间戳
    {
        while (accBuf.front().first <= t0)                         // 如果加速度容器中第一个数据时间戳小于上一图像帧的时间，那么将小于部分的IMU数据pop
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        /*
            |                                                       |
        prev                                                cur
              |    |    |   |   |   |   |   |   |   |   |   |   |   |
     imu_first                                               imu_back
        */
        while (accBuf.front().first < t1)                           
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

/**
 * @brief 判断是否有IMU且IMU是否可用 
 **/
bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)     // 如果加速度容器不为空，且当前时间t<=加速度容器中第一个速度的时间
        return true;  // 那么该IMU数据可用
    else
        return false;
}

/**
 * @brief 
 * 
 * @return ** void 
 */
void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature; // 特征点容器
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector; // 两帧图像之间的加速度、角速度容器
        if(!featureBuf.empty())
        {
            // Step 1、取出当前帧光流追踪以及提取的结果
            feature = featureBuf.front();
            curTime = feature.first + td; // td：是时间补偿，秦通博士处理：将td认为是一个常值（在极短时间内是不变化的）
                                          // 由于触发器等各种原因，IMU和图像帧之间存在时间延迟，因此需要进行补偿
                                          // 详见Online_Temporal_Calibration_for_Monocular_Visual-Inertial_Systems 
            // Step 2、等到合适的IMU数据
            while(1)
            {
                // 等待IMU数据来全，当然没有IMU就算了
                if ((!USE_IMU  || IMUAvailable(feature.first + td))) // USE_IMU = true（系统使用IMU）且IMU数据容器中有可用的数据
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            mBuf.lock(); 
            if(USE_IMU)
                // 得到两帧之间的IMU数据
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();   // 已经取出，就pop掉                   
            mBuf.unlock();
            
            // TODO：为什么不把等待IMU数据的过程放在USE_IMU中？？？——可能是因为，即使不使用，也照常接受IMU数据
            // 如果使用IMU
            /*
                Step 3、若使用IMU——IMU积分
                        若还是第一帧：因为IMU不是水平放置，所以Z轴和{0, 0, 1.0}对齐，通过对齐获得Rs[0]的初始位姿
            */ 
            if(USE_IMU)
            {
                // 如果是第一帧，就根据IMU重力的方向估计出大概的姿态
                // 即使不适用IMU，也照常接受数据
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    // IMU积分处理
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();
            // Step 4、图像处理
            processImage(feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

/**
 * @brief 因为IMU不一定是水平放置，那么IMU的Z轴和{0, 0, 1.0}不对齐
 *        通过对齐，得到 g 到 {0, 0, 1.0} 的旋转（消除绕Z轴的旋转），作为初始位姿Rs[0]
 * @param accVector 加速度和g在短时间是紧耦合的，但是短时间内g占accVector的大部分值，可以通过accVector求得g的估计值
 * @return ** void 
 */
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;  //  取目前为止所有加速度测量值的平均。认为在开始的一段时间内加速度中重力占比大，基本由重力组成
                            // 很多值的累加，导致重力比例更大，即可以忽略其他因素的影响。用这个平均值来代替重力，用于估计世界坐标系到重力方向的转换
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc); // 将IMU的Z轴与重力方向的对齐转换初值
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0; // 用作R的初值，在一开始就将IMU的Z轴和重力方向对齐
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};  // 构建滑窗内的IMU预积分类
    }
    if (frame_count != 0)   // 不是第一帧
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);    // 计算上一图像帧到当前时刻的IMU积分
        //if(solver_flag != NON_LINEAR)
            // TODO：tmp_pre_integration为什么重复计算积分值？？？
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);  

        // 记录当前滑窗中的IMU数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        /**
         * ATTENTION：当两图像帧之间的IMU数据没有处理完成时，j = frame_count是不会改变的
         * 所以Rs[j]、Ps[j]、Vs[j]的值是随着dt慢慢叠加更新的
        */
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
}

/**
 * @brief 
 * 
 * @param image 
 * @param header 
 * @return ** void 
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // Step 1、判断次新帧是否是关键帧，同时完成特征点与帧之间关系的建立
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        // 认为次新帧是关键帧，边缘化最旧帧
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        // 认为次新帧不是关键帧，边缘化次新帧
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    // 使用后，重新建立新的IMU积分对象，以便下一帧的IMU数据继续存在这里面
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // Step 2、进行camera到IMU(body)外参的标定
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    // Step 3、如果没有初始化，先完成初始化。
    /*
    初始化根据传感器类型分为三个模式：
    模式一：单目 + IMU
    模式二：双目 + IMU
    模式三：纯双目
    */
    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        // 这是单目+imu的模式
        if (!STEREO && USE_IMU)
        {
            // TODO：为什么单目+IMU模式要等到11帧之后再初始化？
            // 原因1：猜想：相比较双目，单目的尺度不确定性，需要多帧及特征点恢复
            if (frame_count == WINDOW_SIZE)                                                // WINDOW_SIZE = 10，说明至少有10帧的图像以及相应的IMU数据
            {
                bool result = false;
                // Step A：进行单目 + IMU 的初步初始化
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)          // (header - initial_timestamp) > 0.1，上一次初始化距离这次初始化的时间超过了0.1s
                {
                    result = initialStructure();
                    initial_timestamp = header;   
                }
                // Step B：结果的再优化
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        // 双目+IMU
        if(STEREO && USE_IMU)
        {
            // Step A、使用pnp求解frame_count帧的位姿
            // 注意这里的frame_count是从第一帧开始计算，而不是想单目中得到frame_count >= 11之后才开始初始化
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            // Step B、三角化一些没有三角化的地图点
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            // Step C、
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    // 找跟pnp求出来的位姿的对应关系
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                // 陀螺仪零偏初始化
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        // 纯双目视觉
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            // 那就连陀螺仪零偏都不用估计了
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    // Step 4、若初始化成功，则进行后端优化
    else
    {
        TicToc t_solve;
        // Step A、没有使用IMU，需要先通过上一帧的旋转平移和相应的观测给出当前帧相对运动的处置
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        // Step B、每次都需要特征点三角化。这里三角化的主要原因是恢复之前没有深度没有成功恢复的点
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        // Step C、！！！关键函数！！！后端非线性优化
        optimization();
        // Step D、将重投影误差过大的点删除
        set<int> removeIndex;
        // 计算得到重投影误差过大的点
        outliersRejection(removeIndex);
        // 从feature中移除外点
        f_manager.removeOutlier(removeIndex);
        // Step E、如果不是多线程（因为多线程会降采样），那么可以从前端中的pre_cts中移除外点，避免再次跟踪；
        //          并预测预测下一帧上特征点的像素坐标
        if (! MULTIPLE_THREAD)
        {
            // 通知前端feature tracker移除这些
            featureTracker.removeOutliers(removeIndex);
            // 通过匀速模型预测下一帧上特征点的像素坐标
            predictPtsInNextFrame();
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        // Step F、系统故障检测
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        // Step G、滑窗更新
        slideWindow();
        // Step H、将长期跟踪但是深度仍未成功恢复的点删除
        f_manager.removeFailures();
        // prepare output of VINS
        // Step I、将某些量进行更新
        key_poses.clear(); // 关键帧位姿更新
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }  
}

/**
 * @brief 进行单目 + IMU 的初始化
 *        Step 1 ：纯视觉SFM初始化
 *        Step 2 ：视觉和I惯性的对齐
 * @param[out] true
 * @param[out] false
*/
bool Estimator::initialStructure()
{
    // Step 1 ：纯视觉SFM初始化
    TicToc t_sfm;
    //check imu observibility
    // 检查IMU的可观性
    // Step 11 ：保证IMU数据可用
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 遍历目前为止所有图像之间的IMU预积分
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;    // 当前两帧之间的时间间隔
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;   // 当前两帧之间的预积分delta_v
            sum_g += tmp_g;  // 累计求和delta_v = v_i+1 - v_i + g*dt
        }                                                                                                                                               // sum_g = [(v_1 - v_0 + g*dt) + (v_2 - v_1 + g*dt) + ... + (v_i+1 - v_i + g*dt)]/dt = v_i+1/dt - v_0/dt + n*g
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);   // aver_g 约等于 g
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;     // tmp_g =  (v_i+1)/dt - v_i/dt + g 约等于 g
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g); //  每个tmp_g和aver_g的差值的平方
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));  // 均方根
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // Step 12：在历史帧找到和当前帧（即最新帧）有足够关系（共视点对数>20）且拥有足够视差的帧l，并恢复出l帧到最新帧的相对运动
    //          如果相对运动恢复失败，结束本次初始化；
    //          如果相对运动恢复成功, 继续进行SFM初始化工作
    // global sfm
    // 全局SFM
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    /*
    // 记录用于SFM的特征点
    struct SFMFeature
    {
        bool state;
        int id;
        vector<pair<int,Vector2d>> observation;
        double position[3];
        double depth;
    };
    */
    vector<SFMFeature> sfm_f;
    // 遍历f_manager中的所有点
    for (auto &it_per_id : f_manager.feature)                                                                   
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;  // 当前点ID
        // 遍历所有观测到当前点的图像帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // relativePose()中找到和最新帧拥有足够共视点对的帧l，并恢复帧l和最新帧之间的相对变换
    if (!relativePose(relative_R, relative_T, l))
    {
        // 如果滑窗中，没有帧与最新帧有足够的共视关系用于恢复相对运动，则恢复失败
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    // 进行SFM
    if(!sfm.construct(frame_count + 1, 
                        Q,
                        T,
                        l,
                        relative_R,
                        relative_T,
                        sfm_f,
                        sfm_tracked_points))
    {
        // 如果SFM中BA失败，则初始化失败，并且边缘化最旧帧
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // Step 13   ：根据SFM恢复出的特征点深度，即3D位置；使用PnP恢复所有帧的位姿
    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    // 遍历目前为止的所有图像帧
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i]) //  当前帧和Headers中对应位置上的时间戳一致
        {
            frame_it->second.is_key_frame = true;   //  认为该帧是关键帧
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();  // RIC：表示从相机系到body系的转换
                                                                                //  从body坐标系转换到相机坐标系下
            frame_it->second.T = T[i];          // ！！！:平移是一样的
            i++;                                // 并且i++
            continue;                           // 退出该层循环，直接进入到下一层循环
        }
        if((frame_it->first) > Headers[i])      // 如果当前帧   >   Headers[i]
        {
            i++;                                // TODO：为什么不判断 i < WINDOW_SIZE = 11？？？难道说一定在11帧之内初始化成功吗？？？？？？？
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];  
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历该帧图像中的所有特征点
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;          // 记录该特征点的ID
            for (auto &i_p : id_pts.second)    // 遍历观测到该特征点的所有观测值，并使用pts_3_vector、pts_2_vector记录
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))   // 根据pts_3_vector、pts_2_vector中记录的3D、2D点信息
        {                                                                   // 恢复相对位姿
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // Step 2：视觉-惯性对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

/**
 * @brief 视觉-惯性对齐
 * 
 * @return true 
 * @return false 
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // ATTENTION:Bgs是空的
    // Step 1 ：通过视觉和IMU对齐，完成陀螺仪偏置的校正并初始化当前所有帧的速度以及重力和尺度因子
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // Step 2 ：根据更新的Bgs、Velocity、Gravity、Scale重新计算当前滑窗中位姿
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    // 偏置校正
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 尺度恢复
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    // 速度校正
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    // 恢复了第一帧相机坐标系下的g_c0，通过g_c0旋转就能得到相机坐标系Z轴到世界坐标系Z轴的旋转
    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    // 并将所有量都恢复至世界坐标系中
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    // ATTENTION：将SFM恢复的特征点深度重置，用修复后的pose重新进行三角化
    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

/**
 * @brief Step 1 ：寻找滑窗中历史帧和最新帧之间的共视点对
 *        Step 2 ：如果共视点对的数量满足要求，计算两帧之间的视差
 *        Step 3 ：如果均满足要求，调用solveRelativeRT()恢复当前帧到第l帧的相对运动R、t
 * @param[in&out] relative_R
 * @param[in&out] relative_T
 * @param[in&out] l
 * @param[out] bool true: false:
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 找到与最新帧拥有足够联系以及视差的历史帧
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);    // 当前滑窗中所有历史帧与最新帧的共视关系
        // 只要corres数量超过20且视差满足要求，就退出遍历
        if (corres.size() > 20) //  共视点对超过20对
        {
            // 计算两帧之间的视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();   // 每一共视点对之间的距离
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size()); // 共视点对之间的平均距离
            // 一：共视点对之间的平均距离超过规定阈值
            // 二：共视点对中的内点对数 > 12
            // 如果满足上述两个要求，则返回，不需再继续遍历化滑窗中的帧
            // solveRelativeRT()求出的是最新帧到第i帧的相对运动
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;  // ATTENTION：这里所求应该是最新帧到第l帧的相对运动
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief 将待优化向量转换为double数组形式
 * 
 * @return ** void 
 */
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

// 这个函数实际上没有使用，直接返回false
// 如果需要使用，需将return false注释
bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * @brief 
 * 
 * @return ** void 
 */
void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    // Step 1、将所有待优化变量转变为double数组形式，因为ceres不能直接优化向量vector
    vector2double();

    // Step 2、构建ceres优化问题problem，和损失函数loss_function
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); 

    // Step 3、添加待优化变量，即向ceres问题中添加参数块
    // Step 3.1、添加滑窗中位姿、速度和加速度计以及陀螺仪的偏置
    /*
        para_Pose[i] = Ps[i]、Rs[i]：位置和姿态；维度为 SIZE_POSE = 7(姿态用四元数表示的时候是四维)
        para_SpeedBias[i] = Vs[i]、Ba[i]、Bg[i]：速度、加速度计和陀螺仪的偏置；维度 SIZE_SPEEDBIAS = 9
    */
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        // 不用IMU，这样就是六自由度不可观了，所以索性fix第一帧
        problem.SetParameterBlockConstant(para_Pose[0]);

    // Step 3.2、添加外参
    // 添加外参 para_Ex_Pose[i] = RIC[i]、TIC[i]；维度 SIZE_POSE = 7
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        /*
            第一次优化外参时，需要满足下列条件：
            ESTIMATE_EXTRINSIC：是否需要估计外参
            frame_count == WINDOW_SIZE：仅当滑窗内帧数达到最大时，才优化外参
            Vs[0].norm() > 0.2：还是需要一些运动激励

            之后，openExEstimation = 1 是不改变的，所以会一直优化外参
        */ 
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            // 如果不优化外参，就将外参固定
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }

    // Step 3.3、添加时间补偿，认为IMU和camera之间存在一个时间差异，需要进行时间同步td
    problem.AddParameterBlock(para_Td[0], 1);
    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]); // 如果没有时间同步的需求或者运动激励不够，则将时间同步td固定
    
    
    // Step 4、开始添加ceres残差块，即相应约束
    // Step 4.1、边缘化带来的先验约束
    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    // Step 4.2、IMU预积分约束
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    // Step 4.3、视觉约束
    int f_m_cnt = 0;
    int feature_index = -1;
    // 遍历每个特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4) // 仅添加性能好的特征点带来的视觉约束
            continue;

        // TODO：这个是不是应该放在continue之前？？？
        // 放在continue之后，就有问题：如果某个特征点被跳过了，但是没有计数，那么索引就会出错
        ++feature_index; // 参加视觉约束的特征点索引

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历观测到该特征点的所有帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++; // 如果第一次进入，imu_j++ : imu_j = imu_i，对应start_frame，跳过
            if (imu_i != imu_j)
            {
                // 这个和vins mono一致
                Vector3d pts_j = it_per_frame.point; // pts_j是归一化相机坐标
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }
            // 如果双目并且都能看到这个特征点
            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    // 这个时候就是用另一个相机去约束这两帧位姿，相比之下多一个相机的外参成为优化变量
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    // 如果i等于j，既然是同一帧，那就无法对位姿形成约束，但是可以对外参和特征点深度形成约束
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    // Step 5、ceres求解
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    // Step 6、将优化后的变量从double数组转换为vector类型，后续继续使用
    double2vector();
    //printf("frame_count: %d \n", frame_count);

    // Step 7、进行滑窗的边缘化。构建先验项，包括先验项的雅克比矩阵以及误差块
    if(frame_count < WINDOW_SIZE) // 如果滑窗内帧数没有达到最大值，就直接返回，无需边缘化
        return;
    
    TicToc t_whole_marginalization;
    // Step 7.1、边缘化最旧帧
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        // A、把上一次先验项中的残差项传递到当前先验项中，并从中去除需要丢弃的状态量
        // last_marginalization_info ：上一次先验项中的优化量，即上一次边缘化后留下的优化量
        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set; // 需要从上一从先验项中marg的优化变量ID
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                // 因为此时是对最旧帧做边缘化处理，所以如果上一次先验项中包含滑窗中第一帧的状态量，需要marg
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            // 构建新的边缘化因子
            /*
                ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
                : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}
            */
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            // 为marginalization_info中添加新边缘化因子的残差块信息（优化变量、待marg的变量）
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        // B、添加第0帧和第1帧之间的IMU预积分值以及第0帧和第1帧和IMU相关优化变量
        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 添加第0帧和第1帧之间的IMU预积分值 = pre_integrations[1]
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); // {0, 1} 需要marg掉para_Pose[0], para_SpeedBias[0]
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        // C、挑选出第一次观测帧为第0帧的特征点，将该特征点的所有视觉观测值加入到marginalization_info中
        {
            int feature_index = -1;
            // 遍历所有特征点
            for (auto &it_per_id : f_manager.feature)
            {
                // 筛除掉性能不好的特征点
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index; // TODO：依旧存疑！！！

                // 寻找第一次观测帧是第0帧的特征点
                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point; // 在第0帧中的归一化相机位置

                // 遍历该特征点的所有观测值
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3}); // 待marg量是para_Pose[imu_i](第一帧位姿)和para_Feature[feature_index](该特征点的深度)
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    // 如果双目并且都能看到这个特征点
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        // 这个时候就是用另一个相机去约束这两帧位姿，相比之下多一个相机的外参成为优化变量
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        // 如果i等于j，既然是同一帧，那就无法对位姿形成约束，但是可以对外参和特征点深度形成约束
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        /*
            D、得到每次IMU和视觉观测(cost_function)对应的参数块(parameter_blocks)，雅克比矩阵(Jacobins)，残差值(residuals)
               并为parameter_block_data赋值
        */
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;

        // E、开启多线程构建用于边缘化的H和b ，同时从H,b中恢复出线性化雅克比和残差
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        // F、滑窗预移动
        // 这里仅仅将指针进行了一次移动，指针对应的数据还是旧数据，调用的 slideWindow() 才能实现真正的滑窗移动
        std::unordered_map<long, double *> addr_shift;
        // 从1开始，因为第一帧的状态不要了
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 这一步的操作指的是第i的位置存放的的是i-1的内容，这就意味着窗口向前移动了一格
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        // 根据地址来得到保留的参数块
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    // Step 7.2、边缘化次新帧
    else
    {   
        // last_marginalization_info：上一次边缘化的先验
        /*
            int count(Iterator first, Iterator last, T &val)：
            std::count()返回给定范围内元素的出现次数。返回[first，last)范围内等于val的元素数。
            所以首先检测上一次先验中是否包含次新帧的pose
        */ 
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
            // A、为边缘化添加先验约束
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set; // 待marg量的索引
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    // 查错
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    // 将次新帧的pose丢弃
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            // B、同上
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            // C、同上
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            // D、滑窗预处理
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                // i 为 次新帧，跳过
                if (i == WINDOW_SIZE - 1)
                    continue;
                // i 为最新帧
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                // i 其它帧
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

/**
 * @brief 滑窗更新
 * 
 * @return ** void 
 */
void Estimator::slideWindow()
{
    TicToc t_margin;
    // Step 1、移除最旧帧
    if (marginalization_flag == MARGIN_OLD)
    {
        // 滑窗中第一帧的时间戳以及R、P
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        // 只有当滑窗中满帧才会更新滑窗
        if (frame_count == WINDOW_SIZE)
        {
            // A、状态量以及预积分的更新
            // 将所有帧的变量都前移一个
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            // 把滑窗末尾(10帧)信息给最新一帧(11帧),WINDOW_SIZE = 10.
            // 已经实现了所有信息的前移，此时，最新一帧已经成为了滑窗中的第10帧，这里只是把原先的最新一帧的信息作为下一次最新一帧的初始值
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
                
                // 清空，等待下一帧，即滑窗中第11帧的到来
                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            // B、删除最旧帧中的所有信息，包括预积分和特征点
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            // C、对当前feature的维护，包括点在滑窗中的第一次观测帧以及点的深度
            slideWindowOld();
        }
    }
    // Step 2、移除次新帧
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            // A、次新帧的时间戳、Ps、Rs直接用最新帧代替
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            // B、如果使用IMU，需要完成次次新帧和最新帧之间预积分的连接
            // 因为删除次新帧，那么最新帧和之前的帧之间的IMU积分就断掉，需要人为重新连接
            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    // 用最新帧的IMU积分覆盖次新帧的IMU积分
                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            // C、对当前feature的维护
            slideWindowNew();
        }
    }
}

// TODO:当边缘化次新帧时，如果某特征点的第一次观测帧为最新帧，更新start_frame；
// 如果start_frame不是次新帧，但是观测帧包含次新帧，则删除对应的观测帧
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

// 对当前feature的维护，如果是非初始胡阶段包括点在滑窗中的第一次观测帧以及点的深度:主要函数——f_manager.removeBackShiftDepth(R0, P0, R1, P1)
void Estimator::slideWindowOld()
{
    sum_of_back++;

    // 需要完成初始化
    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    // 如果初始化成功，现在是后端跟踪阶段
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0]; // 待删除帧从相机系到世界系的旋转
        R1 = Rs[0] * ric[0]; // 当前滑窗中第一帧从相机系到世界系的旋转
        P0 = back_P0 + back_R0 * tic[0]; // 待删除帧从相机系到世界系的平移
        P1 = Ps[0] + Rs[0] * tic[0]; // 当前滑窗中第一帧从相机系到世界系的平移
        f_manager.removeBackShiftDepth(R0, P0, R1, P1); // 额外恢复了当前特征点的深度
    }
    // 初始化阶段：仅仅对点的start_frame更新以及点是否保留
    // 因为尚在初始化阶段，没有对点的深度进行更新
    else
        f_manager.removeBack();
}

// 获得当前帧在世界坐标系下的R P
void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

// 获得指定帧index在世界坐标系下的R P
void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

// 通过匀速模型预测下一帧能够跟踪到的特征点的像素坐标
void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    // 分别获得当前帧和上一帧的位姿
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    // 使用一个简单的匀速模型去预测下一帧位姿
    nextT = curT * (prevT.inverse() * curT);
    // 特征点id->预测帧相机坐标系坐标
    map<int, Eigen::Vector3d> predictPts;
    // 遍历所有的特征点
    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            // 计算观测到该点的首帧和末帧的索引
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            // 如果他的末帧等于当前滑窗的最后一帧，说明没有跟丢，才有预测下一帧位置的可能
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                // 特征点位置转到imu坐标系
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                // 转到世界坐标系
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                // 转到预测的下一帧的imu坐标系下去
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                // 转到下一帧的相机坐标系
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    // 设置下一帧预测的像素坐标
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

/**
 * @brief 计算一个点的重投影误差
 * 
 * @param Ri 第i帧的旋转：b ——> w
 * @param Pi 第i帧的平移：b ——> w
 * @param rici 外参旋转：c ——> b
 * @param tici 外参平移：c ——> b
 * @param Rj 第j帧的旋转：b ——> w
 * @param Pj 第j帧的平移：b ——> w
 * @param ricj 和rici一致
 * @param ticj 和tici一致
 * @param depth 
 * @param uvi i帧上的点在归一化相机系下的坐标
 * @param uvj j帧上的点在归一化相机系下的坐标
 * @return ** double 
 */
double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi; // i帧下的点转到世界坐标系
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);  // 再转到j帧的相机坐标系
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();    // 计算了一个归一化相机平面的重投影误差
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

/**
 * @brief 将重投影误差过大的点视为外点
 * 
 * @param[in&out] removeIndex 外点索引
 * @return ** void 
 */
void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    // 遍历每一个特征点
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        // 跳过观测帧数小于4的特征点
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 观测帧数小于4就先不看
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        // 遍历该特征点在每一帧中的观测
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            // 如果是双目的话，同时这一帧也被另一个相机看到
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                // 这两个if else看着一模一样？
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        double ave_err = err / errCnt;
        // 归一化相机 * 焦距 = 像素
        if(ave_err * FOCAL_LENGTH > 3)  // 转换到像素上就是3pixel
            removeIndex.insert(it_per_id.feature_id);   // 就认为是一个outlier

    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
