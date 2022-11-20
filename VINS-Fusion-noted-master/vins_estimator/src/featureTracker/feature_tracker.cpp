/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker.h"

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

double distance(cv::Point2f pt1, cv::Point2f pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
    stereo_cam = 0;
    n_id = 0;
    hasPrediction = false;
}

/**
 * @brief 对跟踪到的特征点，按照被追踪到的次数排序并依次选点，使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀。
 * 
 * @return ** void 
 */
void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    // 目的是使特征点能够被更长时间的跟踪

    /*
    int：该特征点被跟踪次数 = track_cnt[i]
    pair<cv::Point2f, int>：特征点的2D观测位置，以及该特征点的ID号
    */ 
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    // 将特征点按照track_cnt的次数，从大到小排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        // 这个特征点对应的mask值为255，表明点是黑的，还没占有
        // 因为可能在cv::circle(mask, it.second.first, MIN_DIST, 0, -1)的时候，某些特征点被设置为0
        // 这样避免了特征点扎堆现象
        if (mask.at<uchar>(it.second.first) == 255)
        {   
            // 将该点重新存入
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // 并在该点周围MIN_DIST的范围内将像素点设为0，避免在该点附近又进行了像素点的提取，从而避免扎堆现象
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 光流追踪，支持单目和双目模式
/**
 * @brief 对输入图像进行特征点提取以及光流跟踪
 * @param[in] _cur_time 当前帧的时间戳
 * @param[in] _img 输入左目帧
 * @param[in] _img1 输入右目帧
 * @param[out] map 帧中成功跟踪的像素点以及当前帧中新提取的像素点
 **/
map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> FeatureTracker::trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1)
{
    TicToc t_r;
    cur_time = _cur_time; // 当前时间戳
    cur_img = _img; // 左目帧
    row = cur_img.rows;
    col = cur_img.cols;
    cv::Mat rightImg = _img1;   // 右目帧
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    /*
    Step 1、跟踪上一帧
              A、如果是第一帧，跳过Step1，仅做角点提取相关工作
              B、如果是第2、3...帧，需要跟踪上一帧
    */ 
    // 没有额外加right说明的，都是右目帧相关的变量
    cur_pts.clear();
    // Step 1.1、正向跟踪
    // prev_pts非空，说明该帧非第一帧，当前帧需要和上一帧做光流追踪
    // 否则的话，直接跳过这部分
    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;   // 跟踪状态：1 跟踪成功；0 跟踪失败
        vector<float> err;
        // 是否有一些预测的先验值，因为光流追踪本质上是一个高度非线性的优化问题，如果有更好的初值，就可以使用更少的金字塔层数来实现高精度追踪结果
        if(hasPrediction)
        {
            cur_pts = predict_pts;
            // 这里就是使用了两层金字塔（1：0 1）
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            
            int succ_num = 0;
            // 统计跟踪成功的点数
            for (size_t i = 0; i < status.size(); i++)
            {
                if (status[i])
                    succ_num++;
            }
            // 如果追上的结果有点少，那就使用四层金字塔重新追踪一次（3：0 1 2 3）
            if (succ_num < 10)
               cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        }
        else
            // 当然要是没有好的先验，就和之前一样，使用四层金字塔追踪
            cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 3);
        // Step 1.2、反向跟踪
        // reverse check
        // 双向光流检查，通常我们是用前一帧的结果来追踪后一帧，也可以用后一帧结果来追踪前一帧，做一个double check
        if(FLOW_BACK)
        {
            vector<uchar> reverse_status;   // 记录反向跟踪状态
            vector<cv::Point2f> reverse_pts = prev_pts;
            // 由于做过一遍了，置信度也比较高了，所以这里就用两层金字塔了
            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 1, 
            cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), cv::OPTFLOW_USE_INITIAL_FLOW);
            //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3); 
            for(size_t i = 0; i < status.size(); i++)
            {
                // 两遍状态都得好并且反向追踪和原始角点像素差距小于0.5像素，认为是一个合格的追踪结果
                if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 0.5)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }
        // 下面的操作跟之前一直，就是保留被成功追踪的特征点
        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        // 关于ids、track_cnt的说明
        /*
           ids：当有新的特征点进入时，特征点ID号+1，并push_back进入ids里面
           track_cnt：记录特征点成功跟踪的次数
           ATTENTION：ids和track_cnt都只是记录当前帧成功跟踪到的特征点！！！
           如果没有被当前帧成功看到，那么这个点就会从ids和track_cnt中被剔除
           这应该和VINS的某些策略息息相关：因为VINS的关键帧选取就是——最新帧和次新帧的视差对比，从而判断是丢弃最旧帧或是丢弃次新帧
           这样永远都可以将新帧进行保留，而且不需要过多历史帧信息
        */
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //printf("track cnt %d\n", (int)ids.size());
    }

    /* Step 1.3、记录特征点被观测到的次数
        如果是第一帧，会直接跳到这，进行特征点提取和记录
    */
    // 该操作针对第2、3...帧之后的帧
    // 已经将跟踪失败的特征点进行剔除，剩下的特征点说明又被成功跟踪，那么跟踪次数依次+1
    for (auto &n : track_cnt)
        n++;
    
    // Step 2、提取当前帧中新的特征点信息
    // 提取新的特征点，这部分也和之前一致
    if (1)
    {
        // Step 2.1、通过F剔除外点
        //rejectWithF();
        // Step 2.2、设置mask。不仅提出密集的特征点，还设置感兴趣区域防止新特征点密集
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        // Step 2.3、提取新的特征点
        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        /*
        MAX_CNT：规定的每帧中能够提取最大数量的特征点
        static_cast<int>(cur_pts.size())：现已有的特征点数量
        n_max_cnt：还能提取的特征点数量
        */ 
        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            /*void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值
             *)   
             */
            cv::goodFeaturesToTrack(cur_img, n_pts, MAX_CNT - cur_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        // 若提取到新的特征点，就将新特征点ids、track_cnt更新
        for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        //printf("feature cnt after add %d\n", (int)ids.size());
    }

    // Step 3、去畸变并归一化相机平面
    // cur_un_pts记录得是去畸变后的在归一化相机坐标系下的x,y
    // TODO：为什么不对2D像素坐标去畸变？？？目前为止cur_pts应该都没有进行去畸变？？？然后直接使用？？？
    //       或者哪儿我没有注意到？？？
    //       ！！！还是说为了能够和下一帧图像的坐标进行原滋原味的跟踪，就不去畸变？？？
    //       需要和ORB中的相关处理进行比较——光流跟踪和描述子匹配的差异之一
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);

    // Step 4、计算当前特征点速度
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    // Step 5、若是双目，需要额外进行一些处理
    // 针对双目进行处理
    if(!_img1.empty() && stereo_cam)
    {
        // 一些状态位的清零工作
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        // 如果左目帧上的特征点非空
        if(!cur_pts.empty())
        {   
            // 左右目的特征点跟踪
            //printf("stereo image; track feature on right image\n");
            vector<cv::Point2f> reverseLeftPts;
            vector<uchar> status, statusRightLeft;
            vector<float> err;
            // cur left ---- cur right
            // 左目相机去和右目相机做光流追踪
            cv::calcOpticalFlowPyrLK(cur_img, rightImg, cur_pts, cur_right_pts, status, err, cv::Size(21, 21), 3);
            // reverse check cur right ---- cur left
            // 同样可以做双向光流的double check
            if(FLOW_BACK)
            {
                cv::calcOpticalFlowPyrLK(rightImg, cur_img, cur_right_pts, reverseLeftPts, statusRightLeft, err, cv::Size(21, 21), 3);
                for(size_t i = 0; i < status.size(); i++)
                {
                    if(status[i] && statusRightLeft[i] && inBorder(cur_right_pts[i]) && distance(cur_pts[i], reverseLeftPts[i]) <= 0.5)
                        status[i] = 1;
                    else
                        status[i] = 0;
                }
            }

            ids_right = ids;
            // 一些瘦身操作
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // 如果左目的特征点右目没有，也没关系，不会影响左目已经提出来的特征点
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            // 右目去畸变以及计算速度
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    if(SHOW_TRACK)
        drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    // Step 6、将当前帧的状态进行转移，并保存
    // 当前的状态量转换成上一帧的状态量
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;
    hasPrediction = false;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];
    // 对结果进行总结
    // idx -> （cam id - 性质）
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    // 左目帧信息保存
    for (size_t i = 0; i < ids.size(); i++)
    {
        int feature_id = ids[i];
        // 去畸变的归一化相机坐标
        double x, y ,z;
        x = cur_un_pts[i].x;
        y = cur_un_pts[i].y;
        z = 1;
        // TODO : 没有去畸变的像素坐标？？？看后续在后端优化的时候会不会进行额外处理
        double p_u, p_v;
        p_u = cur_pts[i].x;
        p_v = cur_pts[i].y;
        // 左目的camera id = 0
        int camera_id = 0;
        double velocity_x, velocity_y;
        velocity_x = pts_velocity[i].x;
        velocity_y = pts_velocity[i].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    // 若是双目，还需进行右目帧的信息保存
    if (!_img1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x, y ,z;
            x = cur_un_right_pts[i].x;
            y = cur_un_right_pts[i].y;
            z = 1;
            double p_u, p_v;
            p_u = cur_right_pts[i].x;
            p_v = cur_right_pts[i].y;
            // 右目的camera id = 1
            int camera_id = 1;
            double velocity_x, velocity_y;
            velocity_x = right_pts_velocity[i].x;
            velocity_y = right_pts_velocity[i].y;

            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            // 只有右目也有对应特征点的才会emplace back
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }

    //printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}

/**
 * @brief 通过计算基础矩阵F剔除外点
 * 
 * @return ** void 
 */
void FeatureTracker::rejectWithF()
{
    // 至少有8对点，才能进行F的求解
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        // Step 1、得到去畸变后的归一化像素坐标，用于F矩阵的计算
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // liftProjective()根据规定的相机模型将2D像素坐标去畸变，转换为3D相机坐标
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 3D相机坐标/Z，转换到归一化相机坐标；再通过设定的参数转换到归一化像素坐标（此时的像素坐标已经去畸变）
            // 在这里认为fx = fy = FOCAL_LENGTH；cx = col/2.0；cy = row/2.0 
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        // Step 2、根据得到的归一化像素坐标进行F矩阵的求解，再通过status进行外点剔除
        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (size_t i = 0; i < calib_file.size(); i++)
    {
        ROS_INFO("reading paramerter of camera %s", calib_file[i].c_str());
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file[i]);
        m_camera.push_back(camera);
    }
    if (calib_file.size() == 2)
        stereo_cam = 1;
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++)
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}

/**
 * @brief 去畸变，并变换到归一化坐标系下
 * 
 * @param[in] pts 需要处理的2D特征点
 * @param[in] cam 相机模型
 * @return ** vector<cv::Point2f> 返回处理后的无畸变的归一化相机坐标
 */
vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

/**
 * @brief 计算两帧之间的点在归一化相机平面上的速度
 * 
 * @param[in] ids 当前帧上特征点的ID号
 * @param[in] pts 当前帧上特征点2D像素位置
 * @param cur_id_pts 当前帧上的特征点在归一化平面的3D坐标（仅包含x y）
 * @param prev_id_pts 上一帧上的特征点在归一化平面的3D坐标（仅包含x y）
 * @return ** vector<cv::Point2f> 两帧之间的归一化相机3D点在x y轴上的运动速度
 */
vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    // Step 1、通过当前归一化相机坐标位置（仅x, y）和角点ID重新赋值cur_id_pts
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    // Step 2、计算上一帧和当前帧之间点的运动速度（或称为光流速度）
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time; // 两帧之间的时间间隔
        
        // 遍历当前帧上的所有点
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]); // 在上一帧中找到ID相同的点，计算速度
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}

void FeatureTracker::drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && stereo_cam)
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    if (!imRight.empty() && stereo_cam)
    {
        for (size_t i = 0; i < curRightPts.size(); i++)
        {
            cv::Point2f rightPt = curRightPts[i];
            rightPt.x += cols;
            cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            //cv::Point2f leftPt = curLeftPtsTrackRight[i];
            //cv::line(imTrack, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }
    }
    
    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            cv::arrowedLine(imTrack, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
}

/**
 * @brief 将预测的相机坐标转为为像素坐标
 * 
 * @param predictPts 预测的相机坐标
 * @return ** void 
 */
void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    hasPrediction = true;
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        // 如果这个特征点有对应的预测的位置
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            // 相机坐标系投影到像素坐标系
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
            predict_pts_debug.push_back(cv::Point2f(tmp_uv.x(), tmp_uv.y()));
        }
        // 如果没有，将上一次观测位姿放入到预测位置中
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (size_t i = 0; i < ids.size(); i++)
    {
        itSet = removePtsIds.find(ids[i]);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}