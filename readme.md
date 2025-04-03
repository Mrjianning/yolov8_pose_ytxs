
## 引体向上动作分析系统

### 一、项目介绍

本项目是一个基于深度学习的动作分析系统，旨在通过分析人的动作来识别不同的动作类型。该系统使用了深度学习模型来对动作进行分类，并且提供了一个简单易用的用户界面。

### 二、 项目结构

-`data`: 存放数据

-`models`: 存放模型

### 三、 环境要求

- Python 3.8+
- PyTorch 1.7+
- OpenCV

### 四、安装依赖

```

<!-- 创建conda环境 -->

conda create -n pose python=3.8


<!-- 激活环境 -->

conda activate pose


pip install -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple

```



### 五、数据库设计

#### 数据库表设计

##### 1. 用户表 (users)
存储用户基本信息

```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 用户唯一标识符
    name TEXT NOT NULL,                        -- 用户姓名
    registration_date DATETIME DEFAULT CURRENT_TIMESTAMP,  -- 注册日期
    gender TEXT CHECK(gender IN ('男', '女')),  -- 性别（限定为男/女）
    age INTEGER                                -- 年龄
);
```

##### 2. 训练会话表 (training_sessions)
记录每次训练的整体信息

```sql
CREATE TABLE training_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 训练会话ID
    user_id INTEGER NOT NULL,                     -- 关联的用户ID
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP, -- 训练开始时间
    end_time DATETIME,                            -- 训练结束时间
    total_pullups INTEGER DEFAULT 0,              -- 本次训练引体向上总数
    average_left_angle REAL,                      -- 本次训练左手肘平均角度
    average_right_angle REAL,                     -- 本次训练右手肘平均角度
    FOREIGN KEY (user_id) REFERENCES users(user_id)  -- 外键关联用户表
);
```

##### 3. 引体向上细节表 (pullup_details)
记录每个引体向上的详细姿态数据

```sql
CREATE TABLE pullup_details (
    detail_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 细节记录ID
    session_id INTEGER NOT NULL,                 -- 关联的训练会话ID
    pullup_number INTEGER NOT NULL,              -- 当前是第几个引体向上（从1开始计数）
    left_elbow_angle REAL,                       -- 当前动作左手肘角度（单位：度）
    right_elbow_angle REAL,                      -- 当前动作右手肘角度（单位：度）
    chin_over_bar BOOLEAN,                       -- 下巴是否超过横杆（完成标准）
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, -- 动作记录时间戳
    FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)  -- 外键关联训练会话
);
```

##### 4.查询示例

```sql
-- 获取某次训练的角度变化数据（用于折线图）
SELECT 
    pullup_number AS '引体向上序号',
    left_elbow_angle AS '左手角度',
    right_elbow_angle AS '右手角度'
FROM 
    pullup_details
WHERE 
    session_id = 123  -- 替换为具体会话ID
ORDER BY 
    pullup_number;
```

