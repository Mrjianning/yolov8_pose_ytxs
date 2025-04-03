import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class PullUpTrainerDB:
    """引体向上训练数据库操作类"""
    
    def __init__(self, db_path: str = 'pullup_trainer.db'):
        """初始化数据库连接
        
        Args:
            db_path: 数据库文件路径，默认为 pullup_trainer.db
        """
        self.db_path = db_path
        self.conn = None
        
    def __enter__(self):
        """支持 with 语句"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句时自动关闭连接"""
        self.close()
    
    def connect(self):
        """建立数据库连接"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # 使查询返回字典样式的行
        self.conn.execute("PRAGMA foreign_keys = ON")  # 启用外键约束
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def initialize_db(self):
        """初始化数据库表结构"""
        with self.conn:
            # 创建用户表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                registration_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                gender TEXT CHECK(gender IN ('男', '女')),
                age INTEGER
            )""")
            
            # 创建训练会话表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                total_pullups INTEGER DEFAULT 0,
                average_left_angle REAL,
                average_right_angle REAL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )""")
            
            # 创建引体向上细节表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pullup_details (
                detail_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                pullup_number INTEGER NOT NULL,
                left_elbow_angle REAL,
                right_elbow_angle REAL,
                chin_over_bar BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id) ON DELETE CASCADE
            )""")
            
            # 创建索引提高查询性能
            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pullup_details_session 
            ON pullup_details(session_id)
            """)
            
            self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user 
            ON training_sessions(user_id)
            """)

    # ========== 用户管理 ==========
    def add_user(self, name: str, gender: str = None, age: int = None) -> int:
        """添加新用户
        
        Args:
            name: 用户名
            gender: 性别（'男'或'女'）
            age: 年龄
            
        Returns:
            新用户的ID
        """
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO users(name, gender, age) VALUES(?,?,?)",
                (name, gender, age)
            )
            return cursor.lastrowid
    
    def get_user(self, user_id: int) -> Optional[Dict]:
        """获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            包含用户信息的字典，如果用户不存在返回None
        """
        cursor = self.conn.execute(
            "SELECT * FROM users WHERE user_id = ?", 
            (user_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def update_user(self, user_id: int, name: str = None, gender: str = None, age: int = None):
        """更新用户信息
        
        Args:
            user_id: 要更新的用户ID
            name: 新用户名（可选）
            gender: 新性别（可选）
            age: 新年龄（可选）
        """
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"用户ID {user_id} 不存在")
            
        name = name if name is not None else user['name']
        gender = gender if gender is not None else user['gender']
        age = age if age is not None else user['age']
        
        with self.conn:
            self.conn.execute(
                "UPDATE users SET name=?, gender=?, age=? WHERE user_id=?",
                (name, gender, age, user_id)
            )
    
    def delete_user(self, user_id: int):
        """删除用户及其所有训练数据（级联删除）"""
        with self.conn:
            self.conn.execute(
                "DELETE FROM users WHERE user_id = ?",
                (user_id,)
            )

    # ========== 训练会话管理 ==========
    def start_session(self, user_id: int) -> int:
        """开始新的训练会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            新会话的ID
        """
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO training_sessions(user_id) VALUES(?)",
                (user_id,)
            )
            return cursor.lastrowid
    
    def end_session(self, session_id: int):
        """结束训练会话并计算统计数据"""
        # 计算平均角度和总数
        cursor = self.conn.execute(
            """SELECT AVG(left_elbow_angle), AVG(right_elbow_angle), COUNT(*)
               FROM pullup_details WHERE session_id = ?""",
            (session_id,)
        )
        left_avg, right_avg, total = cursor.fetchone()
        
        with self.conn:
            # 更新会话记录
            self.conn.execute(
                """UPDATE training_sessions
                   SET end_time = datetime('now'),
                       total_pullups = ?,
                       average_left_angle = ?,
                       average_right_angle = ?
                   WHERE session_id = ?""",
                (total, left_avg or 0, right_avg or 0, session_id)
            )
    
    def get_session(self, session_id: int) -> Optional[Dict]:
        """获取训练会话信息"""
        cursor = self.conn.execute(
            "SELECT * FROM training_sessions WHERE session_id = ?", 
            (session_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_user_sessions(self, user_id: int, limit: int = 10) -> List[Dict]:
        """获取用户最近的训练会话
        
        Args:
            user_id: 用户ID
            limit: 返回的会话数量限制
            
        Returns:
            按开始时间倒序排列的训练会话列表
        """
        cursor = self.conn.execute(
            """SELECT * FROM training_sessions
               WHERE user_id = ?
               ORDER BY start_time DESC
               LIMIT ?""",
            (user_id, limit)
        )
        return [dict(row) for row in cursor]

    # ========== 引体向上细节管理 ==========
    def add_pullup_detail(
        self,
        session_id: int,
        pullup_number: int,
        left_elbow_angle: float,
        right_elbow_angle: float,
        chin_over_bar: bool = True
    ) -> int:
        """记录单个引体向上细节
        
        Args:
            session_id: 会话ID
            pullup_number: 第几个引体向上
            left_elbow_angle: 左手肘角度
            right_elbow_angle: 右手肘角度
            chin_over_bar: 下巴是否过杆
            
        Returns:
            新记录的ID
        """
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO pullup_details(
                       session_id, pullup_number, left_elbow_angle, 
                       right_elbow_angle, chin_over_bar)
                   VALUES(?,?,?,?,?)""",
                (session_id, pullup_number, left_elbow_angle, 
                 right_elbow_angle, chin_over_bar)
            )
            return cursor.lastrowid
    
    def get_pullup_details(self, session_id: int) -> List[Dict]:
        """获取某次训练的所有引体向上细节
        
        Args:
            session_id: 会话ID
            
        Returns:
            包含所有细节记录的列表，按pullup_number排序
        """
        cursor = self.conn.execute(
            """SELECT * FROM pullup_details 
               WHERE session_id = ? 
               ORDER BY pullup_number""",
            (session_id,)
        )
        return [dict(row) for row in cursor]
    
    def get_angles_for_plotting(self, session_id: int) -> Tuple[List[int], List[float], List[float]]:
        """获取绘制折线图所需的数据
        
        Args:
            session_id: 会话ID
            
        Returns:
            元组：(pullup_numbers, left_angles, right_angles)
        """
        details = self.get_pullup_details(session_id)
        numbers = [d['pullup_number'] for d in details]
        left_angles = [d['left_elbow_angle'] for d in details]
        right_angles = [d['right_elbow_angle'] for d in details]
        return numbers, left_angles, right_angles
    
    def get_session_stats(self, session_id: int) -> Dict:
        """获取训练会话的统计信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            包含统计信息的字典，包括：
            - total_pullups: 总个数
            - completion_rate: 完成率（下巴过杆的比例）
            - left_angle_stats: 左手角度统计（min, max, avg）
            - right_angle_stats: 右手角度统计（min, max, avg）
        """
        # 获取基本会话信息
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"会话ID {session_id} 不存在")
        
        # 计算完成率
        cursor = self.conn.execute(
            """SELECT 
                   COUNT(*) as total,
                   SUM(CASE WHEN chin_over_bar THEN 1 ELSE 0 END) as completed
               FROM pullup_details
               WHERE session_id = ?""",
            (session_id,)
        )
        result = cursor.fetchone()
        completion_rate = result['completed'] / result['total'] if result['total'] > 0 else 0
        
        # 计算角度统计
        cursor = self.conn.execute(
            """SELECT 
                   MIN(left_elbow_angle) as left_min,
                   MAX(left_elbow_angle) as left_max,
                   AVG(left_elbow_angle) as left_avg,
                   MIN(right_elbow_angle) as right_min,
                   MAX(right_elbow_angle) as right_max,
                   AVG(right_elbow_angle) as right_avg
               FROM pullup_details
               WHERE session_id = ?""",
            (session_id,)
        )
        angle_stats = cursor.fetchone()
        
        return {
            'total_pullups': session['total_pullups'],
            'completion_rate': completion_rate,
            'left_angle_stats': {
                'min': angle_stats['left_min'],
                'max': angle_stats['left_max'],
                'avg': angle_stats['left_avg']
            },
            'right_angle_stats': {
                'min': angle_stats['right_min'],
                'max': angle_stats['right_max'],
                'avg': angle_stats['right_avg']
            }
        }

# 测试代码
if __name__ == "__main__":

    # 初始化数据库
    with PullUpTrainerDB() as db:
        db.initialize_db()
        
        # 添加用户
        user_id = db.add_user("李四", "男", 28)
        print(f"添加用户ID: {user_id}")
        
        # 开始训练会话
        session_id = db.start_session(user_id)
        print(f"开始训练会话ID: {session_id}")
        
        # 模拟记录5个引体向上数据
        for i in range(1, 6):
            left_angle = 160 - i * 15  # 左手角度从160°逐渐减小
            right_angle = 155 - i * 14  # 右手角度从155°逐渐减小
            completed = i != 3  # 第三个引体向上故意设为未完成
            db.add_pullup_detail(session_id, i, left_angle, right_angle, completed)
            print(f"记录第{i}个引体向上: 左{left_angle}°, 右{right_angle}°, 完成: {completed}")
        
        # 结束会话
        db.end_session(session_id)
        print("训练会话已结束")
        
        # 获取会话统计信息
        stats = db.get_session_stats(session_id)
        print("\n训练统计:")
        print(f"总个数: {stats['total_pullups']}")
        print(f"完成率: {stats['completion_rate']:.0%}")
        print(f"左手角度 - 最小: {stats['left_angle_stats']['min']:.1f}°, 最大: {stats['left_angle_stats']['max']:.1f}°, 平均: {stats['left_angle_stats']['avg']:.1f}°")
        print(f"右手角度 - 最小: {stats['right_angle_stats']['min']:.1f}°, 最大: {stats['right_angle_stats']['max']:.1f}°, 平均: {stats['right_angle_stats']['avg']:.1f}°")
        
        # 获取绘图数据
        numbers, left_angles, right_angles = db.get_angles_for_plotting(session_id)
        print("\n折线图数据:")
        print(f"序号: {numbers}")
        print(f"左手角度: {left_angles}")
        print(f"右手角度: {right_angles}")

        # 绘制折线图
        import matplotlib.pyplot as plt
        plt.plot(numbers, left_angles, label='左手角度')
        plt.plot(numbers, right_angles, label='右手角度')
        plt.xlabel('引体向上次数')
        plt.ylabel('角度（°）')
        plt.title('引体向上训练数据')
        plt.legend()
        plt.show()