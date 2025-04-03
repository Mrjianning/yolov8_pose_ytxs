from database.db_controller import PullUpTrainerDB

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