import os
import time
import numpy as np
from datetime import datetime
import configparser

class Utils:
    """一个工具类，提供常用的静态方法"""

    @staticmethod
    def create_directory(path):
        """创建目录，如果目录不存在"""
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return False

    @staticmethod
    def get_timestamp_with_microseconds():
        """获取带微秒的时间戳字符串"""
        return datetime.now().strftime("%Y%m%d%H%M%S%f")

    @staticmethod
    def get_file_extension(filename):
        """获取文件扩展名"""
        return os.path.splitext(filename)[1]

    @classmethod
    def format_time(cls, timestamp):
        """格式化时间戳为可读字符串"""
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def load_ini_config(config_path):

        """加载ini配置文件"""
        try:
            config = configparser.ConfigParser()

            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件 {config_path} 不存在")
            
            # 显式指定编码为 utf-8
            with open(config_path, 'r', encoding='utf-8') as f:
                config.read_file(f)

        except Exception as e:
            raise ValueError(f"读取配置文件失败: {e}")
        
        return config
    

# 示例用法（测试代码）
if __name__ == "__main__":
    # 创建目录
    Utils.create_directory("test_folder")
    print("Directory created")

    # 获取微秒时间戳
    timestamp = Utils.get_timestamp_with_microseconds()
    print(f"Timestamp: {timestamp}")

    # 获取文件扩展名
    ext = Utils.get_file_extension("example.txt")
    print(f"File extension: {ext}")

    # 格式化时间
    current_time = time.time()
    formatted = Utils.format_time(current_time)
    print(f"Formatted time: {formatted}")