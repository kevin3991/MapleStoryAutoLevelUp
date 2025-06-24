#!/usr/bin/env python3
"""
性能測試腳本
比較 CPython 和 PyPy 的性能差異
"""

import time
import cv2
import numpy as np
import psutil
import os

def test_image_processing():
    """測試圖像處理性能"""
    print("🖼️ 測試圖像處理性能...")
    
    # 創建測試圖片
    img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    template = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    start_time = time.time()
    
    # 執行多次圖像處理操作
    for i in range(100):
        # 模板匹配
        result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 邊緣檢測
        edges = cv2.Canny(blurred, 100, 200)
        
        # 顏色轉換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 形態學操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ 圖像處理完成: {processing_time:.2f} 秒")
    return processing_time

def test_numpy_operations():
    """測試 NumPy 運算性能"""
    print("🧮 測試 NumPy 運算性能...")
    
    # 創建大型矩陣
    size = 1000
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    
    start_time = time.time()
    
    # 執行多次矩陣運算
    for i in range(50):
        # 矩陣乘法
        c = np.dot(a, b)
        
        # 矩陣轉置
        d = a.T
        
        # 統計運算
        mean_val = np.mean(a)
        std_val = np.std(a)
        
        # 邏輯運算
        mask = a > 0.5
        filtered = a[mask]
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ NumPy 運算完成: {processing_time:.2f} 秒")
    return processing_time

def test_memory_usage():
    """測試記憶體使用"""
    print("💾 測試記憶體使用...")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 創建大量數據
    data_structures = []
    for i in range(1000):
        # 創建大型 NumPy 陣列
        arr = np.random.rand(100, 100)
        data_structures.append(arr)
        
        # 創建字典
        d = {f"key_{j}": j for j in range(100)}
        data_structures.append(d)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"✅ 記憶體使用: 初始 {initial_memory:.1f}MB, 最終 {final_memory:.1f}MB, 增加 {memory_increase:.1f}MB")
    
    # 清理記憶體
    del data_structures
    
    return memory_increase

def test_cpu_usage():
    """測試 CPU 使用"""
    print("🖥️ 測試 CPU 使用...")
    
    process = psutil.Process()
    
    # 執行計算密集型任務
    start_time = time.time()
    cpu_samples = []
    
    for i in range(10):  # 監控 10 次
        # 執行一些計算
        result = 0
        for j in range(100000):
            result += j * j
        
        # 記錄 CPU 使用率
        cpu_percent = process.cpu_percent()
        cpu_samples.append(cpu_percent)
        
        time.sleep(0.1)
    
    end_time = time.time()
    avg_cpu = sum(cpu_samples) / len(cpu_samples)
    
    print(f"✅ CPU 測試完成: 平均使用率 {avg_cpu:.1f}%, 時間 {end_time - start_time:.2f} 秒")
    return avg_cpu

def main():
    """主測試函數"""
    print("🚀 MapleStory 自動練功 - 性能測試")
    print("=" * 50)
    
    # 檢測 Python 版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python 版本: {python_version}")
    
    # 檢測是否為 PyPy
    is_pypy = hasattr(sys, 'pypy_version_info')
    if is_pypy:
        print(f"⚡ PyPy 版本: {sys.pypy_version_info}")
    else:
        print("🐌 CPython")
    
    print()
    
    # 執行各種測試
    results = {}
    
    # 圖像處理測試
    results['image_processing'] = test_image_processing()
    
    # NumPy 運算測試
    results['numpy_operations'] = test_numpy_operations()
    
    # 記憶體使用測試
    results['memory_usage'] = test_memory_usage()
    
    # CPU 使用測試
    results['cpu_usage'] = test_cpu_usage()
    
    # 總結
    print("\n" + "=" * 50)
    print("📊 測試結果總結:")
    print("=" * 50)
    print(f"圖像處理時間: {results['image_processing']:.2f} 秒")
    print(f"NumPy 運算時間: {results['numpy_operations']:.2f} 秒")
    print(f"記憶體增加: {results['memory_usage']:.1f} MB")
    print(f"平均 CPU 使用率: {results['cpu_usage']:.1f}%")
    
    # 性能評分
    total_time = results['image_processing'] + results['numpy_operations']
    if total_time < 5:
        performance_grade = "A+ (優秀)"
    elif total_time < 10:
        performance_grade = "A (良好)"
    elif total_time < 15:
        performance_grade = "B (一般)"
    else:
        performance_grade = "C (較慢)"
    
    print(f"性能評分: {performance_grade}")
    
    # 建議
    print("\n💡 建議:")
    if is_pypy:
        print("- 你正在使用 PyPy，性能應該比 CPython 更好")
        print("- 如果性能不理想，可以嘗試調整 FPS 設定")
    else:
        print("- 考慮使用 PyPy 來提升性能")
        print("- 執行: make setup-pypy && make run-pypy")
    
    print("\n✅ 測試完成!")

if __name__ == "__main__":
    import sys
    main() 