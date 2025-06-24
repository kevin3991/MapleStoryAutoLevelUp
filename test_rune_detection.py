#!/usr/bin/env python3
"""
測試 is_in_rune_game 函數的執行過程
使用 testing/rune_detected_2025-06-18_21-11-26.png 作為測試圖片
"""

import cv2
import numpy as np
import yaml
import os
import sys

# 添加當前目錄到 Python 路徑
sys.path.append('.')

from logger import logger
from util import find_pattern_sqdiff, get_mask, load_image, load_yaml, is_mac, override_cfg

def test_is_in_rune_game():
    """測試 is_in_rune_game 函數的執行過程"""
    
    # 載入配置
    cfg = load_yaml("config/config_default.yaml")
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    cfg = override_cfg(cfg, load_yaml("config/config_edit_me.yaml"))
    
    # 載入測試圖片
    test_image_path = "testing/rune_detected_2025-06-18_21-11-26.png"
    if not os.path.exists(test_image_path):
        logger.error(f"測試圖片不存在: {test_image_path}")
        return
    
    # 載入測試圖片作為 frame
    frame = cv2.imread(test_image_path)
    if frame is None:
        logger.error(f"無法載入測試圖片: {test_image_path}")
        return
    
    logger.info(f"載入測試圖片: {frame.shape}")
    
    # 載入箭頭模板圖片
    img_arrows = {
        "left": [
            load_image("rune/arrow_left_1.png"),
            load_image("rune/arrow_left_2.png"),
            load_image("rune/arrow_left_3.png"),
        ],
        "right": [
            load_image("rune/arrow_right_1.png"),
            load_image("rune/arrow_right_2.png"),
            load_image("rune/arrow_right_3.png"),
        ],
        "up": [
            load_image("rune/arrow_up_1.png"),
            load_image("rune/arrow_up_2.png"),
            load_image("rune/arrow_up_3.png"),
        ],
        "down": [
            load_image("rune/arrow_down_1.png"),
            load_image("rune/arrow_down_2.png"),
            load_image("rune/arrow_down_3.png"),
        ],
    }
    
    # 模擬 is_in_rune_game 函數的執行過程
    logger.info("開始執行 is_in_rune_game 測試...")
    
    # 1. 調整圖片大小到 1296x759
    img_frame = cv2.resize(frame, (1296, 759), interpolation=cv2.INTER_NEAREST)
    logger.info(f"調整圖片大小到: {img_frame.shape}")
    
    # 2. 截取箭頭檢測區域
    x, y = cfg["rune_solver"]["arrow_box_coord"]
    size = cfg["rune_solver"]["arrow_box_size"]
    img_roi = img_frame[y:y+size, x:x+size]
    logger.info(f"截取 ROI 區域: 位置({x}, {y}), 大小({size}x{size})")
    
    # 儲存 ROI 圖片
    cv2.imwrite("debug_roi.png", img_roi)
    logger.info("已儲存 ROI 圖片: debug_roi.png")
    
    # 3. 檢查箭頭是否存在
    best_score = float('inf')
    best_direction = ""
    best_arrow_idx = -1
    
    logger.info("開始比對箭頭模板...")
    
    for direction, arrow_list in img_arrows.items():
        for idx, img_arrow in enumerate(arrow_list):
            _, score, _ = find_pattern_sqdiff(
                img_roi, img_arrow,
                mask=get_mask(img_arrow, (0, 255, 0))
            )
            logger.info(f"方向: {direction}, 箭頭 {idx+1}, 分數: {score:.4f}")
            
            if score < best_score:
                best_score = score
                best_direction = direction
                best_arrow_idx = idx
    
    logger.info(f"最佳匹配: 方向={best_direction}, 箭頭={best_arrow_idx+1}, 分數={best_score:.4f}")
    
    # 4. 檢查是否超過閾值
    threshold = cfg["rune_solver"]["arrow_box_diff_thres"]
    is_detected = best_score < threshold
    
    logger.info(f"閾值: {threshold}")
    logger.info(f"檢測結果: {'是' if is_detected else '否'} (分數 {best_score:.4f} {'<' if is_detected else '>='} {threshold})")
    
    # 5. 創建調試圖片
    img_frame_debug = img_frame.copy()
    
    # 畫出檢測區域
    cv2.rectangle(
        img_frame_debug, 
        (x, y), 
        (x + size, y + size), 
        (0, 0, 255), 
        2
    )
    
    # 添加文字說明
    text = f"ROI: ({x},{y}) {size}x{size}"
    cv2.putText(img_frame_debug, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    text = f"Best: {best_direction} arrow{best_arrow_idx+1}, score: {best_score:.4f}"
    cv2.putText(img_frame_debug, text, (x, y+size+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    text = f"Threshold: {threshold}, Detected: {'YES' if is_detected else 'NO'}"
    cv2.putText(img_frame_debug, text, (x, y+size+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if is_detected else (0, 0, 255), 2)
    
    # 儲存調試圖片
    cv2.imwrite("debug_rune_detection.png", img_frame_debug)
    logger.info("已儲存調試圖片: debug_rune_detection.png")
    
    # 6. 顯示結果
    print("\n" + "="*50)
    print("測試結果摘要:")
    print("="*50)
    print(f"測試圖片: {test_image_path}")
    print(f"圖片尺寸: {frame.shape} -> {img_frame.shape}")
    print(f"ROI 區域: 位置({x}, {y}), 大小({size}x{size})")
    print(f"最佳匹配: {best_direction} 方向, 箭頭 {best_arrow_idx+1}")
    print(f"匹配分數: {best_score:.4f}")
    print(f"閾值: {threshold}")
    print(f"檢測結果: {'✅ 檢測到符文遊戲' if is_detected else '❌ 未檢測到符文遊戲'}")
    print("="*50)
    
    # 7. 顯示圖片
    cv2.imshow("原始測試圖片", frame)
    cv2.imshow("ROI 區域", img_roi)
    cv2.imshow("調試圖片", img_frame_debug)
    
    print("\n按任意鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_is_in_rune_game() 