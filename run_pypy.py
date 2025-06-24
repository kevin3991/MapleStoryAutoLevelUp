#!/usr/bin/env python3
"""
PyPy 啟動腳本
使用方式: pypy3 run_pypy.py [參數]
"""

import sys
import subprocess
import os

def check_pypy_installed():
    """檢查 PyPy 是否已安裝"""
    try:
        result = subprocess.run(['pypy3', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ PyPy 已安裝: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ PyPy 未安裝")
    print("請先安裝 PyPy:")
    print("  macOS: brew install pypy3")
    print("  Linux: sudo apt-get install pypy3")
    print("  Windows: 下載 https://www.pypy.org/download.html")
    return False

def install_pypy_dependencies():
    """安裝 PyPy 依賴套件"""
    print("📦 安裝 PyPy 依賴套件...")
    
    try:
        # 安裝基本依賴
        subprocess.run(['pypy3', '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True)
        
        # 安裝 requirements.txt 中的套件
        subprocess.run(['pypy3', '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        
        print("✅ 依賴套件安裝完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安裝失敗: {e}")
        return False

def run_maple_story_bot(args):
    """運行 MapleStory 機器人"""
    cmd = ['pypy3', 'mapleStoryAutoLevelUp.py'] + args
    
    print(f"🚀 啟動 PyPy 版本的 MapleStory 機器人...")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 程式執行失敗: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ 程式已停止")
        return True
    
    return True

def main():
    """主函數"""
    print("🎮 MapleStory 自動練功 - PyPy 版本")
    print("=" * 50)
    
    # 檢查 PyPy 是否已安裝
    if not check_pypy_installed():
        return 1
    
    # 檢查是否需要安裝依賴
    if not os.path.exists('venv_pypy'):
        print("📦 首次運行，需要安裝依賴套件...")
        if not install_pypy_dependencies():
            return 1
    
    # 獲取命令行參數
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # 如果沒有參數，提供預設參數
    if not args:
        print("📝 使用預設參數...")
        args = ['--map', 'fire_land_2', '--monsters', 'fire_pig', '--attack', 'aoe_skill']
    
    # 運行機器人
    success = run_maple_story_bot(args)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main()) 