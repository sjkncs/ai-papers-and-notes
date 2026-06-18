"""
一键部署脚本: 将项目复制到 E 盘并初始化 Git 仓库
运行: python setup_e_drive.py
"""

import os
import shutil
import subprocess
import sys

PROJECT_NAME = "ai-papers-and-notes"
TARGET_DIR = r"E:\ai-papers-and-notes"
GITHUB_REPO = "https://github.com/sjkncs/ai-papers-and-notes.git"


def main():
    print("=" * 50)
    print(f"部署 {PROJECT_NAME} 到 E 盘")
    print("=" * 50)
    
    # 获取当前脚本所在目录的父目录
    source_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n[1/4] 源目录: {source_dir}")
    
    # 检查 E 盘
    if not os.path.exists("E:\\"):
        print("[ERROR] E 盘不存在或未挂载!")
        sys.exit(1)
    print("[2/4] E 盘已就绪")
    
    # 复制项目
    if os.path.exists(TARGET_DIR):
        print(f"[WARN] {TARGET_DIR} 已存在, 将合并更新...")
        # 合并复制
        for item in os.listdir(source_dir):
            src = os.path.join(source_dir, item)
            dst = os.path.join(TARGET_DIR, item)
            if item in ["__pycache__", ".git", "setup_e_drive.py"]:
                continue
            if os.path.isdir(src):
                if os.path.exists(dst):
                    # 递归合并
                    for root, dirs, files in os.walk(src):
                        rel = os.path.relpath(root, src)
                        dst_root = os.path.join(dst, rel)
                        os.makedirs(dst_root, exist_ok=True)
                        for f in files:
                            shutil.copy2(
                                os.path.join(root, f),
                                os.path.join(dst_root, f)
                            )
                else:
                    shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"[3/4] 文件已更新到 {TARGET_DIR}")
    else:
        shutil.copytree(
            source_dir, TARGET_DIR,
            ignore=shutil.ignore_patterns("__pycache__", ".git")
        )
        print(f"[3/4] 项目已复制到 {TARGET_DIR}")
    
    # Git 初始化
    git_dir = os.path.join(TARGET_DIR, ".git")
    if not os.path.exists(git_dir):
        print("[4/4] 初始化 Git 仓库...")
        subprocess.run(["git", "init"], cwd=TARGET_DIR)
        subprocess.run(
            ["git", "remote", "add", "origin", GITHUB_REPO],
            cwd=TARGET_DIR
        )
        print(f"  远程仓库: {GITHUB_REPO}")
    else:
        print("[4/4] Git 仓库已存在")
    
    print("\n" + "=" * 50)
    print("部署完成!")
    print(f"项目路径: {TARGET_DIR}")
    print("\n后续操作:")
    print(f"  cd {TARGET_DIR}")
    print("  git add .")
    print('  git commit -m "Update: 2026-01-15~21 papers + quant code"')
    print("  git push origin main")
    print("=" * 50)


if __name__ == "__main__":
    main()
