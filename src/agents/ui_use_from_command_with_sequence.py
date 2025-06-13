#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shlex
import pyautogui
import time
import os
import subprocess
import re
from datetime import datetime
from PIL import Image
import json

# 取消 PyAutoGUI 默认的 pause（默认每个操作后暂停 0.1 秒）
pyautogui.PAUSE = 0

def ratio_to_pixel(ratio_x: float, ratio_y: float):
    """
    将 0～1 之间的比例坐标转换为主屏幕像素坐标 (x, y)。
    """
    screen_width, screen_height = pyautogui.size()
    x = int(ratio_x * screen_width)
    y = int(ratio_y * screen_height)
    return x, y

def parse_and_execute(line: str):
    """
    解析一行指令（shlex 处理引号），并根据 action 调用相应的 pyautogui 接口。
    """
    try:
        print(f"接收到的命令: {line}", file=sys.stderr)
        
        # 特殊处理 sequence 命令
        if line.lower().startswith("sequence ") or line.lower().startswith("seq "):
            action = line.split()[0].lower()
            # 提取JSON部分 (第一个空格之后的所有内容)
            json_part = line[len(action)+1:].strip()
            parts = [action, json_part]
            print(f"特殊处理sequence命令，parts: {parts}", file=sys.stderr)
        else:
            parts = shlex.split(line)
    except ValueError as e:
        print(f"解析错误：{e}", file=sys.stderr)
        return

    if not parts:
        return

    action = parts[0].lower()

    # 处理连续动作命令
    if action == "sequence" or action == "seq":
        try:
            # 解析JSON格式的命令序列
            print(f"action parts: {parts}", file=sys.stderr)

            if len(parts) != 2:
                return "error: sequence requires a JSON string of commands, currently got: " + f'{len(parts)} parts'
            
            # 处理JSON字符串
            json_str = parts[1]
            
            # 打印调试信息
            print(f"接收到的JSON字符串: {repr(json_str)}", file=sys.stderr)
            
            # 确保JSON字符串是数组格式
            json_str = json_str.strip()
            if not (json_str.startswith('[') and json_str.endswith(']')):
                # 如果不是数组格式，尝试添加方括号
                if json_str.startswith('"') or json_str.startswith("'"):
                    # 可能是单个命令，将其包装成数组
                    json_str = f"[{json_str}]"
                else:
                    # 尝试将整个字符串当作数组内容
                    json_str = f"[{json_str}]"
                print(f"添加数组括号后: {repr(json_str)}", file=sys.stderr)
            
            # 如果JSON字符串被额外的引号包裹，去掉外层引号
            if (json_str.startswith("'[") and json_str.endswith("]'")) or \
               (json_str.startswith('"[') and json_str.endswith(']"')):
                json_str = json_str[1:-1]
                print(f"移除外层引号后: {repr(json_str)}", file=sys.stderr)
            
            # 尝试手动处理常见转义问题
            try:
                commands = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"尝试修复JSON格式: {e}", file=sys.stderr)
                
                # 替换常见转义序列
                fixed_json = json_str.replace('\\\\\"', '\\"').replace('\\\\"', '\\"')
                
                # 进一步处理可能的格式问题
                if not (fixed_json.startswith('[') and fixed_json.endswith(']')):
                    # 可能是格式化问题，尝试提取命令部分并重新构建JSON
                    cmd_pattern = r'double_click\s+\d+\s+\d+|screenshot|click\s+\d+\s+\d+|type\s+".+"'
                    commands_found = re.findall(cmd_pattern, fixed_json)
                    if commands_found:
                        fixed_json = json.dumps(commands_found)
                    else:
                        # 最后尝试：将逗号分隔的字符串转为JSON数组
                        parts = [p.strip().strip('"\'') for p in fixed_json.strip('[]').split(',')]
                        if len(parts) > 0:
                            fixed_json = json.dumps(parts)
                
                print(f"修复后的JSON: {repr(fixed_json)}", file=sys.stderr)
                commands = json.loads(fixed_json)
            results = []
            
            for cmd in commands:
                # 执行每个命令并收集结果
                result = parse_and_execute(cmd)
                if result:
                    results.append(result)
            
            # 如果有返回结果，以分号分隔返回
            if results:
                return "; ".join(results)
            return "sequence completed"
        except json.JSONDecodeError as e:
            print(f"JSON解析错误：{e}", file=sys.stderr)
            return f"error: invalid JSON format: {e}"
        except Exception as e:
            print(f"执行命令序列时出错：{e}", file=sys.stderr)
            return f"error: sequence execution failed: {e}"

    try:
        if action == "click":
            # click x y
            x = int(parts[1]); y = int(parts[2])
            pyautogui.click(x=x, y=y)

        elif action == "press_enter":
            pyautogui.click(524, 657)  # 如果需要指定位置
            pyautogui.press("enter")

        elif action == "ratio_click":
            rx = float(parts[1]); ry = float(parts[2])
            x, y = ratio_to_pixel(rx, ry)
            pyautogui.click(x=x, y=y)

        elif action in ("double_click", "ratio_double_click"):
            # 解析坐标
            if action == "ratio_double_click":
                rx, ry = float(parts[1]), float(parts[2])
                x, y   = ratio_to_pixel(rx, ry)
                idx     = 3
            else:
                x, y   = int(parts[1]), int(parts[2])
                idx     = 3

            # 可选：两次点击间隔
            interval = float(parts[idx]) if len(parts) > idx else 0.2

            # 移到目标
            pyautogui.moveTo(x, y)
            time.sleep(0.05)

            # 先click一次
            pyautogui.click(x=x, y=y)
            time.sleep(0.1)
            # 调用 cliclick 发双击
            subprocess.run(["cliclick", f"dc:{x},{y}"], check=True)

            # 等待间隔，保持行为一致
            time.sleep(interval)

        elif action == "right_click":
            x = int(parts[1]); y = int(parts[2])
            pyautogui.rightClick(x=x, y=y)

        elif action == "move":
            x = int(parts[1]); y = int(parts[2])
            duration = float(parts[3]) if len(parts) >= 4 else 0
            pyautogui.moveTo(x, y, duration=duration)

        elif action == "ratio_move":
            rx = float(parts[1]); ry = float(parts[2])
            x, y = ratio_to_pixel(rx, ry)
            duration = float(parts[3]) if len(parts) >= 4 else 0
            pyautogui.moveTo(x, y, duration=duration)

        elif action == "drag":
            x1 = int(parts[1]); y1 = int(parts[2])
            x2 = int(parts[3]); y2 = int(parts[4])
            duration = float(parts[5]) if len(parts) >= 6 else 0
            button = parts[6] if len(parts) >= 7 else 'left'
            pyautogui.moveTo(x1, y1)
            pyautogui.dragTo(x2, y2, duration=duration, button=button)

        elif action == "ratio_drag":
            rx1 = float(parts[1]); ry1 = float(parts[2])
            rx2 = float(parts[3]); ry2 = float(parts[4])
            x1, y1 = ratio_to_pixel(rx1, ry1)
            x2, y2 = ratio_to_pixel(rx2, ry2)
            duration = float(parts[5]) if len(parts) >= 6 else 0
            button = parts[6] if len(parts) >= 7 else 'left'
            pyautogui.moveTo(x1, y1)
            pyautogui.dragTo(x2, y2, duration=duration, button=button)

        elif action == "scroll":
            clicks = int(parts[1])
            pyautogui.scroll(clicks)

        elif action == "type":
            text = parts[1]
            pyautogui.write(text)

        elif action == "press":
            key = parts[1]
            pyautogui.press(key)

        elif action == "hotkey":
            keys = parts[1:]
            pyautogui.hotkey(*keys)

        elif action == "position":
            x, y = pyautogui.position()
            print(f"当前鼠标位置: x={x}, y={y}")
            return f"position {x} {y}"

        elif action == "position_ratio":
            x, y = pyautogui.position()
            w, h = pyautogui.size()
            print(f"当前鼠标比例位置: ratio_x={x/w:.4f}, ratio_y={y/h:.4f}")
            return f"position_ratio {x/w:.4f} {y/h:.4f}"

        elif action == "screenshot":
            filename = parts[1] if len(parts) >= 2 else datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
            img = pyautogui.screenshot()
            img.save(filename)
            print(f"截图已保存为: {filename}")
            return f"screenshot {filename}"

        elif action == "region_screenshot":
            x, y, w, h = map(int, parts[1:5])
            filename = parts[5] if len(parts) >= 6 else datetime.now().strftime("region_%Y%m%d_%H%M%S.png")
            img = pyautogui.screenshot(region=(x, y, w, h))
            img.save(filename)
            print(f"区域截图已保存为: {filename}")
            return f"region_screenshot {filename}"

        elif action == "pixel_color":
            x = int(parts[1]); y = int(parts[2])
            color = pyautogui.pixel(x, y)
            print(f"位置 ({x}, {y}) 的像素颜色: RGB{color}")
            return f"pixel_color {x} {y} {color}"

        elif action == "wait_for_image":
            path = parts[1]
            confidence = float(parts[2]) if len(parts) >= 3 else 0.9
            timeout    = float(parts[3]) if len(parts) >= 4 else 30
            start = time.time()
            while time.time() - start < timeout:
                loc = pyautogui.locateOnScreen(path, confidence=confidence)
                if loc:
                    cx, cy = pyautogui.center(loc)
                    print(f"找到图像 '{path}' 在: ({cx}, {cy})")
                    return f"found {cx} {cy}"
                time.sleep(0.5)
            print(f"超时 {timeout}s，未找到 '{path}'")
            return "not_found"

        elif action == "sleep":
            seconds = float(parts[1])
            time.sleep(seconds)

        else:
            print(f"错误：未知动作 '{action}'", file=sys.stderr)
            return f"error: unknown_action {action}"

    except Exception as e:
        print(f"执行 '{action}' 时出错：{e}", file=sys.stderr)
        return f"error: {action} {e}"

def show_help():
    help_text = """
可用命令:
  click x y                                   - 在指定坐标点击
  ratio_click rx ry                           - 按屏幕比例坐标点击
  double_click x y [interval] [button]        - 在指定坐标双击
  ratio_double_click rx ry [interval]         - 按屏幕比例坐标双击
  right_click x y                             - 在指定坐标右键点击
  move x y [duration]                         - 移动到指定坐标
  ratio_move rx ry [duration]                 - 按屏幕比例移动
  drag x1 y1 x2 y2 [duration] [button]        - 拖拽
  ratio_drag rx1 ry1 rx2 ry2 [duration]       - 按屏幕比例拖拽
  scroll clicks                               - 滚动
  type "text"                                 - 输入文本
  press key                                   - 按键
  hotkey key1 key2 ...                        - 组合键
  position                                    - 获取当前鼠标位置
  position_ratio                              - 获取当前鼠标的屏幕比例位置
  screenshot [filename]                       - 截图
  region_screenshot x y w h [filename]        - 区域截图
  pixel_color x y                             - 获取指定位置的像素颜色
  wait_for_image path [conf] [timeout]        - 等待图像出现
  sleep seconds                               - 等待指定秒数
  sequence '["cmd1", "cmd2", ...]'            - 执行命令序列
  help                                        - 显示帮助
"""
    print(help_text, file=sys.stderr)

def main():
    print("GUI Agent 启动，等待指令。Ctrl+D / Ctrl+Z+回车 退出。", file=sys.stderr)
    print("屏幕尺寸:", pyautogui.size(), file=sys.stderr)

    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            if line.lower() == "help":
                show_help()
            else:
                res = parse_and_execute(line)
                if res:
                    print(res)
                    sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nGUI Agent 已退出。", file=sys.stderr)

if __name__ == "__main__":
    main()
