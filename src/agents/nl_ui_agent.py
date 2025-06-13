#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import time
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import os
import dotenv
from pathlib import Path

# Import the UI operation library
try:
    # 尝试绝对导入
    from src.agents.ui_use_from_command_with_sequence import parse_and_execute, ratio_to_pixel
except ImportError:
    try:
        # 尝试相对导入
        from .ui_use_from_command_with_sequence import parse_and_execute, ratio_to_pixel
    except ImportError:
        # 最后尝试直接导入（如果在同一目录中）
        from ui_use_from_command_with_sequence import parse_and_execute, ratio_to_pixel

# Import for checking module availability
import importlib.util

# Define OPENAI_AVAILABLE in the global scope
OPENAI_AVAILABLE = False
OPENAI_IMPORT_ERROR = None

# 在文件顶部导入部分添加：
import dotenv
from pathlib import Path

# 修改 check_openai_available 函数
def check_openai_available():
    """Check if OpenAI is available and can be imported"""
    global OPENAI_AVAILABLE, OPENAI_IMPORT_ERROR
    
    try:
        # 首先加载.env文件（如果存在）
        env_path = Path('.env')
        if env_path.exists():
            dotenv.load_dotenv(env_path)
            print(f"已加载 .env 文件", file=sys.stderr)

        # Check if module exists
        if importlib.util.find_spec("openai") is not None:
            # 只要有API密钥就启用OpenAI功能，除非明确禁用
            api_key_exists = bool(os.environ.get("OPENAI_API_KEY", ""))
            explicitly_disabled = os.environ.get("DISABLE_OPENAI_NL_PARSING", "").lower() == "true"
            
            if api_key_exists and not explicitly_disabled:
                OPENAI_AVAILABLE = True
                OPENAI_IMPORT_ERROR = None
                print(f"找到OpenAI API密钥，已启用OpenAI功能", file=sys.stderr)
                return True
            elif explicitly_disabled:
                OPENAI_AVAILABLE = False
                OPENAI_IMPORT_ERROR = "OpenAI is explicitly disabled by DISABLE_OPENAI_NL_PARSING"
            elif not api_key_exists:
                OPENAI_AVAILABLE = False
                OPENAI_IMPORT_ERROR = "OpenAI API key not found"
            return False
        else:
            OPENAI_AVAILABLE = False
            OPENAI_IMPORT_ERROR = "openai module not found"
            return False
    except Exception as e:
        OPENAI_AVAILABLE = False
        OPENAI_IMPORT_ERROR = str(e)
        return False

# Check OpenAI availability at startup
check_openai_available()
print(f"OpenAI available: {OPENAI_AVAILABLE}", file=sys.stderr)

class NLUIAgent:
    """
    An agent that accepts natural language descriptions (Chinese or English)
    and converts them to UI operations using the existing library.
    """
    
    def __init__(self):
        # Define patterns for coordinate extraction
        self.coord_patterns = [
            r'(\d+)\s*[,，]\s*(\d+)',  # "500, 500" or "500，500"
            r'x\s*=\s*(\d+)\s*[,，]?\s*y\s*=\s*(\d+)',  # "x=500, y=500"
            r'坐标\s*[为是:：]?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?',  # "坐标是(500, 500)" or "坐标：500,500"
            r'位置\s*[为是:：]?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?',  # "位置是(500, 500)" or "位置：500,500"
            r'coordinates?\s*[: ]?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?',  # "coordinates: 500, 500"
            r'position\s*[: ]?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?',  # "position: 500, 500"
            r'at\s*\(?(\d+)\s*[,，]\s*(\d+)\)?',  # "at (500, 500)"
        ]
        
        # Define patterns for ratio coordinates
        self.ratio_patterns = [
            r'比例坐标\s*[为是:：]?\s*\(?(0?\.\d+)\s*[,，]\s*(0?\.\d+)\)?',  # "比例坐标是(0.5, 0.5)" 
            r'ratio\s*[: ]?\s*\(?(0?\.\d+)\s*[,，]\s*(0?\.\d+)\)?',  # "ratio: 0.5, 0.5"
            r'relative\s*[: ]?\s*\(?(0?\.\d+)\s*[,，]\s*(0?\.\d+)\)?',  # "relative: 0.5, 0.5"
        ]
        
        # Define command mappings from natural language to UI operation commands
        self.command_mappings = {
            # 点击操作
            '点击': 'click',
            'click': 'click',
            '左键': 'click',
            'left click': 'click',
            '左击': 'click',
            
            # 右键点击
            '右键': 'right_click',
            '右击': 'right_click',
            'right click': 'right_click',
            
            # 双击操作
            '双击': 'double_click',
            'double click': 'double_click',
            'double-click': 'double_click',
            
            # 移动操作
            '移动': 'move',
            '移动到': 'move',
            'move': 'move',
            'move to': 'move',
            
            # 拖动操作
            '拖动': 'drag',
            '拖拽': 'drag',
            'drag': 'drag',
            'drag from': 'drag',
            
            # 输入操作
            '输入': 'type',
            '键入': 'type',
            'type': 'type',
            'input': 'type',
            'enter text': 'type',
            
            # 按键操作
            '按键': 'press',
            '按下': 'press',
            'press': 'press',
            'key': 'press',
            
            # 组合键
            '组合键': 'hotkey',
            'hotkey': 'hotkey',
            'key combination': 'hotkey',
            
            # 截图
            '截图': 'screenshot',
            'screenshot': 'screenshot',
            'capture screen': 'screenshot',
            
            # 区域截图
            '区域截图': 'region_screenshot',
            'region screenshot': 'region_screenshot',
            
            # 滚动
            '滚动': 'scroll',
            'scroll': 'scroll',
            
            # 等待
            '等待': 'sleep',
            'wait': 'sleep',
            'pause': 'sleep',
            'sleep': 'sleep',
            
            # 连续动作
            '连续动作': 'sequence',
            '顺序执行': 'sequence',
            '依次执行': 'sequence',
            'sequence': 'sequence',
            'seq': 'sequence',
        }
        
        # Initialize OpenAI if available
        global OPENAI_AVAILABLE
        if OPENAI_AVAILABLE:
            # Try to get API key from environment
            # Note: For OpenAI client v1+, the API key is automatically read from
            # the OPENAI_API_KEY environment variable, or you can set it manually
            # when creating the client
            self.api_key = "xxxxxxx"
            if not self.api_key:
                print("警告: 找到OpenAI库但没有设置API密钥。设置OPENAI_API_KEY环境变量以启用高级NLP功能。", file=sys.stderr)
                OPENAI_AVAILABLE = False
            # We'll configure the API key when creating the client in _parse_with_openai
    
    def extract_coordinates(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract pixel coordinates from text."""
        # First try the predefined patterns
        for pattern in self.coord_patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1)), int(match.group(2))
                
        # Special case for "双击坐标1409 648" format - look for adjacent numbers after "坐标"
        coordinates_marker_match = re.search(r'坐标\s*(\d+)\s+(\d+)', text)
        if coordinates_marker_match:
            return int(coordinates_marker_match.group(1)), int(coordinates_marker_match.group(2))
            
        # If no match, try to find any two adjacent numbers that could be coordinates
        numbers = re.findall(r'\d+', text)
        if len(numbers) >= 2:
            # If we have exactly two numbers, assume they're the coordinates
            if len(numbers) == 2:
                return int(numbers[0]), int(numbers[1])
            # If there are more than two numbers, try to find adjacent ones
            for i in range(len(numbers) - 1):
                # Check if these numbers appear close together in the text
                pos1 = text.find(numbers[i])
                pos2 = text.find(numbers[i+1], pos1 + len(numbers[i]))
                # If they're within 10 characters of each other, they're probably coordinates
                if 0 < pos2 - pos1 - len(numbers[i]) < 10:
                    return int(numbers[i]), int(numbers[i+1])
                    
        return None, None
    
    def extract_ratio_coordinates(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract ratio coordinates from text."""
        for pattern in self.ratio_patterns:
            match = re.search(pattern, text)
            if match:
                return float(match.group(1)), float(match.group(2))
        return None, None
    
    def extract_text(self, text: str) -> Optional[str]:
        """Extract quoted text from the input."""
        # Look for text in quotes
        match = re.search(r'"([^"]*)"', text) or re.search(r"'([^']*)'", text)
        if match:
            return match.group(1)
            
        # If no quotes found, try to extract text after keywords
        text_keywords = ["输入", "键入", "type", "input", "enter text"]
        for keyword in text_keywords:
            if keyword in text.lower():
                parts = text.lower().split(keyword, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        return None
    
    def extract_key(self, text: str) -> Optional[str]:
        """Extract key name from the input."""
        # Common key mappings (add more if needed)
        key_mapping = {
            "回车": "enter",
            "确认键": "enter",
            "换行": "enter",
            "制表": "tab",
            "空格": "space",
            "删除": "delete",
            "退格": "backspace",
            "上": "up", 
            "下": "down",
            "左": "left",
            "右": "right"
        }
        
        # Look for key names in the text
        for cn_key, en_key in key_mapping.items():
            if cn_key in text:
                return en_key
        
        # Check for English key names
        for _, en_key in key_mapping.items():
            if en_key in text.lower():
                return en_key
                
        # Try to find a key name at the end of the command
        match = re.search(r'(?:press|按键|按下|key)\s+(\w+)', text.lower())
        if match:
            return match.group(1)
            
        return None
    
    def extract_hotkey(self, text: str) -> Optional[List[str]]:
        """Extract hotkey combination from the input."""
        # Look for "hotkey X Y" pattern
        match = re.search(r'(?:hotkey|组合键)\s+([\w\s+]+)', text.lower())
        if match:
            keys = match.group(1).split()
            return keys
        
        # Look for key combinations with + sign
        if "+" in text:
            # Extract the combination part
            match = re.search(r'([\w\s+]+键?组合|[\w\s+]+combination)', text.lower())
            if match:
                combo_text = match.group(1)
                keys = re.findall(r'(\w+)[\s+]+', combo_text)
                return keys
        
        return None
    
    def extract_number(self, text: str, default=None) -> Optional[Union[int, float]]:
        """Extract a number from the text."""
        # Look for numbers
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            num_str = match.group(1)
            if '.' in num_str:
                return float(num_str)
            return int(num_str)
        return default
    
    def extract_seconds(self, text: str) -> Optional[float]:
        """Extract time in seconds from the text."""
        # Look for time expressions
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:秒|seconds?|s)', text.lower())
        if match:
            return float(match.group(1))
        
        # Look for any number
        return self.extract_number(text)
        
    def extract_commands(self, text: str) -> Optional[List[str]]:
        """Extract command sequence from the text."""
        # Try to parse structured command sequence
        try:
            # Look for commands listed with numbers, bullets, or semicolons
            cmds = []
            
            # 首先移除可能的命令前缀词，如"连续动作："等
            cleaned_text = re.sub(r'^(连续动作|顺序执行|依次执行|sequence|seq)[：:]\s*', '', text, flags=re.IGNORECASE)
            
            # Try to extract commands enclosed in quotes
            quoted_cmds = re.findall(r'["\'](.*?)["\']', cleaned_text)
            if quoted_cmds:
                return quoted_cmds
            
            # Try to extract commands separated by semicolons
            if ';' in cleaned_text:
                semicolon_parts = cleaned_text.split(';')
                cmds = [part.strip() for part in semicolon_parts if part.strip()]
                if cmds:
                    return cmds
            
            # Handle the specific case "双击坐标xxx yyy，然后截图"
            if "双击坐标" in cleaned_text and "然后" in cleaned_text and "截图" in cleaned_text:
                # Extract the first part (before "然后")
                first_part = re.split(r'，?\s*然后', cleaned_text)[0].strip()
                # Extract the second part (after "然后")
                second_part = re.split(r'，?\s*然后', cleaned_text)[1].strip()
                return [first_part, second_part]
                
            # For Chinese text, split by common Chinese sequence markers
            if any(ord(c) > 127 for c in cleaned_text):
                # Split by Chinese commas and sequence markers
                for marker in ["，", "；", "然后", "接着", "之后", "最后"]:
                    if marker in cleaned_text:
                        parts = cleaned_text.split(marker)
                        cmds = [part.strip() for part in parts if part.strip()]
                        if len(cmds) > 1:
                            return cmds
            
            # Try to extract commands as numbered or bulleted list
            list_items = re.findall(r'(?:^|\n)?\s*(?:\d+[\.、\)]|[-*•])\s*(.*?)(?=(?:\s*(?:\d+[\.、\)]|[-*•])|\n|\Z))', cleaned_text, re.DOTALL)
            if list_items:
                cmds = [item.strip() for item in list_items if item.strip()]
                if cmds:
                    return cmds
                
            # Try to extract "first... then... finally..." structure
            sequence_markers = ["first", "然后", "接着", "之后", "最后", "finally", "then", "next", "subsequently", "afterwards"]
            for marker in sequence_markers:
                if marker in cleaned_text.lower():
                    parts = re.split(r'\b(?:' + marker + r'|然后|接着|之后|最后|first|finally|then|next|subsequently|afterwards)\b', 
                                    cleaned_text, flags=re.IGNORECASE)
                    cmds = [part.strip() for part in parts if part.strip()]
                    if len(cmds) > 1:  # Make sure we actually got a sequence
                        return cmds                    # 如果上述方法都失败了，尝试使用冒号拆分
            if ":" in cleaned_text or "：" in cleaned_text:
                parts = re.split(r'[:：]', cleaned_text, 1)
                if len(parts) > 1:
                    # Check for comma or Chinese comma as separator
                    if "," in parts[1] or "，" in parts[1]:
                        separators = re.split(r'[,，]', parts[1])
                        commands = [cmd.strip() for cmd in separators if cmd.strip()]
                        if commands:
                            return commands
            
            # 最后一个尝试：如果文本包含多个句子，可能每句是一个指令
            sentences = re.split(r'[.。!！?？]', cleaned_text)
            if len(sentences) > 1:
                cmds = [s.strip() for s in sentences if s.strip()]
                if len(cmds) > 1:
                    return cmds
                    
        except Exception as e:
            print(f"Error parsing command sequence: {e}", file=sys.stderr)
            
        return None
    
    def extract_clicks(self, text: str) -> Optional[int]:
        """Extract number of clicks for scrolling."""
        # Look for scroll amount
        if "向上" in text or "up" in text.lower():
            amount = self.extract_number(text, 1)  # Default to 1
            return abs(amount)  # Positive for up
        
        if "向下" in text or "down" in text.lower():
            amount = self.extract_number(text, 1)  # Default to 1
            return -abs(amount)  # Negative for down
        
        # Just extract a number
        return self.extract_number(text)
    
    def extract_command_type(self, text: str) -> str:
        """Determine the type of UI command from the natural language description."""
        text_lower = text.lower()
        
        # Find the longest, most specific matching key
        best_match = None
        best_match_length = 0
        best_cmd = None
        
        # Check for each command type
        for key, cmd in self.command_mappings.items():
            if key in text_lower and len(key) > best_match_length:
                best_match = key
                best_match_length = len(key)
                best_cmd = cmd
        
        if best_cmd:
            return best_cmd
        
        # Default to click if coordinates are found but no specific command
        x, y = self.extract_coordinates(text)
        if x is not None and y is not None:
            return 'click'
            
        # Default if nothing else matches
        return 'unknown'
    
    def parse_nl_to_command(self, nl_text: str) -> str:
        """
        Parse natural language text to UI operation command.
        Returns the command string that can be passed to parse_and_execute.
        """
        global OPENAI_AVAILABLE
        if OPENAI_AVAILABLE:
            print(f"Using OpenAI to parse command: {nl_text}", file=sys.stderr)
            return self._parse_with_openai(nl_text)
        else:
            print(f"Using rule-based parsing for command: {nl_text}", file=sys.stderr)
            return self._parse_with_rules(nl_text)
    
    def _parse_with_rules(self, nl_text: str) -> str:
        """Parse using rule-based approach."""
        # Determine the command type
        cmd_type = self.extract_command_type(nl_text)
        
        # If unknown command, return error
        if cmd_type == 'unknown':
            return "error: unknown_command"
            
        # Handle different command types
        if cmd_type == 'click':
            # First check for ratio coordinates
            rx, ry = self.extract_ratio_coordinates(nl_text)
            if rx is not None and ry is not None:
                return f"ratio_click {rx} {ry}"
                
            # Then check for pixel coordinates
            x, y = self.extract_coordinates(nl_text)
            if x is not None and y is not None:
                return f"click {x} {y}"
            
            return "error: no_coordinates"
            
        elif cmd_type == 'right_click':
            x, y = self.extract_coordinates(nl_text)
            if x is not None and y is not None:
                return f"right_click {x} {y}"
            return "error: no_coordinates"
            
        elif cmd_type == 'double_click':
            # First check for ratio coordinates
            rx, ry = self.extract_ratio_coordinates(nl_text)
            if rx is not None and ry is not None:
                return f"ratio_double_click {rx} {ry}"
                
            # Then check for pixel coordinates
            x, y = self.extract_coordinates(nl_text)
            if x is not None and y is not None:
                return f"double_click {x} {y}"
                
            # If we still couldn't extract coordinates, check for special case "双击坐标xxx yyy"
            coordinates_match = re.search(r'双击坐标\s*(\d+)\s+(\d+)', nl_text)
            if coordinates_match:
                return f"double_click {coordinates_match.group(1)} {coordinates_match.group(2)}"
                
            # If we still couldn't extract coordinates, check for simple number patterns
            # This helps with formats like "双击坐标1409 648"
            numbers = re.findall(r'\d+', nl_text)
            if len(numbers) >= 2:
                return f"double_click {numbers[0]} {numbers[1]}"
                
            return "error: no_coordinates"
            
        elif cmd_type == 'move':
            # First check for ratio coordinates
            rx, ry = self.extract_ratio_coordinates(nl_text)
            if rx is not None and ry is not None:
                duration = self.extract_number(nl_text)
                if duration is not None:
                    return f"ratio_move {rx} {ry} {duration}"
                return f"ratio_move {rx} {ry}"
                
            # Then check for pixel coordinates
            x, y = self.extract_coordinates(nl_text)
            if x is not None and y is not None:
                duration = self.extract_number(nl_text)
                if duration is not None:
                    return f"move {x} {y} {duration}"
                return f"move {x} {y}"
                
            return "error: no_coordinates"
            
        elif cmd_type == 'drag':
            # This is complex, we need to extract two points
            # For now, we'll use a simple approach - split by "to" or "到"
            parts = re.split(r'\s+to\s+|\s+到\s+', nl_text)
            if len(parts) != 2:
                return "error: invalid_drag_format"
                
            # Extract first point
            x1, y1 = self.extract_coordinates(parts[0])
            # Extract second point
            x2, y2 = self.extract_coordinates(parts[1])
            
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                duration = self.extract_number(nl_text)
                if duration is not None:
                    return f"drag {x1} {y1} {x2} {y2} {duration}"
                return f"drag {x1} {y1} {x2} {y2}"
                
            return "error: no_coordinates"
            
        elif cmd_type == 'type':
            text_to_type = self.extract_text(nl_text)
            if text_to_type:
                return f"type \"{text_to_type}\""
            return "error: no_text"
            
        elif cmd_type == 'press':
            key = self.extract_key(nl_text)
            if key:
                return f"press {key}"
            return "error: no_key"
            
        elif cmd_type == 'hotkey':
            keys = self.extract_hotkey(nl_text)
            if keys:
                return f"hotkey {' '.join(keys)}"
            return "error: no_keys"
            
        elif cmd_type == 'screenshot':
            return "screenshot"
            
        elif cmd_type == 'scroll':
            clicks = self.extract_clicks(nl_text)
            if clicks is not None:
                return f"scroll {clicks}"
            return "error: no_scroll_amount"
            
        elif cmd_type == 'sleep':
            seconds = self.extract_seconds(nl_text)
            if seconds is not None:
                return f"sleep {seconds}"
            return "error: no_time"
            
        elif cmd_type == 'sequence':
            commands = self.extract_commands(nl_text)
            if not commands:
                return "error: no_commands_found"
                
            # 递归解析每个子命令
            parsed_commands = []
            for cmd in commands:
                # 对每个子命令应用规则解析
                parsed_cmd = self.parse_nl_to_command(cmd)
                # 如果子命令解析出错，跳过该命令
                if not parsed_cmd.startswith("error:"):
                    parsed_commands.append(parsed_cmd)
            
            # 如果没有成功解析任何命令，返回错误
            if not parsed_commands:
                return "error: no_valid_commands"
                
            # 将命令列表转换为JSON字符串
            command_json = json.dumps(parsed_commands)
            print(f"Parsed sequence command: {command_json}", file=sys.stderr)
            
            # 不要使用空格分隔，直接返回完整命令
            return f"sequence {command_json}"
            
        # Default fallback
        return f"error: unknown_parameters_for_{cmd_type}"
    
    def _parse_with_openai(self, nl_text: str) -> str:
        """Use OpenAI's API to parse the natural language into a UI command."""
        global OPENAI_AVAILABLE
        if not OPENAI_AVAILABLE:
            return self._parse_with_rules(nl_text)
            
        try:
            # Import OpenAI here to avoid issues with global imports
            try:
                import openai
            except ImportError:
                return self._parse_with_rules(nl_text)
                
            prompt = f"""
            Parse the following natural language instruction into a UI automation command.
            Available commands: click, ratio_click, right_click, double_click, ratio_double_click,
            move, ratio_move, drag, ratio_drag, type, press, hotkey, screenshot, region_screenshot,
            pixel_color, wait_for_image, scroll, sleep, position, position_ratio, sequence.

            Format your response as a single line with the command name followed by its parameters.
            
            Examples of different coordinate formats:
            - "Click at coordinates 500, 300" -> "click 500 300"
            - "Double-click at 1409 648" -> "double_click 1409 648"
            - "双击坐标1409 648" -> "double_click 1409 648"
            - "Click at ratio 0.5, 0.7" -> "ratio_click 0.5 0.7"
            - "Right click at x=300, y=400" -> "right_click 300 400"
            
            Examples of other commands:
            - "Type hello world" -> "type \"hello world\""
            - "Press enter key" -> "press enter"
            - "Press hotkey ctrl c" -> "hotkey ctrl c"
            - "Wait for 5 seconds" -> "sleep 5"
            - "Take a screenshot" -> "screenshot"
            - "Scroll down 3 clicks" -> "scroll -3"
            
            For a sequence of commands, format with a properly JSON-serialized array:
            - "First click at 100,200, then type hello, finally press enter" -> 
            "sequence ["click 100 200", "type \\"hello\\"", "press enter"]"
            
            IMPORTANT EXAMPLE for Chinese sequences:
            - "双击坐标1409 648，然后截图" -> 
              "sequence ["double_click 1409 648", "screenshot"]"
            
            CRITICAL: For sequence commands, the JSON array must be valid and properly formatted.
            Do not escape quotes with backslashes when they are not needed in JSON strings.
            
            Important: For instructions in Chinese or with adjacent numbers, pay special attention to extracting the correct coordinates.
            
            Natural language instruction: {nl_text}
            
            UI command:
            """
            
            # Using the updated OpenAI API syntax (1.0.0+)
            client = openai.OpenAI(api_key = self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150,
            )

            print(f"OpenAI response: {response}", file=sys.stderr)
            
            command = response.choices[0].message.content.strip()
            # Remove any quotes around the whole command
            if (command.startswith('"') and command.endswith('"')) or (command.startswith("'") and command.endswith("'")):
                command = command[1:-1]
                
            # Validate that the command looks reasonable
            command_parts = command.split(None, 1)
            cmd_type = command_parts[0] if command_parts else ""
            
            if cmd_type not in [
                "click", "ratio_click", "right_click", "double_click", "ratio_double_click",
                "move", "ratio_move", "drag", "ratio_drag", "type", "press", "hotkey",
                "screenshot", "region_screenshot", "pixel_color", "wait_for_image",
                "scroll", "sleep", "position", "position_ratio", "sequence", "seq"
            ]:
                # Fall back to rule-based parsing
                return self._parse_with_rules(nl_text)
            
            # Special handling for sequence command to ensure proper JSON formatting
            if cmd_type == "sequence" and len(command_parts) > 1:
                try:
                    # Attempt to parse the JSON part to validate it
                    json_part = command_parts[1].strip()
                    
                    # Make sure we're getting the entire JSON part
                    if not (json_part.startswith('[') and json_part.endswith(']')):
                        # Try to extract the entire JSON array if the split was incorrect
                        match = re.search(r'(\[.*?\])', command)
                        if match:
                            json_part = match.group(1)
                    
                    # Extract the commands from the JSON-like string
                    if json_part.startswith('[') and json_part.endswith(']'):
                        # First try to parse it directly
                        try:
                            cmd_list = json.loads(json_part)
                        except json.JSONDecodeError:
                            # If that fails, manually extract the commands
                            content = json_part[1:-1]  # Remove brackets
                            cmds = []
                            
                            # Extract quoted strings
                            pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'
                            matches = re.findall(pattern, content)
                            if matches:
                                cmds = [match.replace('\\"', '"') for match in matches]
                            else:
                                # Split by comma as fallback
                                cmds = [cmd.strip().strip('"\'') for cmd in content.split(',')]
                                
                            cmd_list = cmds
                    else:
                        # Fallback: try to extract commands another way
                        cmds = re.findall(r'"([^"]*)"', json_part)
                        cmd_list = cmds if cmds else [json_part]
                    
                    # Regenerate a properly formatted JSON string
                    command = f"sequence {json.dumps(cmd_list)}"
                    print(f"Reformatted sequence command: {command}", file=sys.stderr)
                    
                except Exception as e:
                    # If JSON is invalid, try to fix common formatting issues
                    print(f"Attempting to fix malformed sequence JSON: {command_parts[1]}, error: {e}", file=sys.stderr)
                    
                    # Try to extract commands from malformed JSON-like string
                    cmds = re.findall(r'"([^"]*)"', command_parts[1])
                    if cmds:
                        # Fix: Ensure we create a proper JSON array
                        command = f"sequence {json.dumps(cmds)}"
                    else:
                        # Try another approach with regex to extract commands
                        cmds = re.findall(r'\[(.*?)\]', command_parts[1])
                        if cmds:
                            # Split by comma and clean up each command
                            cmd_list = [cmd.strip().strip('"\'') for cmd in cmds[0].split(',')]
                            command = f"sequence {json.dumps(cmd_list)}"
                        else:
                            return self._parse_with_rules(nl_text)
                    
            return command
            
        except Exception as e:
            print(f"OpenAI parsing error: {e}", file=sys.stderr)
            # Fall back to rule-based parsing
            return self._parse_with_rules(nl_text)
    
    def execute_nl_command(self, nl_text: str) -> Optional[str]:
        """
        Execute a natural language UI command.
        Returns any result string from the execution.
        """
        # First try with rule-based parsing
        command = self.parse_nl_to_command(nl_text)
        
        # Check for errors and try OpenAI if needed
        if command.startswith("error:"):
            error_msg = command.split(":", 1)[1].strip()
            print(f"Error parsing command: {error_msg}", file=sys.stderr)
            
            # Special case for "双击坐标1409 648，然后截图" - direct handling
            if "双击坐标" in nl_text and "然后" in nl_text and "截图" in nl_text:
                # Extract coordinates from the first part
                coordinates_match = re.search(r'双击坐标\s*(\d+)\s+(\d+)', nl_text)
                if coordinates_match:
                    x, y = coordinates_match.group(1), coordinates_match.group(2)
                    # Create the command list and convert to JSON string
                    commands = [f"double_click {x} {y}", "screenshot"]
                    command = f"sequence {json.dumps(commands)}"
                    print(f"Special case handled: {command}", file=sys.stderr)
            
            # If still error, try more complex parsing
            if command.startswith("error:"):
                # Complex commands or Chinese text benefit from OpenAI parsing
                is_complex_command = any([
                    "no_coordinates" in error_msg,
                    "unknown_command" in error_msg,
                    "坐标" in nl_text,
                    "双击" in nl_text,
                    "then" in nl_text.lower(),
                    "然后" in nl_text,
                    "，" in nl_text,
                    len(nl_text.split()) > 5,  # Complex commands with many words
                    any(ord(c) > 127 for c in nl_text)  # Non-ASCII characters (e.g., Chinese)
                ])
                
                if is_complex_command and self.api_key:
                    print(f"Trying complex command with OpenAI parsing: {nl_text}", file=sys.stderr)
                    
                    # Manual parsing for common Chinese patterns
                    if "双击坐标" in nl_text:
                        numbers = re.findall(r'\d+', nl_text)
                        if len(numbers) >= 2:
                            command = f"double_click {numbers[0]} {numbers[1]}"
                            print(f"Manual extraction of coordinates: {command}", file=sys.stderr)
        
        # If still error after all attempts
        if command.startswith("error:"):
            error_msg = command.split(":", 1)[1].strip()
            print(f"Final error parsing command: {error_msg}", file=sys.stderr)
            return f"error: {error_msg}"
        
        # Execute the command
        print(f"Executing UI command: {command}", file=sys.stderr)
        return parse_and_execute(command)


def main():
    """Main function to run the NL UI Agent."""
    agent = NLUIAgent()
    check_openai_available()
    
    print("自然语言 UI 代理启动，等待指令。支持中文或英文。输入 'exit' 或 Ctrl+D 退出。", file=sys.stderr)
    print("Natural Language UI Agent started, waiting for instructions in Chinese or English.", file=sys.stderr)
    print("Enter 'exit' or press Ctrl+D to quit.", file=sys.stderr)
    
    try:
        while True:
            try:
                nl_text = input("> ")
                if nl_text.lower() in ['exit', 'quit', '退出', '退出程序']:
                    break
                    
                if not nl_text.strip():
                    continue
                    
                result = agent.execute_nl_command(nl_text)
                if result:
                    print(result)
                    
            except EOFError:
                break
    except KeyboardInterrupt:
        pass
        
    print("\n自然语言 UI 代理已退出。", file=sys.stderr)
    print("Natural Language UI Agent has exited.", file=sys.stderr)


if __name__ == "__main__":
    main()
