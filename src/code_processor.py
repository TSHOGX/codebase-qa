import os
import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set to DEBUG level for more verbose output
logger.setLevel(logging.DEBUG)

class CodeBlock(BaseModel):
    """Representation of a code block with metadata."""
    code: str = Field(..., description="The actual code content")
    file_path: str = Field(..., description="Path to the source file")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    language: str = Field(..., description="Programming language")
    block_type: str = Field(..., description="Type of block (function, class, etc.)")
    comments: str = Field(default="", description="Associated comments")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.code,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "block_type": self.block_type,
            "comments": self.comments,
            "metadata": {
                "file_path": self.file_path,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "language": self.language,
                "block_type": self.block_type
            }
        }

class CodeProcessor:
    """Process code files and extract meaningful code blocks."""
    
    def __init__(self):
        # File extensions to language mapping
        self.language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".rs": "rust",
            ".html": "html",
            ".css": "css"
        }
        
        # Block pattern recognition
        self.patterns = {
            "python": {
                "function": r"def\s+(\w+)\s*\(.*?\):\s*(?:\s*#.*?(?:\n|$))?",
                "class": r"class\s+(\w+)(?:\(.*?\))?:\s*(?:\s*#.*?(?:\n|$))?",
                "docstring": r'"""(?:.|\n)*?"""',
                "comment": r"#.*"
            },
            "javascript": {
                "function": r"function\s+(\w+)\s*\(.*?\)\s*{",
                "method": r"(\w+)\s*\(.*?\)\s*{",
                "class": r"class\s+(\w+)(?:\s+extends\s+\w+)?\s*{",
                "arrow_function": r"(?:const|let|var)?\s*(\w+)\s*=\s*(?:\(.*?\)|(?:\w+))\s*=>\s*(?:{|(?:.|\n)*?)",
                "comment": r"\/\/.*",
                "multiline_comment": r"\/\*(?:.|\n)*?\*\/"
            }
        }
        
    def detect_language(self, file_path: str) -> str:
        """Detect programming language based on file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return self.language_map.get(ext, "unknown")
        
    def process_file(self, file_path: str) -> List[CodeBlock]:
        """Process a single code file and extract code blocks."""
        logger.info(f"Processing file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return []
            
        language = self.detect_language(file_path)
        if language == "unknown":
            logger.warning(f"Unsupported file type: {file_path}")
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            logger.debug(f"File content length: {len(content)} characters")
            blocks = self._split_into_blocks(content, file_path, language)
            logger.info(f"Extracted {len(blocks)} code blocks from {file_path}")
            return blocks
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
    
    def process_directory(self, directory_path: str) -> List[CodeBlock]:
        """Process all code files in a directory recursively."""
        logger.info(f"Processing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return []
            
        all_blocks = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext in self.language_map:
                    blocks = self.process_file(file_path)
                    all_blocks.extend(blocks)
        
        logger.info(f"Extracted a total of {len(all_blocks)} code blocks from {directory_path}")
        return all_blocks
    
    def _split_into_blocks(self, content: str, file_path: str, language: str) -> List[CodeBlock]:
        """Split code content into meaningful blocks."""
        blocks = []
        
        # For simplicity, we'll split by logical structures (functions, classes)
        # and accumulate comments as we go
        
        if language == "python":
            return self._split_python_code(content, file_path)
        elif language in ["javascript", "typescript"]:
            return self._split_js_code(content, file_path, language)
        else:
            # Generic approach for other languages: split by functions if patterns exist
            if language in self.patterns:
                patterns = self.patterns[language]
                # Implementation for specific languages would go here
                pass
            
            # Fallback: just split by logical code sections with empty lines
            return self._split_generic_code(content, file_path, language)
    
    def _split_python_code(self, content: str, file_path: str) -> List[CodeBlock]:
        """Split Python code into blocks."""
        blocks = []
        lines = content.split('\n')
        
        logger.debug(f"Splitting Python file {file_path} with {len(lines)} lines")
        
        # Regular expressions for Python constructs
        class_pattern = re.compile(r'^\s*class\s+(\w+)')
        func_pattern = re.compile(r'^\s*def\s+(\w+)')
        docstring_start = re.compile(r'^\s*"""')
        docstring_end = re.compile(r'"""$')
        comment_pattern = re.compile(r'^\s*#')
        
        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
                
            start_line = i + 1  # 1-indexed line numbers
            
            # Check for class definition
            class_match = class_pattern.match(lines[i])
            if class_match:
                logger.debug(f"Found class at line {start_line}: {lines[i].strip()}")
                block_start = i
                block_type = "class"
                class_name = class_match.group(1)
                indent = len(lines[i]) - len(lines[i].lstrip())
                
                # Get the docstring if it exists
                comments = []
                j = i + 1
                in_docstring = False
                
                if j < len(lines) and docstring_start.match(lines[j].strip()):
                    in_docstring = True
                    comments.append(lines[j])
                    j += 1
                    
                    while j < len(lines) and (not docstring_end.search(lines[j]) or in_docstring):
                        comments.append(lines[j])
                        if docstring_end.search(lines[j]):
                            in_docstring = False
                        j += 1
                        
                    if j < len(lines) and docstring_end.search(lines[j]):
                        comments.append(lines[j])
                        j += 1
                
                # Find the end of the class (by indentation)
                while j < len(lines):
                    if lines[j].strip() and not lines[j].startswith(' ' * (indent + 1)):
                        break
                    j += 1
                
                end_line = j  # End line is exclusive
                class_code = '\n'.join(lines[block_start:j])
                
                blocks.append(CodeBlock(
                    code=class_code,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language="python",
                    block_type=f"class:{class_name}",
                    comments='\n'.join(comments)
                ))
                
                i = j
                continue
                
            # Check for function definition
            func_match = func_pattern.match(lines[i])
            if func_match:
                logger.debug(f"Found function at line {start_line}: {lines[i].strip()}")
                block_start = i
                block_type = "function"
                func_name = func_match.group(1)
                indent = len(lines[i]) - len(lines[i].lstrip())
                
                # Get the docstring if it exists
                comments = []
                j = i + 1
                in_docstring = False
                
                if j < len(lines) and docstring_start.match(lines[j].strip()):
                    in_docstring = True
                    comments.append(lines[j])
                    j += 1
                    
                    while j < len(lines) and not docstring_end.search(lines[j]):
                        comments.append(lines[j])
                        j += 1
                        
                    if j < len(lines) and docstring_end.search(lines[j]):
                        comments.append(lines[j])
                        j += 1
                
                # Find the end of the function (by indentation)
                while j < len(lines):
                    if lines[j].strip() and not lines[j].startswith(' ' * (indent + 1)):
                        break
                    j += 1
                
                end_line = j  # End line is exclusive
                func_code = '\n'.join(lines[block_start:j])
                
                blocks.append(CodeBlock(
                    code=func_code,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language="python",
                    block_type=f"function:{func_name}",
                    comments='\n'.join(comments)
                ))
                
                i = j
                continue
                
            # Check for standalone comment blocks
            if comment_pattern.match(lines[i]):
                block_start = i
                comments = [lines[i]]
                j = i + 1
                
                while j < len(lines) and (comment_pattern.match(lines[j]) or not lines[j].strip()):
                    if lines[j].strip():
                        comments.append(lines[j])
                    j += 1
                
                end_line = j
                
                if len(comments) > 1:  # Only include multi-line comment blocks
                    blocks.append(CodeBlock(
                        code='\n'.join(comments),
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        language="python",
                        block_type="comment_block",
                        comments='\n'.join(comments)
                    ))
                
                i = j
                continue
            
            # If we reach here, this line is part of a code block that's not a function or class
            # For simplicity, we'll group these as "other" blocks
            block_start = i
            j = i + 1
            
            while j < len(lines) and not (class_pattern.match(lines[j]) or func_pattern.match(lines[j])):
                j += 1
                
            end_line = j
            code_block = '\n'.join(lines[block_start:j])
            
            if code_block.strip():  # Only include non-empty blocks
                blocks.append(CodeBlock(
                    code=code_block,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language="python",
                    block_type="code_block",
                    comments=""
                ))
            
            i = j
            
        return blocks
    
    def _split_js_code(self, content: str, file_path: str, language: str) -> List[CodeBlock]:
        """Split JavaScript/TypeScript code into blocks."""
        blocks = []
        lines = content.split('\n')
        
        # Regular expressions for JS/TS constructs
        class_pattern = re.compile(r'^\s*class\s+(\w+)')
        func_pattern = re.compile(r'^\s*function\s+(\w+)')
        method_pattern = re.compile(r'^\s*(\w+)\s*\([^)]*\)\s*{')
        arrow_func_pattern = re.compile(r'^\s*(?:const|let|var)?\s*(\w+)\s*=\s*(?:\([^)]*\)|(?:\w+))\s*=>')
        comment_start = re.compile(r'^\s*\/\*')
        comment_end = re.compile(r'\*\/$')
        line_comment = re.compile(r'^\s*\/\/')
        
        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
                
            start_line = i + 1  # 1-indexed line numbers
            
            # Check for class definition
            class_match = class_pattern.match(lines[i])
            if class_match:
                block_start = i
                class_name = class_match.group(1)
                
                # Find the matching closing brace
                bracket_count = 0
                j = i
                while j < len(lines):
                    # Count opening braces
                    bracket_count += lines[j].count('{')
                    # Count closing braces
                    bracket_count -= lines[j].count('}')
                    
                    if bracket_count <= 0 and j > i:
                        j += 1
                        break
                    j += 1
                
                end_line = j
                class_code = '\n'.join(lines[block_start:j])
                
                blocks.append(CodeBlock(
                    code=class_code,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    block_type=f"class:{class_name}",
                    comments=""
                ))
                
                i = j
                continue
                
            # Check for function definition
            func_match = func_pattern.match(lines[i])
            if func_match:
                block_start = i
                func_name = func_match.group(1)
                
                # Find the matching closing brace
                bracket_count = 0
                j = i
                while j < len(lines):
                    # Count opening braces
                    bracket_count += lines[j].count('{')
                    # Count closing braces
                    bracket_count -= lines[j].count('}')
                    
                    if bracket_count <= 0 and j > i:
                        j += 1
                        break
                    j += 1
                
                end_line = j
                func_code = '\n'.join(lines[block_start:j])
                
                blocks.append(CodeBlock(
                    code=func_code,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    block_type=f"function:{func_name}",
                    comments=""
                ))
                
                i = j
                continue
                
            # Check for method or arrow function
            method_match = method_pattern.match(lines[i])
            arrow_match = arrow_func_pattern.match(lines[i])
            
            if method_match or arrow_match:
                block_start = i
                func_name = method_match.group(1) if method_match else arrow_match.group(1)
                block_type = "method" if method_match else "arrow_function"
                
                # Find the matching closing brace
                bracket_count = 0
                j = i
                while j < len(lines):
                    # Count opening braces
                    bracket_count += lines[j].count('{')
                    # Count closing braces
                    bracket_count -= lines[j].count('}')
                    
                    if bracket_count <= 0 and j > i:
                        j += 1
                        break
                    
                    # For arrow functions that don't use braces
                    if arrow_match and bracket_count == 0 and '=>' in lines[j]:
                        if not '{' in lines[j]:
                            j += 1
                            break
                    j += 1
                
                end_line = j
                func_code = '\n'.join(lines[block_start:j])
                
                blocks.append(CodeBlock(
                    code=func_code,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    block_type=f"{block_type}:{func_name}",
                    comments=""
                ))
                
                i = j
                continue
                
            # Check for comment blocks
            if comment_start.match(lines[i]) or line_comment.match(lines[i]):
                block_start = i
                is_multiline = comment_start.match(lines[i]) is not None
                comments = [lines[i]]
                j = i + 1
                
                if is_multiline:
                    # Find the end of the multiline comment
                    while j < len(lines) and not comment_end.search(lines[j]):
                        comments.append(lines[j])
                        j += 1
                    
                    if j < len(lines):
                        comments.append(lines[j])
                        j += 1
                else:
                    # Find consecutive single-line comments
                    while j < len(lines) and (line_comment.match(lines[j]) or not lines[j].strip()):
                        if lines[j].strip():
                            comments.append(lines[j])
                        j += 1
                
                end_line = j
                
                blocks.append(CodeBlock(
                    code='\n'.join(comments),
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    block_type="comment_block",
                    comments='\n'.join(comments)
                ))
                
                i = j
                continue
            
            # If we reach here, collect code that's not in a function or class
            block_start = i
            j = i + 1
            
            while j < len(lines) and not (class_pattern.match(lines[j]) or 
                                         func_pattern.match(lines[j]) or 
                                         method_pattern.match(lines[j]) or 
                                         arrow_func_pattern.match(lines[j]) or
                                         comment_start.match(lines[j]) or 
                                         line_comment.match(lines[j])):
                j += 1
                
            end_line = j
            code_block = '\n'.join(lines[block_start:j])
            
            if code_block.strip():  # Only include non-empty blocks
                blocks.append(CodeBlock(
                    code=code_block,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    block_type="code_block",
                    comments=""
                ))
            
            i = j
            
        return blocks
    
    def _split_generic_code(self, content: str, file_path: str, language: str) -> List[CodeBlock]:
        """Generic code splitting for languages without specific implementations."""
        blocks = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
                
            start_line = i + 1  # 1-indexed line numbers
            block_start = i
            
            # Find the next empty line sequence (2+ empty lines) or end of file
            j = i + 1
            empty_count = 0
            while j < len(lines):
                if not lines[j].strip():
                    empty_count += 1
                else:
                    empty_count = 0
                
                if empty_count >= 2:
                    j -= 1  # Don't include the empty lines
                    break
                
                j += 1
                
            end_line = j + 1
            code_block = '\n'.join(lines[block_start:j])
            
            if code_block.strip():  # Only include non-empty blocks
                blocks.append(CodeBlock(
                    code=code_block,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    block_type="code_block",
                    comments=""
                ))
            
            i = j + 1  # Skip past the empty lines
            
        return blocks 