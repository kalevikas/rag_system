"""
Utility functions and helpers for the RAG system
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import colorlog


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 use_colors: bool = True) -> logging.Logger:
    # ...existing code...
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    # ...existing code...
    if use_colors:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    # ...existing code...
    handlers = []
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        handlers.append(file_handler)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_env_file(env_path: str = ".env"):
    from dotenv import load_dotenv
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info(f"Environment variables loaded from {env_path}")
    else:
        logging.warning(f".env file not found at {env_path}")


def format_sources_for_display(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "No sources available"
    lines = ["Sources:"]
    for i, source in enumerate(sources, 1):
        file_name = source.get("file_name", "Unknown")
        page = source.get("page_number", "")
        score = source.get("score", 0.0)
        lines.append(f"\n[{i}] {file_name} (Page {page}) - Score: {score:.3f}")
        preview = source.get("preview", "")
        if preview:
            lines.append(f"    {preview[:150]}...")
    return "\n".join(lines)


def print_query_result(result: Dict[str, Any], show_sources: bool = True):
    print("\n" + "="*80)
    print(f"QUESTION: {result['question']}")
    print("="*80)
    print(f"\nANSWER:")
    print(result['answer'])
    print(f"\nConfidence: {result.get('confidence', 0.0):.2f}")
    print(f"Number of sources: {result.get('num_sources', 0)}")
    if show_sources and result.get('sources'):
        print(f"\n{format_sources_for_display(result['sources'])}")
    print("\n" + "="*80 + "\n")


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
