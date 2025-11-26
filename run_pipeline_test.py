#!/usr/bin/env python
"""
Quick script to run the full pipeline test.

Usage:
    # Make sure HF_TOKEN is set in .env file, then:
    python run_pipeline_test.py

    # Or run just preprocessing (no token needed)
    python run_pipeline_test.py --preprocess-only
"""

import argparse
import os
import sys
from pathlib import Path


def safe_print(text):
    """Print text safely handling Unicode errors on Windows."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback for Windows console without UTF-8 support
        print(text.encode('ascii', 'replace').decode('ascii'))


def check_env_file():
    """Check if .env file exists and has HF_TOKEN."""
    env_file = Path(".env")
    if not env_file.exists():
        print("ERROR: .env file not found!")
        print()
        print("Please create .env file in project root.")
        print("You can copy .env.example:")
        print()
        print("  cp .env.example .env")
        print()
        print("Then edit .env and set HF_TOKEN=your_token_here")
        print()
        return False

    # Check if HF_TOKEN is set in .env
    env_content = env_file.read_text(encoding='utf-8')
    has_token = False
    for line in env_content.split('\n'):
        if line.strip().startswith('#'):
            continue
        if 'HF_TOKEN=' in line and 'your_' not in line:
            has_token = True
            break

    if not has_token:
        print("ERROR: HF_TOKEN not set in .env file!")
        print()
        print("Please edit .env and set your HuggingFace token:")
        print()
        print("  HF_TOKEN=hf_xxxxxxxxxxxxx")
        print()
        print("Get your token at: https://huggingface.co/settings/tokens")
        print()
        return False

    print("[OK] .env file found with HF_TOKEN")
    return True


def check_test_file():
    """Check if test audio file exists."""
    test_file = Path("tests/chunk_001.mp3")
    if not test_file.exists():
        print(f"ERROR: Test audio file not found: {test_file}")
        print()
        print("Please ensure chunk_001.mp3 exists in the tests/ directory.")
        print()
        return False

    file_size = test_file.stat().st_size / (1024 * 1024)  # MB
    print(f"[OK] Test audio file found: {test_file} ({file_size:.2f} MB)")
    return True


def run_test(preprocess_only=False):
    """Run the pipeline test."""
    import subprocess

    if preprocess_only:
        test_name = "test_pipeline_without_models_validation_only"
        print("\n" + "=" * 70)
        print("Running PREPROCESSING TEST (no models needed)")
        print("=" * 70 + "\n")
    else:
        if not check_env_file():
            sys.exit(1)

        test_name = "test_full_pipeline_with_real_models"
        print("\n" + "=" * 70)
        print("Running FULL PIPELINE TEST (with real models from .env)")
        print("=" * 70 + "\n")
        print("NOTE: First run will download models (~2-3GB)")
        print("      This may take several minutes...")
        print()

    if not check_test_file():
        sys.exit(1)

    # Run pytest
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"tests/test_full_pipeline.py::TestFullPipeline::{test_name}",
        "-v",
        "-s",
        "--tb=short",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the speech-bot full pipeline test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline test (reads HF_TOKEN from .env)
  python run_pipeline_test.py

  # Run only preprocessing test (no token needed)
  python run_pipeline_test.py --preprocess-only

Before running, make sure .env file has:
  HF_TOKEN=hf_xxxxxxxxxxxxx
        """
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Run only preprocessing test (no models needed)"
    )

    args = parser.parse_args()

    print("Speech-Bot Pipeline Test Runner")
    print()

    run_test(preprocess_only=args.preprocess_only)


if __name__ == "__main__":
    main()
