#!/usr/bin/env python3
"""
Test script to demonstrate the optimized Twitter Sentiment Analysis with small sample
"""

import subprocess
import time

def test_with_100_samples():
    """Test the script with 100 samples"""
    print("\n" + "="*80)
    print("TESTING TWITTER SENTIMENT ANALYSIS WITH 100 SAMPLES")
    print("="*80)
    print("\nThis test demonstrates the new sampling feature that allows:")
    print("• Fast testing with small datasets (100 samples)")
    print("• Development mode with simplified parameters")
    print("• Configurable sample sizes")
    print("\n" + "-"*80)
    
    # Run with 100 samples
    print("\n🚀 Running with --sample-size 100 for ultra-fast execution...")
    print("-"*80)
    
    start_time = time.time()
    
    cmd = [
        "python", 
        "/mnt/user-data/outputs/twitter_sentiment_main.py",
        "--data_path", "/mnt/user-data/uploads/Twitter_Data.csv",
        "--sample-size", "100"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for 100 samples
        )
        
        # Print output
        if result.stdout:
            # Show key parts of the output
            lines = result.stdout.split('\n')
            
            # Show configuration
            in_config = False
            for line in lines:
                if "CONFIGURATION" in line:
                    in_config = True
                if in_config:
                    print(line)
                    if "SAMPLE MODE" in line or "samples" in line:
                        in_config = False
                        break
            
            # Show sampling info
            print("\n" + "-"*80)
            print("SAMPLING OUTPUT:")
            print("-"*80)
            for i, line in enumerate(lines):
                if "DATASET SAMPLING" in line:
                    for j in range(i, min(i+15, len(lines))):
                        print(lines[j])
                    break
            
            # Show model results summary
            print("\n" + "-"*80)
            print("MODEL TRAINING RESULTS:")
            print("-"*80)
            for line in lines:
                if "Accuracy:" in line or "F1-Macro:" in line or "Best params:" in line:
                    print(line)
                if "BEST MODEL" in line:
                    break
                    
            # Show completion
            for i, line in enumerate(lines):
                if "PIPELINE COMPLETED" in line:
                    print("\n" + "-"*80)
                    for j in range(i, min(i+5, len(lines))):
                        print(lines[j])
                    break
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Execution time: {elapsed_time:.2f} seconds")
        
        if elapsed_time < 30:
            print("✅ EXCELLENT! Script completed in under 30 seconds with 100 samples")
        elif elapsed_time < 60:
            print("✅ GOOD! Script completed in under 1 minute with 100 samples")
        else:
            print("⚠️  Script took longer than expected for 100 samples")
            
    except subprocess.TimeoutExpired:
        print("❌ Script timed out after 60 seconds")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    print("\n💡 TIP: You can adjust the sample size with --sample-size parameter:")
    print("   python twitter_sentiment_main.py --sample-size 500")
    print("   python twitter_sentiment_main.py --sample-size 1000")
    print("   python twitter_sentiment_main.py --sample-size 5000")
    print("   python twitter_sentiment_main.py --dev  # Development mode with 1000 samples")
    print("   python twitter_sentiment_main.py       # Full dataset")

if __name__ == "__main__":
    test_with_100_samples()
