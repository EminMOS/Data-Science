#!/usr/bin/env python3
"""
Benchmark script to compare execution times with different sample sizes
"""

import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt

def run_with_sample_size(sample_size):
    """Run the analysis with a specific sample size and measure time"""
    
    cmd = [
        "python", 
        "C:\Users\emina\OneDrive\Desktop\Uni\Semester3\Data-Science\Final_Project\Other\twitter_sentiment_main.py",
        "--data_path", "C:\Users\emina\OneDrive\Desktop\Uni\Semester3\Data-Science\Final_Project\Other\Twitter_Data.csv",
        "--sample-size", str(sample_size)
    ]
    
    print(f"\n🔄 Testing with {sample_size} samples...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract accuracy from output
        accuracy = None
        for line in result.stdout.split('\n'):
            if "BEST PERFORMING MODEL" in line:
                for next_line in result.stdout.split('\n')[result.stdout.split('\n').index(line):]:
                    if "Accuracy:" in next_line:
                        try:
                            accuracy = float(next_line.split(':')[1].strip())
                        except:
                            pass
                        break
                break
        
        print(f"   ✅ Completed in {elapsed_time:.2f} seconds")
        return elapsed_time, accuracy
        
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout after 180 seconds")
        return None, None
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None, None

def main():
    print("\n" + "="*80)
    print("TWITTER SENTIMENT ANALYSIS - PERFORMANCE BENCHMARK")
    print("="*80)
    print("\nThis benchmark tests different sample sizes to show the trade-off")
    print("between execution time and model accuracy.")
    
    # Test different sample sizes
    sample_sizes = [100, 250, 500, 1000, 2500, 5000]
    results = []
    
    print("\n" + "-"*80)
    print("RUNNING BENCHMARKS")
    print("-"*80)
    
    for size in sample_sizes:
        exec_time, accuracy = run_with_sample_size(size)
        if exec_time:
            results.append({
                'Sample Size': size,
                'Execution Time (seconds)': exec_time,
                'Accuracy': accuracy if accuracy else 'N/A'
            })
    
    # Display results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    if results:
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        
        # Performance recommendations
        print("\n" + "-"*80)
        print("RECOMMENDATIONS")
        print("-"*80)
        
        print("\n📊 Sample Size Guidelines:\n")
        print("• 100 samples:   Ultra-fast testing & debugging (~30 sec)")
        print("• 500 samples:   Quick feature development (~1-2 min)")
        print("• 1000 samples:  Development mode, good balance (~2-3 min)")
        print("• 2500 samples:  Better accuracy testing (~4-5 min)")
        print("• 5000 samples:  Near-production testing (~5-7 min)")
        print("• Full dataset:  Production training (~15-30 min)")
        
        print("\n💡 Tips:")
        print("• Use --dev flag for automatic 1000 samples with simplified parameters")
        print("• Start with 100 samples when debugging code changes")
        print("• Use 1000-2500 samples for model experimentation")
        print("• Run full dataset only for final model training")
        
        # Create visualization if matplotlib is available
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(df['Sample Size'], df['Execution Time (seconds)'], 
                   marker='o', linewidth=2, markersize=8, color='#3498db')
            ax.set_xlabel('Sample Size', fontsize=12)
            ax.set_ylabel('Execution Time (seconds)', fontsize=12)
            ax.set_title('Execution Time vs Sample Size', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add annotations
            for i, row in df.iterrows():
                ax.annotate(f"{row['Execution Time (seconds)']:.1f}s", 
                          (row['Sample Size'], row['Execution Time (seconds)']),
                          textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
            print("\n📈 Visualization saved as 'benchmark_results.png'")
        except:
            pass
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED")
    print("="*80)

if __name__ == "__main__":
    # Note: This benchmark may take 10-15 minutes to complete all tests
    print("\n⚠️  Note: This benchmark will take approximately 10-15 minutes to complete.")
    response = input("Do you want to continue? (y/n): ")
    
    if response.lower() == 'y':
        main()
    else:
        # Run a quick demo instead
        print("\nRunning quick demo with 100 samples instead...")
        exec_time, accuracy = run_with_sample_size(100)
        if exec_time:
            print(f"\n✅ Demo completed in {exec_time:.2f} seconds")
            print("\nTo run the full benchmark later, execute:")
            print("   python benchmark_sample_sizes.py")
