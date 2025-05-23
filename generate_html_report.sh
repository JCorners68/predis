#!/bin/bash

# Compile and run the HTML report generator
echo "Compiling HTML report generator..."
cd build
cmake .. 
make -j$(nproc)

echo "Generating HTML report from benchmark_results_1747901736.json..."
./bin/generate_professional_report ../benchmark_results/benchmark_results_1747901736.json ../doc/results/epic3_performance_report.html

echo "HTML report generated at doc/results/epic3_performance_report.html"