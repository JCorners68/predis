#!/bin/bash

# Compile and run the professional report generator

echo "Compiling professional report generator..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Compile the standalone report generator
g++ -std=c++17 \
    ../src/benchmarks/standalone_report_generator.cpp \
    ../src/benchmarks/professional_report_generator.cpp \
    -I../src \
    -o generate_report \
    -lpthread

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Generating professional report..."
    
    # Run the report generator
    ./generate_report
    
    if [ $? -eq 0 ]; then
        echo "Report generated successfully!"
        echo "Opening report in browser..."
        
        # Try to open the report in the default browser
        if command -v xdg-open > /dev/null; then
            xdg-open epic2_professional_report.html
        elif command -v open > /dev/null; then
            open epic2_professional_report.html
        else
            echo "Report saved as: build/epic2_professional_report.html"
            echo "Please open it manually in your browser."
        fi
    else
        echo "Error generating report"
        exit 1
    fi
else
    echo "Compilation failed"
    exit 1
fi