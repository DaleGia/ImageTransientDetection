#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <chrono>
#include <iostream>

class Benchmark 
{
    public:
        Benchmark() : totalTime(0), numRuns(0) {}

        void start() 
        {
            startTime = std::chrono::high_resolution_clock::now();
        }

        double stop() 
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
            totalTime += duration.count();
            numRuns++;
            return duration.count() / 1e6;  // Convert microseconds to seconds for output
        }

        void print(std::string name) 
        {
            if (numRuns == 0) 
            {
                return;
            }
            double time = stop();
            double averageTime = totalTime / numRuns;

            std::cout << std::fixed << std::setprecision(6);
            std::cout << std::left << name << std::endl;
            std::cout << std::setw(25) << std::left << "last: ";
            std::cout << time << "s" << std::endl;
            std::cout << std::setw(25) << std::left << "average: ";
            std::cout << averageTime << "us" << std::endl;
            std::cout << std::setw(25) << std::left << "total: ";
            std::cout << totalTime << "us" << std::endl;  // Convert microseconds to seconds for output
            std::cout << std::setw(25) << std::left << "count: ";
            std::cout << numRuns << "s" << std::endl;
        }

    private:
        std::chrono::high_resolution_clock::time_point startTime;
        double totalTime;
        int numRuns;
};

#endif
