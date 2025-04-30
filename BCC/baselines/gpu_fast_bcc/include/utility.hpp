/******************************************************************************
* Functionality: CPU related Utility manager
 ******************************************************************************/

#ifndef UTILITY_H
#define UTILITY_H

//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <filesystem>

template <typename T>
bool verify(const std::vector<T>& arr_1, const std::vector<T>& arr_2) {
    if(arr_1.size() != arr_2.size()) {
        std::cout <<"arr_1 size = " << arr_1.size() << std::endl;
        std::cout <<"arr_2 size = " << arr_2.size() << std::endl;
        std::cerr <<"Error in size.";
        return false;
    }
    for (size_t i = 0; i < arr_1.size(); ++i) {
        if (arr_1[i] != arr_2[i]) {
            std::cout << "Mismatch at index " << i << ":\n";
            std::cout << "arr_1[" << i << "] = " << arr_1[i] << "\n";
            std::cout << "arr_2[" << i << "] = " << arr_2[i] << "\n";
            return false;
        }
    }

    return true;
}

inline std::string get_file_extension(std::string filename) {

    std::filesystem::path file_path(filename);

    // Extracting filename with extension
    filename = file_path.filename().string();
    // std::cout << "Filename with extension: " << filename << std::endl;

    // Extracting filename without extension
    std::string filename_without_extension = file_path.stem().string();
    // std::cout << "Filename without extension: " << filename_without_extension << std::endl;

    return filename_without_extension;
}

inline void write_CSR(const std::vector<long>& vertices, const std::vector<int>& edges, std::string filename) {
    
    filename = get_file_extension(filename);
    std::string output_file = filename + "_csr.log";
    std::ofstream outFile(output_file);

        if(!outFile) {
            std::cerr <<"Unable to create file for writing.\n";
            return;
        }
    int numVertices = vertices.size() - 1;
    for (int i = 0; i < numVertices; ++i) {
        outFile << "Vertex " << i << " is connected to: ";
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            outFile << edges[j] << " ";
        }
        outFile << "\n";
    }
}

inline void print_CSR(const std::vector<long>& vertices, const std::vector<int>& edges) {
    int numVertices = vertices.size() - 1;
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Vertex " << i << " is connected to: ";
        for (int j = vertices[i]; j < vertices[i + 1]; ++j) {
            std::cout << edges[j] << " ";
        }
        std::cout << "\n";
    }
}

template <typename T>
void print(const std::vector<T>& arr) {
    for(const auto &i : arr) 
        std::cout << i <<" ";
    std::cout << std::endl;
}

template <typename T>
void print(const std::vector<T>& arr, const std::string& str) {
    std::cout << "\n" << str <<" starts" << "\n";
    int j = 0;
    for(const auto &i : arr) 
        std::cout << j++ <<"\t" << i << "\n";
    std::cout <<str <<" ends" << std::endl;
}

template <typename T>
void print(const T* arr, size_t size, const std::string& str) {
    std::cout << "\n" << str << " starts" << "\n";
    for(size_t i = 0; i < size; ++i) {
        std::cout << i << "\t" << arr[i] << "\n";
    }
    std::cout << str << " ends" << std::endl;
}

// Function to print vectors
template <typename T>
void print_vector(const std::vector<T>& vec, const std::string& name) {
    std::cout << name << ": ";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i] << (i < vec.size() - 1 ? ", " : "\n");
    }
}

// std::cout << "Reading input: " << formatDuration(194826) << std::endl;
// std::cout << "Allocation of device memory: " << formatDuration(3185) << std::endl;
inline std::string formatDuration(double timeInMs) {
    std::ostringstream stream;

    if (timeInMs < 1000) {
        // Less than 1 second, show in milliseconds
        stream << timeInMs << " ms";
    } else if (timeInMs < 60000) {
        // Less than 1 minute, show in seconds
        stream << std::fixed << std::setprecision(3) << (timeInMs / 1000.0) << " sec";
    } else {
        // Show in minutes and seconds
        long minutes = static_cast<long>(timeInMs / 60000);
        double seconds = static_cast<long>(timeInMs) % 60000 / 1000.0;
        stream << minutes << " min " << std::fixed << std::setprecision(3) << seconds << " sec";
    }

    return stream.str();
}

#endif // UTILITY_H