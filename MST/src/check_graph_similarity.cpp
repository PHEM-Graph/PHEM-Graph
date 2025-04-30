#include <fstream>
#include <iostream>
#include <string>
#include <vector>



int main(int argc, char* argv[]){
    std::string file1 = argv[1];
    //print the contents of file1
    std::ifstream file(file1);
    int num_nodes = atoi(argv[3]);
    std::string str;



    int weight1 =0;
    int weight2 =0;
    std::vector<int> visited1(num_nodes, 0);
    int u, v, w;
    while (std::getline(file, str))
    {
        //split the string into u, v, w
        sscanf(str.c_str(), "%d %d %d", &u, &v, &w);
        visited1[u-1] = 1;
        visited1[v-1] = 1;
        weight1 += w;
    }
    file.close();
    std::ifstream file2(argv[2]);
    std::string str2;
    std::vector<int> visited2(num_nodes, 0);
    while (std::getline(file2, str2))
    {
        //split the string into u, v, w
        sscanf(str2.c_str(), "%d %d %d", &u, &v, &w);
        visited2[u-1] = 1;
        visited2[v-1] = 1;
        weight2 += w;
    }

    std::cout << "weight 1: " << weight1 << std::endl;
    std::cout << "weight 2: " << weight2 << std::endl;

    if(weight1 != weight2){
        std::cout << "weights are not equal" << std::endl;
        return 1;
    }

    //check if visited1[i] == visted2[i] for all i
    for(int i = 0; i < num_nodes; i++){
        if(visited1[i] != visited2[i]){
            std::cout << "visited arrays are not equal at " << i << std::endl;
            return 1;
        }
    }
    std::cout << "graphs are equal" << std::endl;


    return 0;

}