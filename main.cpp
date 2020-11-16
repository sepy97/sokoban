//
//  main.cpp
//  Sokoban
//
//  Created by Semen Pyankov on 30.10.2020.
//

#include <iostream>

#include "./sokoban/sokoban.h"

int main(int argc, const char * argv[])
{
    std::string input = "sokoban00.txt";
    if (argc >= 2)
    {
        input = std::string (argv[1]);
    }
    
    sokoban* body = scan (input);
    sokoban* output = new_state();
    
    dump (*body);
    
    bool finished = makeMove (body, 'D', output);
    
    dump (*output);
    dump (*body);
    
    std::cout << "Game is finished? " << finished << std::endl;

    delete body;
    delete output;
}
