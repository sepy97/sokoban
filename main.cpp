//
//  main.cpp
//  Sokoban
//
//  Created by Semen Pyankov on 30.10.2020.
//

#include <iostream>

#include "sokoban.h"

int main(int argc, const char * argv[])
{
    std::string input = "sokoban00.txt";
    if (argc >= 2)
    {
        input = std::string (argv[1]);
    }
    
    sokoban body = scan (input);
    
    dump (body);
    
    bool finished = makeMove (&body, 'D', &body);
    
    dump (body);
    
    std::cout << "Game is finished? " << finished << std::endl;
}
