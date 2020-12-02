//
//  main.cpp
//  Sokoban
//
//  Created by Semen Pyankov on 30.10.2020.
//

#include <iostream>

#include "./sokoban/environment/sokoban.h"

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

    bool iscornered = isDeadlocked(body);

    std::cout << "Is box in a corner? " << iscornered << std::endl;

    // std::vector <sokoban> tmp = expand (output);
    auto exp = new_state_vector(); // = (sokoban**) calloc (4, sizeof (sokoban*));
    expand (body, exp);

    std::cout << "Expanding moves: " << std:: endl;
    for (int i = 0; i < 4; i++) {
	    dump (*exp[i]);
    }

    delete body;
    delete output;
    for (auto& state : exp) {
        delete state;
    }
}
