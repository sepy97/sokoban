//
//  main.cpp
//  Sokoban
//
//  Created by Semen Pyankov on 30.10.2020.
//

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

// Helper function to index 2-dimensional arrays
// Fortran order -> Column first
constexpr int index(int dim0, int dim1, const int i0, const int i1) {
    return i1 * dim0 + i0;
};

enum square
{
    ACQUIRED = 5,
    WALL = 4,
    TARGET = 3,
    PLAYER = 2,
    BOX = 1,
    FREE = 0
};

typedef struct pos
{
    int x;
    int y;
} pos;

typedef struct sokoban
{
    int numOfBoxes;
    pos player;
    pos dim;
    square* map;
    std::vector <pos> boxes;
    std::vector <pos> targets;
   // int numTargets;
} sokoban;

void dump (sokoban game)
{
    printf ("Sokoban: \nPlayer: %d %d\nDimensions of the map: %d %d\nMap: \n", \
            game.player.y, game.player.x, game.dim.y, game.dim.x);
    for (int i = 0; i < game.dim.y; i++)
    {
        for (int j = 0; j < game.dim.x; j++)
        {
                const auto board_idx = index(game.dim.x, game.dim.y, j, i);
 //               printf ("%d ", game.map[board_idx]);
            switch (game.map[board_idx])
            {
                case FREE:
                    printf (" ");
                    break;
                case BOX:
                    printf ("$");
                    break;
                case PLAYER:
                    printf ("@");
                    break;
                case TARGET:
                    printf (".");
                    break;
                case WALL:
                    printf ("#");
                    break;
                case ACQUIRED:
                    printf ("*");
                    break;
                default:
                    printf ("!!!!!");
                    break;
            }
        }
        printf ("\n");
    }
}

sokoban scan (std::string arg)
{
    std::ifstream myfile (arg.c_str());
    
    sokoban result;
    int sizeH = 0, sizeV = 0;
    
    if (myfile.is_open())
    {
        myfile >> sizeH;
        myfile >> sizeV;
    }
    
    result.map = new square[sizeH * sizeV];
    
    for (int y = 0; y < sizeV; y++)
    {
        for (int x = 0; x < sizeH; x++)
        {
            const int idx = index (sizeH, sizeV, x, y);
            result.map[idx] = FREE;
        }
    }

    result.dim.x = sizeH;
    result.dim.y = sizeV;

    int numOfWalls = 0;
    myfile >> numOfWalls;
    
    for (int i = 0; i < numOfWalls; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
        const int wall_idx = index (sizeH, sizeV, x - 1, y - 1);
        result.map[wall_idx] = WALL;
    }

    myfile >> result.numOfBoxes;

    for (int i = 0; i < result.numOfBoxes; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
        pos tmp;
        tmp.y = y-1;
        tmp.x = x-1;

        const int box_idx = index (sizeH, sizeV, x - 1, y - 1);
        result.boxes.push_back (tmp);
        result.map[box_idx] = BOX;
    }

    int numOfTargets = 0;
    myfile >> numOfTargets;
   // result.numTargets = numOfTargets;
    for (int i = 0; i < numOfTargets; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
        pos tmp;
        tmp.y = y-1;
        tmp.x = x-1;
        result.targets.push_back (tmp);

        const int target_idx = index (sizeH, sizeV, x - 1, y - 1);
    //    if (result.map[target_idx] == BOX) result.numTargets--;
        result.map[target_idx] = TARGET;
    }
        
    int player_x = 0, player_y = 0;
    myfile >> player_y >> player_x;
    result.player.x = player_x-1;
    result.player.y = player_y-1;

    const int player_idx = index (sizeH, sizeV, result.player.x, result.player.y);
    result.map[player_idx] = PLAYER;

    return result;
}

bool isValid (sokoban current, pos move)
{
    if (move.y >= 0 and move.x < current.dim.x and move.y >= 0 and move.y < current.dim.y)
    {
        return (current.map[index (current.dim.x, current.dim.y, move.x, move.y)] != WALL); 
    }
    else
    {
        return false;
    }
}

bool hasBox (sokoban current, pos box)
{
    return (current.map[index (current.dim.x, current.dim.y, box.x, box.y)] == BOX);// || current.map[index (current.dim.x, current.dim.y, box.x, box.y) == ACQUIRED);
}

bool makeMove (const sokoban* current, char move, sokoban* output)
{
    sokoban result = *current;
    pos toMove;
    switch (move)
    {
        case 'U':
            toMove.x = 0;
            toMove.y = -1;
            break;
        case 'D':
            toMove.x = 0;
            toMove.y = 1;
            break;
        case 'L':
            toMove.x = -1;
            toMove.y = 0;
            break;
        case 'R':
            toMove.x = 1;
            toMove.y = 0;
            break;
        default:
            printf ("Error on makeMove function!");
            break;
    }
    
    pos madeMove;
    madeMove.x = result.player.x + toMove.x;
    madeMove.y = result.player.y + toMove.y;
	
    if (isValid (*current, madeMove))
    {
        if (hasBox (*current, madeMove))
        {
            pos boxMove;
            boxMove.x = madeMove.x + toMove.x;
            boxMove.y = madeMove.y + toMove.y;
            if (isValid (*current, boxMove) && !hasBox (*current, boxMove))
            {
                for (int i = 0; i < result.numOfBoxes; i++)
                {
                    auto it = result.boxes[i];
                    if (it.x == madeMove.x && it.y == madeMove.y)
                    {
                        if (result.map [index (result.dim.x, result.dim.y, madeMove.x, madeMove.y)] == BOX)
                        {
                            result.boxes[i] = boxMove;
                            result.map [index (result.dim.x, result.dim.y, result.player.x, result.player.y)] = FREE;
                            result.map [index (result.dim.x, result.dim.y, madeMove.x, madeMove.y)] = PLAYER;
                            result.map [index (result.dim.x, result.dim.y, boxMove.x, boxMove.y)] = BOX;
                
                            pos plyr, mdmv, bxmv;
                            plyr.x = result.player.x;
                            plyr.y = result.player.y;
                            mdmv.x = madeMove.x;
                            mdmv.y = madeMove.y;
                            bxmv.x = boxMove.x;
                            bxmv.y = boxMove.y;

                            /*
                            if (std::find (result.targets.begin(), result.targets.end(), plyr) != result.targets.end ())
                            {
                                result.map [index (result.dim.x, result.dim.y, result.player.x, result.player.y)] = TARGET;
                            }
                            
                            if (std::find (result.targets.begin(), result.targets.end(), mdmv) != result.targets.end ())
                            {
                                result.numTargets++;
                            }

                            if (std::find (result.targets.begin(), result.targets.end(), bxmv) != result.targets.end ())
                            {
                                result.numTargets--;
                            }
                            */

                        }
                                
                        result.player.x = madeMove.x;
                        result.player.y = madeMove.y;
                    }
                }
            }
        }
        else
        {
            result.map [index (result.dim.x, result.dim.y, result.player.x, result.player.y)] = FREE;
            result.map [index (result.dim.x, result.dim.y, madeMove.x, madeMove.y)] = PLAYER;
            pos plyr;
            plyr.x = result.player.x;
            plyr.y = result.player.y;
                
            /*if (std::find (result.targets.begin(), result.targets.end(), plyr) != result.targets.end ())
            {
                result.map [index (result.dim.x, result.dim.y, result.player.x, result.player.y)] = TARGET;
            }*/
		    

            result.player.x = madeMove.x;
            result.player.y = madeMove.y;
        }
    }
	*output = result;

    for (int i = 0; i < result.targets.size(); i++)
    {
        if (result.map [index (result.dim.x, result.dim.y, result.targets[i].x, result.targets[i].y)] != BOX) return false;
    }
    return true;
    //return (result.numTargets == 0);
    //return false;
}

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
