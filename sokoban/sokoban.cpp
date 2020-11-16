#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "sokoban.h"


// Helper function to index 2-dimensional arrays
// Fortran order -> Column first
constexpr int index(const int dim0, const int dim1, const int i0, const int i1) {
    return i1 * dim0 + i0;
};

// Dummy function that is necessary to dynamically create the sokoban struct from python.
sokoban* blank_state() {
    return new sokoban;
}

// Dummy function that is necessary to clean up struct memory from python.
void delete_state(sokoban* state) {
    delete state;
}

void dump (const sokoban& game)
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

sokoban* scan (const std::string& arg)
{
    std::ifstream myfile (arg.c_str());
    
    sokoban* result = new sokoban;
    int sizeH = 0, sizeV = 0;
    
    if (myfile.is_open())
    {
        myfile >> sizeH;
        myfile >> sizeV;
    }
    
    // result->map = new square[sizeH * sizeV];
    
    for (int y = 0; y < sizeV; y++)
    {
        for (int x = 0; x < sizeH; x++)
        {
            result->map.push_back(FREE);
            // const int idx = index (sizeH, sizeV, x, y);
            // result->map[idx] = FREE;
        }
    }

    result->dim.x = sizeH;
    result->dim.y = sizeV;

    int numOfWalls = 0;
    myfile >> numOfWalls;
    
    for (int i = 0; i < numOfWalls; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
        const int wall_idx = index (sizeH, sizeV, x - 1, y - 1);
        result->map[wall_idx] = WALL;
    }

    myfile >> result->numOfBoxes;

    for (int i = 0; i < result->numOfBoxes; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
        pos tmp;
        tmp.y = y-1;
        tmp.x = x-1;

        const int box_idx = index (sizeH, sizeV, x - 1, y - 1);
        result->boxes.push_back (tmp);
        result->map[box_idx] = BOX;
    }

    int numOfTargets = 0;
    myfile >> numOfTargets;
   // output->numTargets = numOfTargets;
    for (int i = 0; i < numOfTargets; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
        pos tmp;
        tmp.y = y-1;
        tmp.x = x-1;
        result->targets.push_back (tmp);

        const int target_idx = index (sizeH, sizeV, x - 1, y - 1);
    //    if (output->map[target_idx] == BOX) output->numTargets--;
        result->map[target_idx] = TARGET;
    }
        
    int player_x = 0, player_y = 0;
    myfile >> player_y >> player_x;
    result->player.x = player_x-1;
    result->player.y = player_y-1;

    const int player_idx = index (sizeH, sizeV, result->player.x, result->player.y);
    result->map[player_idx] = PLAYER;

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

bool makeMove (const sokoban* const current, char move, sokoban* output)
{
    if (output == nullptr) {
        output = blank_state();
    }

    *output = *current;

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
    madeMove.x = output->player.x + toMove.x;
    madeMove.y = output->player.y + toMove.y;
	
    if (isValid (*current, madeMove))
    {
        if (hasBox (*current, madeMove))
        {
            pos boxMove;
            boxMove.x = madeMove.x + toMove.x;
            boxMove.y = madeMove.y + toMove.y;
            if (isValid (*current, boxMove) && !hasBox (*current, boxMove))
            {
                for (int i = 0; i < output->numOfBoxes; i++)
                {
                    auto it = output->boxes[i];
                    if (it.x == madeMove.x && it.y == madeMove.y)
                    {
                        if (output->map [index (output->dim.x, output->dim.y, madeMove.x, madeMove.y)] == BOX)
                        {
                            output->boxes[i] = boxMove;
                            output->map [index (output->dim.x, output->dim.y, output->player.x, output->player.y)] = FREE;
                            output->map [index (output->dim.x, output->dim.y, madeMove.x, madeMove.y)] = PLAYER;
                            output->map [index (output->dim.x, output->dim.y, boxMove.x, boxMove.y)] = BOX;
                
                            pos plyr, mdmv, bxmv;
                            plyr.x = output->player.x;
                            plyr.y = output->player.y;
                            mdmv.x = madeMove.x;
                            mdmv.y = madeMove.y;
                            bxmv.x = boxMove.x;
                            bxmv.y = boxMove.y;

                            /*
                            if (std::find (output->targets.begin(), output->targets.end(), plyr) != output->targets.end ())
                            {
                                output->map [index (output->dim.x, output->dim.y, output->player.x, output->player.y)] = TARGET;
                            }
                            
                            if (std::find (output->targets.begin(), output->targets.end(), mdmv) != output->targets.end ())
                            {
                                output->numTargets++;
                            }

                            if (std::find (output->targets.begin(), output->targets.end(), bxmv) != output->targets.end ())
                            {
                                output->numTargets--;
                            }
                            */

                        }
                                
                        output->player.x = madeMove.x;
                        output->player.y = madeMove.y;
                    }
                }
            }
        }
        else
        {
            output->map [index (output->dim.x, output->dim.y, output->player.x, output->player.y)] = FREE;
            output->map [index (output->dim.x, output->dim.y, madeMove.x, madeMove.y)] = PLAYER;
            pos plyr;
            plyr.x = output->player.x;
            plyr.y = output->player.y;
                
            /*if (std::find (output->targets.begin(), output->targets.end(), plyr) != output->targets.end ())
            {
                output->map [index (output->dim.x, output->dim.y, output->player.x, output->player.y)] = TARGET;
            }*/
		    

            output->player.x = madeMove.x;
            output->player.y = madeMove.y;
        }
    }

	// *output = result;

    for (int i = 0; i < output->targets.size(); i++)
    {
        if (output->map [index (output->dim.x, output->dim.y, output->targets[i].x, output->targets[i].y)] != BOX) return false;
    }
    return true;
    //return (output->numTargets == 0);
    //return false;
}