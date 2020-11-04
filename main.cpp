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

// Helper function to index 2-dimensional arrays
// Fortran order -> Column first
constexpr int index(int dim0, int dim1, const int i0, const int i1) {
    return i1 * dim0 + i0;
};

enum square
{
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
    /*std::vector <std::vector <square> > */
    int numOfBoxes;
    pos player;
    pos dim;
    /*pos*  boxes;*/
    square* map;
    std::vector <pos> boxes;
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
                printf ("%d ", game.map[board_idx]);
   /*         switch (game.map[i*game.dim.y+j])
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
                default:
                    printf ("!!!!!");
                    break;
            }
     */   }
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
    
//    printf ("%d %d \n", sizeH, sizeV);
//    result.map = (square*) calloc (sizeH * sizeV, sizeof (int));

    result.map = new square[sizeH * sizeV];
    
    for (int y = 0; y < sizeV; y++)
    {
        //std::vector<square> tst;
        for (int x = 0; x < sizeH; x++)
        {
            const int idx = index(sizeH, sizeV, x, y);
            result.map[idx] = FREE;
            //tst.push_back (FREE);
        }
        //result.map.push_back (tst);
    }

    result.dim.x = sizeH;
    result.dim.y = sizeV;

    dump(result);
    
    int numOfWalls = 0;
    myfile >> numOfWalls;
    
//    printf ("\n%d\n", numOfWalls);
    
    for (int i = 0; i < numOfWalls; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
  //      printf ("WALL: %d %d ", y, x);
        const int wall_idx = index(sizeH, sizeV, x - 1, y - 1);
        result.map[wall_idx] = WALL;
    }

//dump(result);
//printf ("BOXES!!!\n");    

    //int numOfBoxes = 0;
    myfile >> result.numOfBoxes;
    //pos* bxs = (pos*) calloc (result.numOfBoxes, 2*sizeof (int));
    //result.boxes = (pos*) calloc (result.numOfBoxes, sizeof (pos));

dump(result);

//    printf ("\n%d\n", numOfBoxes);
    for (int i = 0; i < result.numOfBoxes; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
  //      printf ("%d %d ", y, x);
        pos tmp;
        tmp.y = y-1;
        tmp.x = x-1;
	//bxs[i] = tmp;
	//result.boxes [i] = tmp;

        const int box_idx = index(sizeH, sizeV, x - 1, y - 1);
        result.boxes.push_back (tmp);
        result.map[box_idx] = square(1);//BOX;
  //      printf ("@@@BOX: %d ", result.map[(y-1)*sizeV+x-1]);
    }
    //result.boxes = (pos*) malloc (result.numOfBoxes*2* sizeof (int));
    //memcpy (result.boxes, bxs, 2*sizeof(int));// = 
dump(result);

    int numOfTargets = 0;
    myfile >> numOfTargets;
//    printf ("\n%d\n", numOfTargets);
    for (int i = 0; i < numOfTargets; i++)
    {
        int x = 0, y = 0;
        myfile >> y;
        myfile >> x;
//        printf ("TARGET: %d %d ", y, x);
        const int target_idx = index(sizeH, sizeV, x - 1, y - 1);
        result.map[target_idx] = square(3);//TARGET;
//        printf ("@@@!!! TARGET: %d ", result.map[(y-1)*sizeV+x-1]);
    }
    
dump(result);

    int player_x = 0, player_y = 0;
    myfile >> player_y >> player_x;
//    printf ("\n%d %d \n", player_y, player_x);
    result.player.x = player_x-1;
    result.player.y = player_y-1;

    const int player_idx = index(sizeH, sizeV, result.player.x, result.player.y);
    result.map[player_idx] = PLAYER;

    dump(result);

    return result;


}

bool isValid (sokoban current, pos move)
{
    if (move.y >= 0 and move.x < current.dim.x and move.y >= 0 and move.y < current.dim.y)
    {
     
  //      printf ("is Valid\n");
//        printf ("%d %d: %d\n", move.y, move.x, current.map[move.y*current.dim.y+move.x]);
        return (current.map[move.y*current.dim.y+move.x] != WALL);
    }
    else
    {
        
 //       printf ("NOT Valid\n");
        return false;
    }
}

bool hasBox (sokoban current, pos box)
{
    
    return (current.map[box.y*current.dim.y+box.x] == BOX);
}

sokoban makeMove (sokoban* current, char move)
{
    sokoban result = *current;
    pos toMove;
    switch (move)
    {
        case 'U':
            toMove.x = 0;// result.player.x;
            toMove.y = -1;// result.player.y-1;
   //         printf ("up\n");
            break;
        case 'D':
            toMove.x = 0;// result.player.x;
            toMove.y = 1;// result.player.y+1;
     //       printf ("down\n");
            break;
        case 'L':
            toMove.x = -1;// result.player.x-1;
            toMove.y = 0;// result.player.y;
     //       printf ("left\n");
            break;
        case 'R':
            toMove.x = 1;// result.player.x+1;
            toMove.y = 0;// result.player.y;
    //        printf ("right\n");
            break;
        default:
            printf ("Error on makeMove function!");
            break;
    }
    
    pos madeMove;
    madeMove.x = result.player.x + toMove.x;
    madeMove.y = result.player.y + toMove.y;
	
//	printf ("%d %d \n", madeMove.x, madeMove.y);
    if (isValid (*current, madeMove))
    {
  //      printf ("MOVE VALID\n");
        if (hasBox (*current, madeMove))
        {
            pos boxMove;
            boxMove.x = madeMove.x + toMove.x;
            boxMove.y = madeMove.y + toMove.y;
            if (isValid (*current, boxMove) && !hasBox (*current, boxMove))
            {
//		printf ("MOVING box \n");
		for (int i = 0; i < result.numOfBoxes; i++)
		{
			auto it = result.boxes[i];/*
                for (auto it = result.boxes.begin(); it != result.boxes.end(); it++)
                {*/
                    if (it.x == madeMove.x && it.y == madeMove.y)
                    {
                        it = boxMove;
                        result.map [result.player.y*result.dim.y+result.player.x] = FREE;
                        result.map [madeMove.y*result.dim.y+madeMove.x] = PLAYER;
                        result.map [boxMove.y*result.dim.y+boxMove.x] = BOX;
                        
                            result.player.x = madeMove.x;
                            result.player.y = madeMove.y;
                        //result.map[it->y][it->x]
                    }
                }/*
                                for (int i = 0; i < nboxes; i++)
                                {
                                    if (get_x(s.boxes[i]) == x and get_y(s.boxes[i]) == y)
                                    {
                                        s.boxes[i] = get_pos(bx, by);
                                        break;
                                    }
                                }*/
            }
        }
        else
        {
            result.map [result.player.y*result.dim.y+result.player.x] = FREE;
            result.map [madeMove.y*result.dim.y+madeMove.x] = PLAYER;
            result.player.x = madeMove.x;
            result.player.y = madeMove.y;
            
        }
    }
    else
    {
                printf ("@@@@INVALID\n");
    }
//dump (result);   
    return result;
}

int main(int argc, const char * argv[])
{
   // printf ("TTT\n");
    std::string input = "sokoban00.txt";
    if (argc >= 2)
    {
        input = std::string (argv[1]);
    }
    
    //printf ("getting input\n");
    
    sokoban body = scan (input);
    
    //printf ("scanning sokoban\n");
    
  //  dump (body);
    
    body = makeMove (&body, 'D');
    
 //   dump (body);
    
 //   printf ("made move \n");
    
}
