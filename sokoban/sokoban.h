#ifndef SOKOBAN_H
#define SOKOBAN_H

#include <vector>
#include <string>

enum square : int64_t
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
    std::vector<square> map;
    std::vector<pos> boxes;
    std::vector<pos> targets;
   // int numTargets;
} sokoban;


void dump (sokoban game);
sokoban* scan (const std::string& arg);
bool makeMove (const sokoban* current, char move, sokoban* output);

#endif