#ifndef SOKOBAN_H
#define SOKOBAN_H

#include <vector>
#include <string>
#include <tuple>

// the uint8_t enforces that this enum will be compatible with numpy
enum square : uint8_t
{
    ACQUIRED = 5,
    WALL = 4,
    TARGET = 3,
    PLAYER = 2,
    BOX = 1,
    FREE = 0
};

// SUPER DUPER SKETCHY
// We make sure that the struct is packed, I think this only works on GCC
// Also we swap the order or x and y because our board gets
// transposed when it goes through numpy for some reason.
typedef struct __attribute__ ((__packed__)) pos
{
    int y;
    int x;
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

using TStateVector = std::tuple<sokoban*, sokoban*, sokoban*, sokoban*>;
using TResultVector = std::tuple<bool, bool, bool, bool>;

sokoban* new_state();
sokoban* copy_state(const sokoban* const state);
std::vector<sokoban*> new_state_vector();
void delete_state(sokoban* state);

void dump (const sokoban& game);
sokoban* scan (const std::string& arg);
sokoban *generate(const std::string &wall_file, int num_targets, int num_steps);

bool makeMove (const sokoban* const current, const char move, sokoban* output);
bool inverseMove(const sokoban* const current, const char move, sokoban* output);
bool randomSequence(const sokoban* const current, const int length, sokoban* output);

std::vector<bool> expand (const sokoban* const current, std::vector<sokoban*>& output);

bool isDeadlocked (const sokoban* const state);

#endif
