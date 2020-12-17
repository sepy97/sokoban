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

    bool const operator==(const sokoban &o) const {
        return this->map == o.map;
    }

    bool const operator<(const sokoban &o) const {
        return this->map < o.map;
    }

} sokoban;

sokoban* new_state();
std::vector<sokoban*> new_state_vector();
std::vector<sokoban*> new_state_vector(const int size);
sokoban* copy_state(const sokoban* const state);
void delete_state(sokoban* state);

void dump (const sokoban& game);
sokoban* scan (const std::string& arg);
sokoban *generate(const std::string &wall_file, int num_targets, int num_steps);

bool checkSolved(const sokoban* const output);
bool isDeadlocked (const sokoban* const state);

bool makeMove (const sokoban* const current, const char move, sokoban* output);
bool inverseMove(const sokoban* const current, const char move, sokoban* output);
bool randomSequence(const sokoban* const current, const int length, sokoban* output);

std::vector<bool> expand (const sokoban* const current, std::vector<sokoban*>& output);
std::vector<bool> parallelExpand (const std::vector<sokoban*>& current, std::vector<sokoban*>& output);

#endif
