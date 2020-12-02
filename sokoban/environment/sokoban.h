#ifndef SOKOBAN_H
#define SOKOBAN_H

#include <vector>
#include <string>
#include <tuple>

enum square : uint8_t
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

using TStateVector = std::tuple<sokoban*, sokoban*, sokoban*, sokoban*>;
using TResultVector = std::tuple<bool, bool, bool, bool>;

sokoban* new_state();
sokoban* copy_state(const sokoban* const state);
std::vector<sokoban*> new_state_vector();
void delete_state(sokoban* state);

void dump (const sokoban& game);
sokoban* scan (const std::string& arg);
bool makeMove (const sokoban* const current, const char move, sokoban* output);

std::vector<bool> expand (const sokoban* const current, std::vector<sokoban*>& output);

// bool isHorisontalPairing (const pos& box1, const pos& box2);
// bool isVerticalPairing (const pos& box1, const pos& box2);
// bool isBlockedFromTop (const pos& box1, const pos& box2, const sokoban* const state);
// bool isBlockedFromBottom (const pos& box1, const pos& box2, const sokoban* const state);
// bool isBlockedFromLeft (const pos& box1, const pos& box2, const sokoban* const state);
// bool isBlockedFromRight (const pos& box1, const pos& box2, const sokoban* const state);

// bool isOnTarget (const pos& box, const sokoban* state);
// bool isBoxCornered (const pos& box, const sokoban* state);
bool isDeadlocked (const sokoban* const state);

#endif
