#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <tuple>

#include "sokoban.h"


// Helper function to index 2-dimensional arrays
// Fortran order -> Column first
constexpr int index(const int dim0, const int dim1, const int i0, const int i1) {
    return i1 * dim0 + i0;
};

// Dummy function that is necessary to dynamically create the sokoban struct from python.
sokoban* new_state() {
    return new sokoban;
}

// Dummy function that is necessary to dynamically copy a sokoban struct from python
sokoban* copy_state(const sokoban* const state) {
    sokoban* output = new_state();
    *output = *state;
    return output;
}

// Dummy function that is necessary to dynamically create the sokoban struct from python.
std::vector<sokoban*> new_state_vector() {
    return {new_state(), new_state(), new_state(), new_state()};
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
                    // printf ("$");
                    printf("▇");
                    break;
                case PLAYER:
                    // printf("@");
                    printf("☺");
                    break;
                case TARGET:
                    // printf(".");
                    printf(".");
                    break;
                case WALL:
                    // printf ("#");
                    printf("▒");
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
    if (player_idx > 0) {
        result->map[player_idx] = PLAYER;
    }

    return result;
}

sokoban* generate(const std::string &wall_file, int num_targets, int num_steps) {
    auto state = scan(wall_file);
    const auto size_x = state->dim.x;
    const auto size_y = state->dim.y;

    int x, y;

    while (state->numOfBoxes < num_targets) {
        x = rand() % size_x;
        y = rand() % size_y;
        const auto current_index = index(size_x, size_y, x, y);

        if (state->map[current_index] != FREE)
            continue;

        state->map[current_index] = BOX;
        state->targets.push_back({.y = y, .x = x});
        state->boxes.push_back({.y = y, .x = x});
        state->numOfBoxes++;
    }

    bool found_location = false;
    for (auto& delta : {-1, 1}) {
        for (auto& vertical : {false, true}) {
            const auto player_x = vertical ? x : x + delta;
            const auto player_y = vertical ? y + delta : y;
            const auto current_index = index(size_x, size_y, player_x, player_y);

            if (!found_location && state->map[current_index] == FREE) {
                state->map[current_index] = PLAYER;
                state->player.x = player_x;
                state->player.y = player_y;
                found_location = true;
            }
        }
    }

    if (!found_location) {
        printf("No location found for player, very bad wall configuration!");
    }

    dump(*state);
    auto output = new_state();
    randomSequence(state, num_steps, output);

    delete_state(state);
    return output;
}

constexpr bool isValid (const sokoban& current, const pos& move)
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

bool hasBox (const sokoban& current, const pos& box)
{
    return (current.map[index (current.dim.x, current.dim.y, box.x, box.y)] == BOX);// || current.map[index (current.dim.x, current.dim.y, box.x, box.y) == ACQUIRED);
}

pos createMoveDelta(const char move) {
    pos toMove {0, 0};

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

    return toMove;
}

// The full update board after moving a box
void updateState(sokoban* output, const pos& old_player, const pos& new_player, const pos& old_box, const pos& new_box)
{
    // Update player location
    output->player.x = new_player.x;
    output->player.y = new_player.y;

    // Find the box that was moved and update its location
    for (auto& box : output->boxes)
    {
        if (box.x == old_box.x && box.y == old_box.y)
        {
            box.x = new_box.x;
            box.y = new_box.y;

            // We move at most one box every action, so if we found it we can break out of the loop
            break;
        }
    }

    // Update the box action on the map
    output->map [index (output->dim.x, output->dim.y, old_box.x, old_box.y)] = FREE;
    output->map [index (output->dim.x, output->dim.y, old_player.x, old_player.y)] = FREE;

    output->map [index (output->dim.x, output->dim.y, new_box.x, new_box.y)] = BOX;
    output->map [index (output->dim.x, output->dim.y, new_player.x, new_player.y)] = PLAYER;

    // Re-add any targets to the map that were covered up before
    // TODO: We dont actually have to check every target, only the box and player ones
    for (auto& target : output->targets)
    {
        const auto current_index = index (output->dim.x, output->dim.y, target.x, target.y);
        if (output->map [current_index] == FREE) {
            output->map [current_index] = TARGET;
        }
    }
}

// The partial update board if we only move a player and not a box
void updateState(sokoban* output, const pos& old_player, const pos& new_player) {
    // Update player location
    output->player.x = new_player.x;
    output->player.y = new_player.y;

    // Update the map with the new player sprite
    output->map [index (output->dim.x, output->dim.y, old_player.x, old_player.y)] = FREE;
    output->map [index (output->dim.x, output->dim.y, new_player.x, new_player.y)] = PLAYER;

    // Re-add any targets to the map that were covered up before
    // TODO: We dont actually have to check every target, only the target the player was over
    for (auto& target : output->targets)
    {
        const auto current_index = index(output->dim.x, output->dim.y, target.x, target.y);
        if (output->map[current_index] == FREE) {
            output->map[current_index] = TARGET;
        }
    }
}

bool checkSolved(const sokoban* const output) {
    for (auto& target : output->targets)
    {
        if (output->map [index (output->dim.x, output->dim.y, target.x, target.y)] != BOX)
            return false;
    }
    return true;
}

bool makeMove (const sokoban* const current, const char move, sokoban* output)
{
    if (output == nullptr) {
        output = new_state();
    }

    *output = *current;

    const pos move_delta = createMoveDelta(move);

    const pos& old_player = current->player;
    const pos new_player {
        .y = old_player.y + move_delta.y,
        .x = old_player.x + move_delta.x
    };
	
    if (isValid (*current, new_player))
    {
        if (hasBox (*current, new_player))
        {
            const pos& old_box = new_player;
            const pos new_box {
                .y = old_box.y + move_delta.y,
                .x = old_box.x + move_delta.x
            };

            if (isValid (*current, new_box) && !hasBox (*current, new_box))
            {
                updateState (output, old_player, new_player, old_box, new_box);
            }
        }
        else
        {
            updateState (output, old_player, new_player);
        }
    }

    return checkSolved(output);
}

bool inverseMove(const sokoban* const current, const char move, sokoban* output) {
    if (output == nullptr) {
        output = new_state();
    }

    *output = *current;

    // Create two sets of moves.
    // The forward move which is used to check for boxes and
    // The inverse move, which moves the player and is used to check for validity
    const pos move_delta = createMoveDelta(move);

    const pos forward_move {
        .y = output->player.y + move_delta.y,
        .x = output->player.x + move_delta.x
    };

    const pos inverse_move {
        .y = output->player.y - move_delta.y,
        .x = output->player.x - move_delta.x
    };

    const pos& old_player = current->player;
    const pos& new_player = inverse_move;

    // Notice that only the inverse move has to be valid, since any boxes will always move into a valid position.
    // However, we need to check if there is another box behind us because we can't push in this version.
    if (isValid (*current, new_player) && !hasBox (*current, new_player)) {

        // Now we check if the forward move has a box to see if we should pull something.
        // Notice we dont need to check if there is another box in front of the current box
        if (hasBox (*current, forward_move)) {
            const pos& old_box = forward_move;

            const pos new_box {
                .y = old_box.y - move_delta.y,
                .x = old_box.x - move_delta.x
            };

            updateState (output, old_player, new_player, old_box, new_box);
        }
        else {
            updateState (output, old_player, new_player);
        }
    }

    return checkSolved(output);
}

std::vector<bool> expand (const sokoban* const current, std::vector<sokoban*>& output)
{
	//std::vector <sokoban> result;
	const char moves[4] = {'U', 'R', 'D', 'L'};
    std::vector<bool> solved(4);

    //try to parallelize this loop with omp
	for (int i = 0; i < 4; i++) {
		solved[i] = makeMove (current, moves[i], output[i]);
	}
	
	return solved;
}

bool randomSequence(const sokoban* const current, const int count, sokoban* output) {
    const char moves[4] = {'U', 'R', 'D', 'L'};

    bool result = false;
    sokoban* state_1 = new_state();
    sokoban* state_2 = new_state();

    *state_1 = *current;

    for (int i = 0; i < count; ++i) {
        const int action = rand() % 4;

        result = inverseMove(state_1, moves[action], state_2);

        std::swap(state_1, state_2);
    }

    if (output == nullptr) {
        output = new_state();
    }

    *output = *state_1;

    delete_state(state_1);
    delete_state(state_2);

    return result;
}

bool isOnTarget (const pos& box, const sokoban* const state)
{
    for (int i = 0; i < state->targets.size (); i++)
    {
        if ( (box.x == state->targets[i].x) && (box.y == state->targets[i].y) ) return true;
    }
    return false;
}

constexpr bool isBoxCornered (const pos& box, const sokoban* const state)
{
    if (box.x == 0 && box.y == 0)
    {
        return true;
    }
    else if (box.x == 0 && box.y == state->dim.y-1)
    {
        return true;
        
    }
    else if (box.y == 0 && box.x == state->dim.x-1)
    {
        return true;
    }
    else if (box.x == state->dim.x-1 && box.y == state->dim.y-1)
    {
        return true;
    }
    else
    {
        if (box.x == 0)
        {
            if (state->map[index (state->dim.x, state->dim.y, box.x+1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y-1)] == WALL) return true;
            if (state->map[index (state->dim.x, state->dim.y, box.x+1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y+1)] == WALL) return true;
        }
        else if (box.y == 0)
        {
            if (state->map[index (state->dim.x, state->dim.y, box.x-1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y+1)] == WALL) return true;
            if (state->map[index (state->dim.x, state->dim.y, box.x+1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y+1)] == WALL) return true;
        }
        else if (box.x == state->dim.x-1)
        {
            if (state->map[index (state->dim.x, state->dim.y, box.x-1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y-1)] == WALL) return true;
            if (state->map[index (state->dim.x, state->dim.y, box.x-1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y+1)] == WALL) return true;
        }
        else if (box.y == state->dim.y-1)
        {
            if (state->map[index (state->dim.x, state->dim.y, box.x-1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y-1)] == WALL) return true;
            if (state->map[index (state->dim.x, state->dim.y, box.x+1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y-1)] == WALL) return true;
        }
        else
        {
            if (state->map[index (state->dim.x, state->dim.y, box.x-1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y-1)] == WALL) return true;
            if (state->map[index (state->dim.x, state->dim.y, box.x+1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y-1)] == WALL) return true;
            if (state->map[index (state->dim.x, state->dim.y, box.x-1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y+1)] == WALL) return true;
            if (state->map[index (state->dim.x, state->dim.y, box.x+1, box.y)] == WALL && state->map[index (state->dim.x, state->dim.y, box.x, box.y+1)] == WALL) return true;
        }
    }
    
    return false;
}

constexpr bool isHorisontalPairing (const pos& box1, const pos& box2)
{
    if (box1.y == box2.y)
    {
        return ( ((box1.x - box2.x) * (box1.x - box2.x)) == 1 );
    }
    return false;
}

constexpr bool isVerticalPairing (const pos& box1, const pos& box2)
{
    if (box1.x == box2.x)
    {
        return ( ((box1.y - box2.y) * (box1.y - box2.y)) == 1 );
    }
    return false;
}

constexpr bool isBlockedFromTop (const pos& box1, const pos& box2, const sokoban* const state)
{
    if (box1.y == 0 || box2.y == 0) return true;
    if ((state->map[index (state->dim.x, state->dim.y, box1.x, box1.y-1)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box1.x, box1.y-1)] == BOX) &&
        (state->map[index (state->dim.x, state->dim.y, box2.x, box2.y-1)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box2.x, box2.y-1)] == BOX)) return true;
    return false;
}

constexpr bool isBlockedFromBottom (const pos& box1, const pos& box2, const sokoban* const state)
{
    if (box1.y == state->dim.y-1 || box2.y == state->dim.y-1) return true;
    if ((state->map[index (state->dim.x, state->dim.y, box1.x, box1.y+1)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box1.x, box1.y+1)] == BOX) &&
        (state->map[index (state->dim.x, state->dim.y, box2.x, box2.y+1)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box2.x, box2.y+1)] == BOX)) return true;
    return false;
}

constexpr bool isBlockedFromLeft (const pos& box1, const pos& box2, const sokoban* const state)
{
    if (box1.x == 0 || box2.x == 0) return true;
    if ((state->map[index (state->dim.x, state->dim.y, box1.x-1, box1.y)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box1.x-1, box1.y)] == BOX) &&
        (state->map[index (state->dim.x, state->dim.y, box2.x-1, box2.y)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box2.x-1, box2.y)] == BOX)) return true;

    return false;
}

constexpr bool isBlockedFromRight (const pos& box1, const pos& box2, const sokoban* const state)
{
    if (box1.x == state->dim.x-1 || box2.x == state->dim.x-1) return true;
    if ((state->map[index (state->dim.x, state->dim.y, box1.x+1, box1.y)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box1.x+1, box1.y)] == BOX) &&
        (state->map[index (state->dim.x, state->dim.y, box2.x+1, box2.y)] == WALL ||
         state->map[index (state->dim.x, state->dim.y, box2.x+1, box2.y)] == BOX)) return true;
    return false;
}

bool isDeadlocked (const sokoban* const state)
{
    for (int i = 0; i < state->numOfBoxes; i++)
    {
        if (!isOnTarget (state->boxes[i], state))
        {
            if (isBoxCornered (state->boxes[i], state)) return true;
            //else
            for (int j = 0; j < state->numOfBoxes; j++)
            {
                if (isHorisontalPairing (state->boxes[i], state->boxes[j]))
                {
                    if (isBlockedFromTop (state->boxes[i], state->boxes[j], state)) return true;
                        
                    if (isBlockedFromBottom (state->boxes[i], state->boxes[j], state)) return true;
                        
                }
                else if (isVerticalPairing (state->boxes[i], state->boxes[j]))
                {
                    if (isBlockedFromLeft (state->boxes[i], state->boxes[j], state)) return true;
                            
                    if (isBlockedFromRight (state->boxes[i], state->boxes[j], state)) return true;
                            
                }
            }
        }
    }
    
    return false;
}
