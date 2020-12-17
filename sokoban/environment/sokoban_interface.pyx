# cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True, initializedcheck=False, binding=False, embedsignature=True

from cython.operator cimport dereference as deref

from libc.stdint cimport uint8_t, int32_t, int64_t, uint64_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.queue cimport priority_queue
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp cimport bool

import numpy as np
cimport numpy as np

from path import Path


cdef extern from "sokoban.h":
    cdef enum square:
        ACQUIRED = 5,
        WALL = 4,
        TARGET = 3,
        PLAYER = 2,
        BOX = 1,
        FREE = 0

    cdef struct pos:
        int x
        int y

    cdef struct sokoban:
        int numOfBoxes
        pos player
        pos dim
        vector[square] map
        vector[pos] boxes
        vector[pos] targets

    cdef sokoban* new_state()
    cdef sokoban* copy_state(sokoban* state)
    cdef vector[sokoban*] new_state_vector()
    cdef vector[sokoban*] new_state_vector(int size)
    cdef void delete_state(sokoban* state)

    cdef void dump (const sokoban& game)
    cdef sokoban* scan(const string& arg)
    cdef sokoban* generate(const string& wall_file, int num_targets, int num_steps)

    cdef bool isDeadlocked(const sokoban* state)
    cdef bool checkSolved(const sokoban* output)
    cdef bool makeMove(const sokoban* current, const char move, sokoban* output)
    cdef bool inverseMove(const sokoban* current, const char move, sokoban* output)
    cdef bool randomSequence(const sokoban* current, const int count, sokoban* output)

    cdef vector[bool] expand (const sokoban* current, vector[sokoban*]& output)
    cdef vector[bool] parallelExpand (const vector[sokoban*]& current, vector[sokoban*]& output)


cdef class SokobanState:
    cdef sokoban *_state
    cdef uint8_t[:, ::1] _state_buffer
    cdef int32_t[:, ::1] _boxes_buffer
    cdef int32_t[:, ::1] _targets_buffer
    cdef int32_t[::1] _player_buffer
    cdef bool _solved

    cdef bool ptr_owner
    cdef int _size_x
    cdef int _size_y
    cdef int _num_boxes
    cdef int _num_targets

    def __cinit__(self, other=None):
        if other and type(other) is SokobanState:
            other_state = <SokobanState> other
            self.ptr_owner = True
            self._size_x = other_state._size_x
            self._size_y = other_state._size_y
            self._solved = other_state._solved
            self._num_boxes = other_state._num_boxes
            self._num_targets = other_state._num_targets

            self._state = copy_state(other_state._state)
            self._state_buffer = <uint8_t[:self._size_y, :self._size_x]> <uint8_t*> self._state.map.data()
            self._boxes_buffer = <int32_t[:self._num_boxes, :2]> <int32_t*> self._state.boxes.data()
            self._targets_buffer = <int32_t[:self._num_targets, :2]> <int32_t*> self._state.targets.data()
            self._player_buffer = <int32_t[:2]> <int32_t*> &self._state.player

        else:
            self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._state is not NULL and self.ptr_owner is True:
            delete_state(self._state)
            self._state = NULL

    cpdef display(self):
        dump(deref(self._state))

    cpdef list expand(self):
        cdef vector[sokoban*] output = new_state_vector()
        cdef vector[bool] solved = expand(self._state, output)
    
        return [SokobanState.from_state(output[i], solved[i]) for i in range(4)]

    cpdef SokobanState next_state(self, int action):
        cdef char move = action_to_string(action)
        cdef sokoban *output = new_state();

        cdef bool solved = makeMove(self._state, move, output)

        return SokobanState.from_state(output, solved)

    cpdef SokobanState previous_state(self, int action):
        cdef char move = action_to_string(action)
        cdef sokoban *output = new_state();

        cdef bool solved = inverseMove(self._state, move, output)

        return SokobanState.from_state(output, solved)

    cpdef SokobanState random_sequence(self, int length):
        cdef sokoban *output = new_state();
        cdef bool solved = randomSequence(self._state, length, output)
        return SokobanState.from_state(output, solved)

    cpdef bool _dead_lock(self):
        return isDeadlocked(self._state)

    # Extension class properties
    @property
    def map(self):
        if self._state is NULL or self._size_x == 0:
            return None

        return np.ctypeslib.as_array(self._state_buffer)

    @property
    def map_copy(self):
        if self._state is NULL or self._size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._state_buffer))

    @property
    def boxes(self):
        if self._state is NULL or self._size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._boxes_buffer))

    @property
    def targets(self):
        if self._state is NULL or self._size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._targets_buffer))

    @property
    def player(self):
        if self._state is NULL or self._size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._player_buffer))
 
    @property
    def dead_lock(self):
        return self._dead_lock()

    @property
    def solved(self):
        return self._solved

    @property
    def size_x(self):
        return self._size_x

    @property
    def size_y(self):
        return self._size_y

    def __hash__(self):
        return hash(self.map.tobytes())

    def __eq__(self, other):
        return (self.map == other.map).all()

    def __repr__(self):
        return str(self.map)

    def __str__(self):
        return str(self.map)

    @staticmethod
    def load(filepath):
        if not Path(filepath).exists():
            raise FileNotFoundError()

        return load_state(filepath)

    @staticmethod
    def generate(wall_file, num_targets, num_steps):
        if not Path(wall_file).exists():
            raise FileNotFoundError()

        return generate_state(wall_file, num_targets, num_steps)

    @staticmethod
    cdef SokobanState from_state(sokoban *_state, bool solved, bool owner = True):
        cdef SokobanState wrapper = SokobanState.__new__(SokobanState)
        wrapper._state = _state
        wrapper._size_x = _state.dim.x
        wrapper._size_y = _state.dim.y

        wrapper._num_boxes = _state.boxes.size()
        wrapper._num_targets = _state.targets.size()

        # This is a little bit sketchy, because we are coercing the enum into an uint8_t
        # However, we define the enum to be an uint8_t in the header file so its probably fine...
        wrapper._state_buffer = <uint8_t[:wrapper._size_y, :wrapper._size_x]> <uint8_t*> _state.map.data()
        wrapper._boxes_buffer = <int32_t[:wrapper._num_boxes, :2]> <int32_t*> _state.boxes.data()
        wrapper._targets_buffer = <int32_t[:wrapper._num_targets, :2]> <int32_t*> _state.targets.data()
        wrapper._player_buffer = <int32_t[:2]> <int32_t*> &_state.player
        wrapper.ptr_owner = owner
        wrapper._solved = solved
        return wrapper


cdef char action_to_string(int action):
    if action == 0:
        return b'U'
    elif action == 1:
        return b"R"
    elif action == 2:
        return b"D"
    elif action == 3:
        return b"L"

cpdef SokobanState load_state(str filepath):
    cdef sokoban* state = scan(filepath.encode('UTF-8'))
    return SokobanState.from_state(state, False)

cpdef SokobanState generate_state(str wall_file, int num_targets, int num_steps):
    cdef sokoban* state = generate(wall_file.encode('UTF-8'), num_targets, num_steps)
    cdef bool solved = checkSolved(state)
    return SokobanState.from_state(state, solved)

cpdef SokobanState[::1] parallel_expand(SokobanState[::1] states):
    cdef vector[sokoban*] current 
    for i in range(states.shape[0]):
        current.push_back(states[i]._state)

    cdef vector[sokoban*] output = new_state_vector(states.shape[0])
    cdef vector[bool] solved = parallelExpand(current, output)

    return np.array([SokobanState.from_state(output[i], solved[i]) for i in range(4 * states.shape[0])])


cdef void _states_to_numpy(vector[sokoban*] states, uint8_t[:, :, ::1] output, uint64_t size):
    cdef uint64_t i, j, k, x_offset, y_offset
    cdef sokoban* state
    cdef uint8_t[:, ::1] state_buffer

    for i in range(states.size()):
        state = states[i]
        state_buffer = <uint8_t[:state.dim.x, :state.dim.y]> <uint8_t*> state.map.data()

        x_offset = (size - state.dim.x) // 2
        y_offset = (size - state.dim.y) // 2
        for j in range(state.dim.x):
            for i in range(state.dim.y):
                output[i, x_offset + j, y_offset + k] = state_buffer[j, k]


# cpdef states_to_numpy(SokobanState[::1] states, uint64_t size):
#     cdef vector[sokoban*] _states
#     for i in range(states.shape[0]):
#         _states.push_back(states[i]._state)
#
#     cdef uint8_t[:, :, ::1] output = np.full((_states.size(), size, size))

cdef class AstarData:
    cdef uint64_t current_length
    cdef uint64_t current_size

    cdef int64_t[::1] parents
    cdef int64_t[::1] actions
    cdef float[::1] costs
    cdef int64_t[::1] state_table

    cdef priority_queue[pair[float, int64_t]] open_set
    cdef vector[pair[sokoban, bool]] inverse_state_mapping
    cdef map[sokoban, int64_t] state_mapping

    def __init__(self, initial_size = 1024):
        self.current_length = 0
        self.current_size = initial_size

        self.parents = np.full(initial_size, -1, dtype=np.int64)
        self.actions = np.full(initial_size, -1, dtype=np.int64)
        self.costs = np.full(initial_size, np.inf, dtype=np.float32)
        self.state_table = np.full(initial_size, np.iinfo(np.int64).max, dtype=np.int64)

    def __len__(self):
        return self.current_length

    cpdef add(self, SokobanState state):
        cdef sokoban _state = deref(state._state)
        cdef bool solved = state._solved

        if self.state_mapping.find(_state) != self.state_mapping.end():
            return

        cdef int64_t[::1] new_state_table
        cdef int64_t[::1] new_actions
        cdef float[::1] new_costs

        if self.current_length >= self.current_size:
            self.current_size = self.current_size * 2

            new_parents = np.full(self.current_size, -1, dtype=np.int64)
            new_actions = np.full(self.current_size, -1, dtype=np.int64)
            new_costs = np.full(self.current_size, np.inf, dtype=np.float32)
            new_state_table = np.full(self.current_size, np.iinfo(np.int64).max, dtype=np.int64)

            new_costs[:self.current_length] = self.costs[:]
            new_parents[:self.current_length] = self.parents[:]
            new_actions[:self.current_length] = self.actions[:]
            new_state_table[:self.current_length] = self.state_table[:]

            self.costs = new_costs
            self.parents = new_parents
            self.actions = new_actions
            self.state_table = new_state_table

        self.state_mapping[_state] = self.current_length
        self.inverse_state_mapping.push_back(pair[sokoban, bool](_state, solved))

        self.current_length += 1

    cpdef has_elements(self):
        return self.open_set.size() > 0

    cpdef void set_cost(self, int64_t state_id, float cost):
        self.costs[state_id] = cost

    cpdef void push_id(self, int64_t state_id, float priority):
        self.open_set.push(pair[float, int64_t](-priority, state_id))

    cpdef int64_t pop_id(self):
        cdef pair[float, int64_t] top = self.open_set.top()
        self.open_set.pop()

        return top.second

    cpdef void push(self, SokobanState state, float priority):
        self.add(state)
        self.push_id(self.state_to_id(state), priority)

    cpdef SokobanState pop(self):
        return self.id_to_state(self.pop_id())

    cpdef int64_t state_to_id(self, SokobanState state):
        return self.state_mapping[deref(state._state)]

    cpdef SokobanState id_to_state(self, int64_t id):
        cdef pair[sokoban, bool]* state = &self.inverse_state_mapping[id]
        return SokobanState.from_state(&deref(state).first, deref(state).second, False)

    cpdef extract_path(self, start, goal):
        path = []
        actions = []

        start_id = self.state_to_id(start)
        state_id = self.state_to_id(goal)

        while state_id != start_id:
            path.append(SokobanState(self.id_to_state(state_id)))
            action = self.actions[state_id]
            state_id = self.parents[state_id]
            actions.append(action)

        path.append(start)
        path.reverse()
        actions.reverse()
        return path, actions

    cpdef pop_batch(self, int64_t batch_size):
        cdef int64_t size = min(batch_size, self.open_set.size())

        cdef uint64_t i
        cdef int64_t state_id
        cdef int64_t[::1] state_ids = np.empty(size, np.int64)
        cdef list states = []

        for i in range(size):
            state_id = self.pop_id()
            state_ids[i] = state_id
            states.append(self.id_to_state(state_id))

        return np.array(states), state_ids

    cpdef SokobanState check_solved(self, SokobanState[::1] states):
        cdef uint64_t i
        for i in range(states.shape[0]):
            if states[i].solved:
                return states[i]

        return None

    cpdef void process_children(self,
                                int64_t[::1] state_ids,
                                float[::1] heuristics,
                                SokobanState[::1] children,

                                int64_t num_of_moves,
                                float weight,
                                ):
        cdef uint64_t num_children = children.shape[0]
        cdef uint64_t i
        cdef SokobanState child

        state_ids = np.repeat(state_ids, 4)
        cdef bool[::1] dead_locks = np.empty(num_children, np.bool)
        cdef int64_t[::1] children_ids = np.empty(num_children, np.int64)
        cdef float[::1] state_costs = np.empty(num_children, np.float32)

        for i in range(num_children):
            child = children[i]
            self.add(child)
            dead_locks[i] = child._dead_lock()
            children_ids[i] = self.state_to_id(child)
            state_costs[i] = self.costs[state_ids[i]]

        self.astar_update(
            state_ids,
            state_costs,
            children_ids,
            heuristics,
            dead_locks,

            num_of_moves,
            weight,
        )


    cpdef void astar_update(
            self,
            int64_t[::1] state_ids,
            float[::1] state_costs,
            int64_t[::1] children,
            float[::1] heuristics,
            bool[::1] deadlocks,

            int64_t num_of_moves,
            float weight,
        ):
            cdef int64_t num_output = 0
            cdef int64_t child
            cdef int64_t parent
            cdef int64_t stored_num_of_moves
            cdef int64_t action
            cdef float cost

            cdef uint64_t i
            cdef uint64_t num_children = children.shape[0]

            for i in range(num_children):
                child = children[i]
                parent = state_ids[i]

                if deadlocks[i]:
                    continue

                stored_num_of_moves = self.state_table[child]
                if stored_num_of_moves <= num_of_moves:
                    continue

                action = i % 4
                cost = state_costs[i] + 1
                self.state_table[child] = num_of_moves

                if cost < self.costs[child]:
                    self.parents[child] = parent
                    self.actions[child] = action
                    self.costs[child] = cost

                    self.push_id(child, cost + weight * heuristics[i])