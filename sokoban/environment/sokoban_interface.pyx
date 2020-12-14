# cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True, initializedcheck=False, binding=False, embedsignature=True

from cython.operator cimport dereference as deref

from libc.stdlib cimport malloc, free
from libc.stdint cimport uint8_t, int32_t

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

import numpy as np
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

    sokoban* new_state()
    sokoban* copy_state(sokoban* state)
    vector[sokoban*] new_state_vector()
    void delete_state(sokoban* state)

    void dump (const sokoban& game)
    sokoban* scan(const string& arg)
    sokoban* generate(const string& wall_file, int num_targets, int num_steps);
    bool checkSolved(const sokoban* const output);

    bool makeMove(const sokoban* const current, const char move, sokoban* output)
    bool inverseMove(const sokoban* const current, const char move, sokoban* output)
    
    vector[bool] expand (const sokoban* current, vector[sokoban*]& output)
    bool randomSequence(const sokoban* const current, const int count, sokoban* output)

    bool isDeadlocked(const sokoban* state);
    

cdef class SokobanState:
    """A wrapper class for a C/C++ data structure"""
    cdef sokoban *_state
    cdef uint8_t[:, ::1] _state_buffer
    cdef int32_t[:, ::1] _boxes_buffer
    cdef int32_t[:, ::1] _targets_buffer
    cdef int32_t[::1] _player_buffer
    cdef bool _solved

    cdef bool ptr_owner
    cdef int size_x
    cdef int size_y
    cdef int _num_boxes
    cdef int _num_targets

    def __cinit__(self, other=None):
        if other and type(other) is SokobanState:
            other_state = <SokobanState> other
            self.ptr_owner = True
            self.size_x = other_state.size_x
            self.size_y = other_state.size_y
            self._solved = other_state._solved

            self._state = copy_state(other_state._state)
            self._state_buffer = <uint8_t[:self.size_y, :self.size_x]> <uint8_t*> self._state.map.data()

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
        return expand_state(self)

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
        return random_sequence(self, length)

    cpdef bool _dead_lock(self):
        return isDeadlocked(self._state)

    # Extension class properties
    @property
    def map(self):
        if self._state is NULL or self.size_x == 0:
            return None

        return np.ctypeslib.as_array(self._state_buffer)

    @property
    def map_copy(self):
        if self._state is NULL or self.size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._state_buffer))

    @property
    def boxes(self):
        if self._state is NULL or self.size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._boxes_buffer))

    @property
    def targets(self):
        if self._state is NULL or self.size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._targets_buffer))

    @property
    def player(self):
        if self._state is NULL or self.size_x == 0:
            return None

        return np.copy(np.ctypeslib.as_array(self._player_buffer))
 
    @property
    def dead_lock(self):
        return self._dead_lock()

    @property
    def solved(self):
        return self._solved

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
        wrapper.size_x = _state.dim.x
        wrapper.size_y = _state.dim.y

        wrapper._num_boxes = _state.boxes.size()
        wrapper._num_targets = _state.targets.size()

        # This is a little bit sketchy, because we are coercing the enum into an uint8_t
        # However, we define the enum to be an uint8_t in the header file so its probably fine...
        wrapper._state_buffer = <uint8_t[:wrapper.size_y, :wrapper.size_x]> <uint8_t*> _state.map.data()
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

cpdef list expand_state(SokobanState state):
    cdef vector[sokoban*] output = new_state_vector()
    cdef vector[bool] solved = expand(state._state, output)
    
    return [SokobanState.from_state(output[i], solved[i]) for i in range(4)]

cpdef SokobanState random_sequence(SokobanState state, int length):
    cdef sokoban *output = new_state();

    cdef bool solved = randomSequence(state._state, length, output)

    return SokobanState.from_state(output, solved)

cpdef SokobanState generate_state(str wall_file, int num_targets, int num_steps):
    cdef sokoban* state = generate(wall_file.encode('UTF-8'), num_targets, num_steps)
    cdef bool solved = checkSolved(state)
    return SokobanState.from_state(state, solved)