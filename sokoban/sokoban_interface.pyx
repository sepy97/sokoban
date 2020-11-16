# cython: language_level=3, boundscheck=False, wraparound=False, overflowcheck=False, cdivision=True, initializedcheck=False, binding=False, embedsignature=True

from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool

import numpy as np
cimport numpy as np

from cpython.array cimport array, clone


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

    cdef bool makeMove(const sokoban* current, char move, sokoban* output)
    void dump (sokoban game)
    sokoban* scan(const string& arg)


cdef class SokobanState:
    """A wrapper class for a C/C++ data structure"""
    cdef sokoban *_state
    cdef np.int64_t[:, ::1] _state_buffer

    cdef bool ptr_owner
    cdef int size_x
    cdef int size_y

    def __cinit__(self):
        self.ptr_owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self._state is not NULL and self.ptr_owner is True:
            free(self._state)
            self._state = NULL

    # Extension class properties
    @property
    def map(self):
        if self._state is NULL or self.size_x == 0:
            return None

        return np.ctypeslib.as_array(self._state_buffer)

    @staticmethod
    cdef SokobanState from_state(sokoban *_state, bool owner = True):
        cdef SokobanState wrapper = SokobanState.__new__(SokobanState)
        wrapper._state = _state
        wrapper.size_x = _state.dim.x
        wrapper.size_y = _state.dim.y

        # This is a little bit sketchy, because we are coercing the enum into an int64_t
        # However, we define the enum to be an int64_t in the header file so its probably fine...
        wrapper._state_buffer = <np.int64_t[:wrapper.size_y, :wrapper.size_x]> <np.int64_t*> _state.map.data()

        wrapper.ptr_owner = owner
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
    return SokobanState.from_state(state, True)

cpdef SokobanState next_state(SokobanState state, int action):
    cdef char move = action_to_string(action)
    cdef sokoban *output = <sokoban *>malloc(sizeof(sokoban))

    cdef bool result = makeMove(state._state, move, output)

    return SokobanState.from_state(output, True)