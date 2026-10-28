#ifndef PROTEUS_INDEX_TYPE_H
#define PROTEUS_INDEX_TYPE_H
#include <stdint.h>

/**
   \file index_type.h
   \brief The integer type of mesh entity numbers, counts and offsets.

   One definition read by both sides: mesh.h declares struct Mesh with it, and
   cmeshTools exports its width to Python as cmeshTools.INDEX_DTYPE, which is
   the dtype of every index array a CMesh hands to numpy.

   Changing the width is not yet a one-line change: the kernels and the Python
   modules still say int and 'i'. Changing it here makes every place that
   disagrees a compile error, or a failure in test/test_index_type.py.

   Local numbers with a small fixed range (an element's own faces, material
   tags) are a different type and stay int.
*/
typedef int32_t proteus_index_t;

#endif
