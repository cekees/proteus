#ifndef SCOREC_DUMP_MESH_H
#define SCOREC_DUMP_MESH_H

#include <mesh.h>
#include <cstdio>
#include <PCU_C.h>

void dump_proteus_mesh(Mesh* m, FILE* f, PCU_t PCUObj);

#endif
