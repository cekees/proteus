#include "partitioning.h"
#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <vector>

/* A regular quadrilateral or hexahedral grid generated directly in pieces: each
   rank builds its own subdomain mesh, with the numbering and the maps
   partitionNodes produces when it is given the global grid from
   generateQuadrilateralMeshFromRectangularGrid or
   generateHexahedralMeshFromRectangularGrid and MeshTools.cartesianNodeOwners
   for the same rank grid -- entity for entity, without the global mesh, a graph
   partitioner or PETSc. See CARTESIAN_PARTITIONER_CAMPAIGN.md, Phase 2.

   What has to be reproduced, and how:

   - Old numbers are the generators'. Nodes: quads y-fastest (i*nny + j),
     hexes x-fastest. Elements: x-fastest for both. Element boundaries and edges
     are numbered by the std::map/std::set over sorted node tuples that
     constructElementBoundaryElementsArray_{quadrilateral,hexahedron} build, so
     an entity's old number is the number of entities whose sorted tuple of old
     node numbers is lexicographically smaller. On a grid that is a count over
     boxes (EntityFamily::oldNumber). Quad "edges" are every node pair of an
     element, diagonals included; hex edges are the twelve cell edges.
   - Owners are partitionNodes's: a node's owner is its brick; an element or
     element boundary belongs to the lowest rank that reaches it from an owned
     node through the node's element star -- an element through any of its
     nodes, an element boundary through the nodes partitionNodes selects (the
     face's own nodes for hexahedra, every node of the element but the one with
     the face's local number for quadrilaterals) -- and an edge to the owner of
     its node with the lower new number.
   - New numbers: each rank's owned entities in increasing old number, ranks in
     order. Nodes and elements are bricks, so their new numbers are arithmetic.
     Element boundaries and edges are counted per rank (MPI_Allgather) and the
     new numbers of ghosts are asked of their owners (MPI_Alltoallv).
   - The subdomain is what partitionNodes gives with one layer of overlap: the
     elements with a node in the brick, and their nodes, element boundaries and
     edges, owned ones first and ghosts after in increasing new number. Its
     connectivity and geometry come from the same builders partitionNodes uses.
*/
namespace proteus
{
namespace
{
typedef long long Index;
typedef std::array<int,3> Ijk;

/* A family of entities (element boundaries or edges) on the node grid: each
   type is a set of node offsets from the entity's node with the lowest old
   number (its origin). */
struct EntityType
{
  std::vector<Ijk> offsets;
  int lo[3], hi[3];
};

class CartesianGrid
{
public:
  int nNodes_element, dim, size;
  int nn[3], nc[3], p[3];
  std::vector<int> starts[3];
  Index nodeStride[3], cellStride[3];
  int nodeOrder[3], cellOrder[3]; //axes from slowest to fastest
  std::vector<Ijk> elementNodes;  //node offsets of an element, in element order
  double h[3];

  CartesianGrid(int nNodes_element_in, const int* nnIn, const double* L, const int* rankGrid)
  {
    nNodes_element = nNodes_element_in;
    dim = nNodes_element == 8 ? 3 : 2;
    for (int d = 0; d < 3; d++)
      {
        nn[d] = d < dim ? nnIn[d] : 1;
        nc[d] = d < dim ? nn[d] - 1 : 1;
        p[d] = rankGrid[d];
        h[d] = d < dim ? L[d]/(nn[d]-1.0) : 0.0;
        starts[d].resize(p[d]+1);
        starts[d][0] = 0;
        for (int a = 0; a < p[d]; a++)
          starts[d][a+1] = starts[d][a] + nn[d]/p[d] + (nn[d]%p[d] > a);
      }
    size = p[0]*p[1]*p[2];
    if (dim == 2)
      {
        //quadrilateral nodes are numbered y-fastest, elements x-fastest
        nodeStride[0] = nn[1]; nodeStride[1] = 1; nodeStride[2] = 0;
        nodeOrder[0] = 2; nodeOrder[1] = 0; nodeOrder[2] = 1;
        elementNodes = {{0,0,0},{0,1,0},{1,1,0},{1,0,0}};
      }
    else
      {
        nodeStride[0] = 1; nodeStride[1] = nn[0]; nodeStride[2] = Index(nn[0])*nn[1];
        nodeOrder[0] = 2; nodeOrder[1] = 1; nodeOrder[2] = 0;
        elementNodes = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
      }
    cellStride[0] = 1; cellStride[1] = nc[0]; cellStride[2] = Index(nc[0])*nc[1];
    cellOrder[0] = 2; cellOrder[1] = 1; cellOrder[2] = 0;
  }

  int chunk(int d, int i) const
  {
    return int(std::upper_bound(starts[d].begin()+1, starts[d].end(), i) - (starts[d].begin()+1));
  }
  int rankOf(int a, int b, int c) const { return a + p[0]*(b + p[1]*c); }
  int nodeOwner(const Ijk& n) const { return rankOf(chunk(0,n[0]), chunk(1,n[1]), chunk(2,n[2])); }
  void brick(int r, int lo[3], int hi[3]) const
  {
    const int a[3] = {r % p[0], (r/p[0]) % p[1], r/(p[0]*p[1])};
    for (int d = 0; d < 3; d++)
      {
        lo[d] = starts[d][a[d]];
        hi[d] = starts[d][a[d]+1];
      }
  }
  //the elements a rank owns: those whose lowest node is in its brick
  void cellBrick(int r, int lo[3], int hi[3]) const
  {
    brick(r, lo, hi);
    for (int d = 0; d < 3; d++)
      hi[d] = std::max(lo[d], std::min(hi[d], nc[d]));
  }
  Index nodeOld(const Ijk& n) const { return n[0]*nodeStride[0] + n[1]*nodeStride[1] + n[2]*nodeStride[2]; }
  Index cellOld(const Ijk& c) const { return c[0]*cellStride[0] + c[1]*cellStride[1] + c[2]*cellStride[2]; }
  bool nodeExists(const Ijk& n) const
  {
    for (int d = 0; d < 3; d++)
      if (n[d] < 0 || n[d] >= nn[d])
        return false;
    return true;
  }
  bool cellExists(const Ijk& c) const
  {
    for (int d = 0; d < 3; d++)
      if (c[d] < 0 || c[d] >= nc[d])
        return false;
    return true;
  }
  Ijk elementNode(const Ijk& c, int nN) const
  {
    const Ijk& o = elementNodes[nN];
    return Ijk{c[0]+o[0], c[1]+o[1], c[2]+o[2]};
  }
  //position of x in the box [lo,hi), axes taken slowest first in order
  static Index boxIndex(const Ijk& x, const int lo[3], const int hi[3], const int order[3])
  {
    Index local = 0;
    for (int o = 0; o < 3; o++)
      {
        const int d = order[o];
        local = local*(hi[d]-lo[d]) + (x[d]-lo[d]);
      }
    return local;
  }
  static Index boxSize(const int lo[3], const int hi[3])
  {
    return Index(hi[0]-lo[0])*(hi[1]-lo[1])*(hi[2]-lo[2]);
  }
};

/* Numbering of nodes and elements: bricks, so arithmetic. */
class BrickNumbering
{
public:
  BrickNumbering(const CartesianGrid& g, bool cells) : g_(g), cells_(cells), offsets_(g.size+1)
  {
    offsets_[0] = 0;
    for (int r = 0; r < g.size; r++)
      {
        int lo[3], hi[3];
        box(r, lo, hi);
        offsets_[r+1] = offsets_[r] + CartesianGrid::boxSize(lo, hi);
      }
  }
  void box(int r, int lo[3], int hi[3]) const
  {
    if (cells_)
      g_.cellBrick(r, lo, hi);
    else
      g_.brick(r, lo, hi);
  }
  int owner(const Ijk& x) const { return g_.nodeOwner(x); } //an element's owner is its lowest node's
  Index newNumber(const Ijk& x) const
  {
    const int r = owner(x);
    int lo[3], hi[3];
    box(r, lo, hi);
    return offsets_[r] + CartesianGrid::boxIndex(x, lo, hi, cells_ ? g_.cellOrder : g_.nodeOrder);
  }
  Index offset(int r) const { return offsets_[r]; }
private:
  const CartesianGrid& g_;
  bool cells_;
  std::vector<Index> offsets_;
};

/* Element boundaries or edges: the types, sorted as their tuples sort. */
class EntityFamily
{
public:
  EntityFamily(const CartesianGrid& g, std::vector<std::vector<Ijk> > types) : g_(g)
  {
    //sort the types by the sorted old-number offsets of their nodes, which
    //is how entities with the same origin are ordered in the global map
    std::vector<std::pair<std::vector<Index>,std::vector<Ijk> > > keyed;
    for (size_t t = 0; t < types.size(); t++)
      {
        std::vector<Index> key;
        for (const Ijk& o : types[t])
          key.push_back(o[0]*g.nodeStride[0] + o[1]*g.nodeStride[1] + o[2]*g.nodeStride[2]);
        std::sort(key.begin(), key.end());
        keyed.push_back(std::make_pair(key, types[t]));
      }
    std::sort(keyed.begin(), keyed.end());
    for (auto& k : keyed)
      {
        EntityType et;
        et.offsets = k.second;
        for (int d = 0; d < 3; d++)
          {
            et.lo[d] = 0;
            et.hi[d] = 0;
            for (const Ijk& o : et.offsets)
              {
                et.lo[d] = std::min(et.lo[d], o[d]);
                et.hi[d] = std::max(et.hi[d], o[d]);
              }
          }
        types_.push_back(et);
      }
  }
  int nTypes() const { return int(types_.size()); }
  const EntityType& type(int t) const { return types_[t]; }
  //the type of the entity with these nodes, and its origin
  void identify(const std::vector<Ijk>& nodes, int& t, Ijk& origin) const
  {
    origin = nodes[0];
    for (const Ijk& n : nodes)
      if (g_.nodeOld(n) < g_.nodeOld(origin))
        origin = n;
    std::vector<Ijk> offsets;
    for (const Ijk& n : nodes)
      offsets.push_back(Ijk{n[0]-origin[0], n[1]-origin[1], n[2]-origin[2]});
    std::sort(offsets.begin(), offsets.end());
    for (t = 0; t < nTypes(); t++)
      {
        std::vector<Ijk> o = types_[t].offsets;
        std::sort(o.begin(), o.end());
        if (o == offsets)
          return;
      }
    std::cerr<<"generateCartesianSubdomain: an entity matches no type"<<std::endl;
    std::abort();
  }
  bool exists(int t, const Ijk& origin) const
  {
    for (int d = 0; d < 3; d++)
      if (origin[d] + types_[t].lo[d] < 0 || origin[d] + types_[t].hi[d] >= g_.nn[d])
        return false;
    return true;
  }
  /* The old number: how many entities have a smaller sorted tuple. Those are
     the ones whose origin has a smaller old number, plus the earlier types at
     this origin. The origins with smaller old numbers form at most three
     boxes, and over a box the count of one type factors by axis. */
  Index oldNumber(int t, const Ijk& origin) const
  {
    Index count = countBefore(origin);
    for (int tt = 0; tt < t; tt++)
      if (exists(tt, origin))
        count++;
    return count;
  }
  Index total() const
  {
    int lo[3] = {0,0,0}, hi[3] = {g_.nn[0], g_.nn[1], g_.nn[2]};
    return countInBox(lo, hi);
  }
private:
  Index countInBox(const int lo[3], const int hi[3]) const
  {
    Index count = 0;
    for (const EntityType& et : types_)
      {
        Index c = 1;
        for (int d = 0; d < 3; d++)
          {
            const int first = std::max(lo[d], -et.lo[d]), last = std::min(hi[d], g_.nn[d] - et.hi[d]);
            c *= std::max(0, last - first);
          }
        count += c;
      }
    return count;
  }
  Index countBefore(const Ijk& a) const
  {
    Index count = 0;
    for (int level = 0; level < 3; level++)
      {
        int lo[3], hi[3];
        for (int o = 0; o < 3; o++)
          {
            const int d = g_.nodeOrder[o];
            if (o < level)
              {
                lo[d] = a[d];
                hi[d] = a[d]+1;
              }
            else if (o == level)
              {
                lo[d] = 0;
                hi[d] = a[d];
              }
            else
              {
                lo[d] = 0;
                hi[d] = g_.nn[d];
              }
          }
        count += countInBox(lo, hi);
      }
    return count;
  }
  const CartesianGrid& g_;
  std::vector<EntityType> types_;
};

/* The local element boundaries of an element, as the constructors number them. */
std::vector<std::vector<int> > elementBoundarySlots(int nNodes_element)
{
  if (nNodes_element == 4)
    return {{0,1},{1,2},{2,3},{3,0}};
  return {{0,1,2,3},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7},{4,5,6,7}};
}

/* The node pairs of an element the constructors call edges. */
std::vector<std::vector<int> > elementEdges(int nNodes_element)
{
  if (nNodes_element == 4)
    return {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
  return {{0,1},{1,2},{2,3},{3,0},{0,4},{1,5},{2,6},{3,7},{4,5},{5,6},{6,7},{7,4}};
}

std::vector<std::vector<Ijk> > typesFromElement(const CartesianGrid& g, const std::vector<std::vector<int> >& local)
{
  //every distinct offset pattern among an element's entities
  std::vector<std::vector<Ijk> > types;
  for (const std::vector<int>& e : local)
    {
      std::vector<Ijk> nodes;
      for (int nN : e)
        nodes.push_back(g.elementNodes[nN]);
      Ijk origin = nodes[0];
      for (const Ijk& n : nodes)
        if (g.nodeOld(n) < g.nodeOld(origin))
          origin = n;
      std::vector<Ijk> offsets;
      for (const Ijk& n : nodes)
        offsets.push_back(Ijk{n[0]-origin[0], n[1]-origin[1], n[2]-origin[2]});
      std::sort(offsets.begin(), offsets.end());
      if (std::find(types.begin(), types.end(), offsets) == types.end())
        types.push_back(offsets);
    }
  return types;
}

/* What partitionNodes does with an element's local element boundary: the
   nodes from which it reaches it. */
std::vector<int> selectingNodes(int nNodes_element, int slot)
{
  const std::vector<std::vector<int> > slots = elementBoundarySlots(nNodes_element);
  if (nNodes_element == 8)
    return slots[slot];
  std::vector<int> nodes;
  for (int nN = 0; nN < 4; nN++)
    if (nN != slot)
      nodes.push_back(nN);
  return nodes;
}

struct Entity
{
  int type;
  Ijk origin;
  int owner;
  Index newNumber;
};

/* New numbers for entities that are counted, not arithmetic: each rank's owned
   ones in increasing old number, ranks in order; ghosts are asked of their
   owners. */
void numberCountedEntities(const MPI_Comm& comm, int rank, int size, std::map<Index,Entity>& entities,
                           std::vector<Index>& offsets)
{
  std::vector<Index> ownedOld;
  for (auto& e : entities)
    if (e.second.owner == rank)
      ownedOld.push_back(e.first);//the map is in increasing old number
  long long nOwned = ownedOld.size();
  std::vector<long long> counts(size);
  MPI_Allgather(&nOwned,1,MPI_LONG_LONG,counts.data(),1,MPI_LONG_LONG,comm);
  offsets.assign(size+1,0);
  for (int r = 0; r < size; r++)
    offsets[r+1] = offsets[r] + counts[r];
  for (size_t i = 0; i < ownedOld.size(); i++)
    entities[ownedOld[i]].newNumber = offsets[rank] + Index(i);
  //ask each owner for the new numbers of the ghosts it owns
  std::vector<std::vector<Index> > ask(size);
  for (auto& e : entities)
    if (e.second.owner != rank)
      ask[e.second.owner].push_back(e.first);
  std::vector<int> sendCounts(size), recvCounts(size), sendDispl(size+1,0), recvDispl(size+1,0);
  for (int r = 0; r < size; r++)
    sendCounts[r] = int(ask[r].size());
  MPI_Alltoall(sendCounts.data(),1,MPI_INT,recvCounts.data(),1,MPI_INT,comm);
  for (int r = 0; r < size; r++)
    {
      sendDispl[r+1] = sendDispl[r] + sendCounts[r];
      recvDispl[r+1] = recvDispl[r] + recvCounts[r];
    }
  std::vector<Index> sendOld(sendDispl[size]), recvOld(recvDispl[size]);
  for (int r = 0; r < size; r++)
    std::copy(ask[r].begin(), ask[r].end(), sendOld.begin() + sendDispl[r]);
  MPI_Alltoallv(sendOld.data(),sendCounts.data(),sendDispl.data(),MPI_LONG_LONG,
                recvOld.data(),recvCounts.data(),recvDispl.data(),MPI_LONG_LONG,comm);
  std::vector<Index> reply(recvOld.size());
  for (size_t i = 0; i < recvOld.size(); i++)
    {
      std::vector<Index>::iterator it = std::lower_bound(ownedOld.begin(), ownedOld.end(), recvOld[i]);
      if (it == ownedOld.end() || *it != recvOld[i])
        {
          std::cerr<<"generateCartesianSubdomain: rank "<<rank<<" was asked for an entity it does not own"<<std::endl;
          MPI_Abort(comm,1);
        }
      reply[i] = offsets[rank] + Index(it - ownedOld.begin());
    }
  std::vector<Index> answers(sendOld.size());
  MPI_Alltoallv(reply.data(),recvCounts.data(),recvDispl.data(),MPI_LONG_LONG,
                answers.data(),sendCounts.data(),sendDispl.data(),MPI_LONG_LONG,comm);
  for (size_t i = 0; i < sendOld.size(); i++)
    entities[sendOld[i]].newNumber = answers[i];
}

/* Owned entities in increasing new number, then ghosts in increasing new number. */
std::vector<Index> subdomainOrder(const std::map<Index,Entity>& entities, int rank)
{
  std::vector<std::pair<Index,Index> > owned, ghost;
  for (const auto& e : entities)
    (e.second.owner == rank ? owned : ghost).push_back(std::make_pair(e.second.newNumber, e.first));
  std::sort(owned.begin(), owned.end());
  std::sort(ghost.begin(), ghost.end());
  std::vector<Index> order;
  for (auto& o : owned)
    order.push_back(o.second);
  for (auto& o : ghost)
    order.push_back(o.second);
  return order;
}

int checkedInt(const MPI_Comm& comm, Index i)
{
  if (i > Index(2147483647))
    {
      std::cerr<<"generateCartesianSubdomain: "<<i<<" does not fit in a proteus int"<<std::endl;
      MPI_Abort(comm,1);
    }
  return int(i);
}

int nodeMaterial(const CartesianGrid& g, const Ijk& n)
{
  //regularMeshNodes2D and regularMeshNodes
  if (g.dim == 2)
    {
      if (n[1] == 0) return 1;
      if (n[1] == g.nn[1]-1) return 3;
      if (n[0] == 0) return 4;
      if (n[0] == g.nn[0]-1) return 2;
      return 0;
    }
  if (n[2] == 0) return 1;
  if (n[2] == g.nn[2]-1) return 4;
  if (n[1] == 0) return 2;
  if (n[1] == g.nn[1]-1) return 6;
  if (n[0] == 0) return 3;
  if (n[0] == g.nn[0]-1) return 5;
  return 0;
}

/* The global mesh's material for an element boundary: the generators' boundary
   flags on the boundary of the box, INTERIOR_ELEMENT_BOUNDARY_MATERIAL (0)
   inside -- including on the cut between owned and ghost elements. */
int elementBoundaryMaterial(const CartesianGrid& g, const EntityType& et, const Ijk& origin)
{
  int normal = -1;
  for (int d = 0; d < g.dim; d++)
    if (et.lo[d] == 0 && et.hi[d] == 0)
      normal = d;
  const bool atLow = origin[normal] == 0, atHigh = origin[normal] == g.nn[normal]-1;
  if (!atLow && !atHigh)
    return 0;
  if (g.dim == 2)
    {
      //regularRectangularToTriangularElementBoundaryMaterials
      if (normal == 1) return atLow ? 1 : 3;
      return atLow ? 4 : 2;
    }
  //regularHexahedralMeshElementBoundaryMaterials
  if (normal == 2) return atLow ? 1 : 4;
  if (normal == 1) return atLow ? 2 : 6;
  return atLow ? 3 : 5;
}
}//namespace

int cartesianGlobalCounts(int nNodes_element, const int* nn, int* counts)
{
  const int grid[3] = {1,1,1};
  const double L[3] = {1.0,1.0,1.0};
  CartesianGrid g(nNodes_element, nn, L, grid);
  EntityFamily faces(g, typesFromElement(g, elementBoundarySlots(nNodes_element)));
  EntityFamily edges(g, typesFromElement(g, elementEdges(nNodes_element)));
  counts[0] = int(Index(g.nn[0])*g.nn[1]*g.nn[2]);
  counts[1] = int(Index(g.nc[0])*g.nc[1]*g.nc[2]);
  counts[2] = int(faces.total());
  counts[3] = int(edges.total());
  return 0;
}

int cartesianNumbering(int nNodes_element, const int* nn, const int* rankGrid, int elements,
                       int n, const int* oldNumbers, int* newNumbers)
{
  const double L[3] = {1.0,1.0,1.0};
  CartesianGrid g(nNodes_element, nn, L, rankGrid);
  BrickNumbering numbering(g, elements != 0);
  for (int i = 0; i < n; i++)
    {
      const Index old = oldNumbers[i];
      Ijk x;
      if (elements)
        {
          x[0] = int(old % g.nc[0]);
          x[1] = int((old / g.nc[0]) % g.nc[1]);
          x[2] = int(old / (Index(g.nc[0])*g.nc[1]));
        }
      else if (g.dim == 2)
        {
          x[0] = int(old / g.nn[1]);
          x[1] = int(old % g.nn[1]);
          x[2] = 0;
        }
      else
        {
          x[0] = int(old % g.nn[0]);
          x[1] = int((old / g.nn[0]) % g.nn[1]);
          x[2] = int(old / (Index(g.nn[0])*g.nn[1]));
        }
      newNumbers[i] = int(numbering.newNumber(x));
    }
  return 0;
}

int generateCartesianSubdomain(const MPI_Comm& PROTEUS_COMM_WORLD, int nNodes_element, const int* nn,
                               const double* L, const int* rankGrid, int nLayersOfOverlap, Mesh& mesh)
{
  using namespace std;
  int size, rank;
  MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);
  if (nNodes_element != 4 && nNodes_element != 8)
    {
      cerr<<"generateCartesianSubdomain: quadrilaterals (4 nodes) or hexahedra (8), not "<<nNodes_element<<endl;
      MPI_Abort(PROTEUS_COMM_WORLD,1);
    }
  if (rankGrid[0]*rankGrid[1]*rankGrid[2] != size || (nNodes_element == 4 && rankGrid[2] != 1))
    {
      cerr<<"generateCartesianSubdomain: rank grid "<<rankGrid[0]<<"x"<<rankGrid[1]<<"x"<<rankGrid[2]
          <<" for "<<size<<" ranks"<<endl;
      MPI_Abort(PROTEUS_COMM_WORLD,1);
    }
  if (nLayersOfOverlap != 1)
    {
      cerr<<"generateCartesianSubdomain: only one layer of overlap so far, not "<<nLayersOfOverlap<<endl;
      MPI_Abort(PROTEUS_COMM_WORLD,1);
    }
  CartesianGrid g(nNodes_element, nn, L, rankGrid);
  for (int d = 0; d < g.dim; d++)
    if (g.nn[d] < 2)
      {
        cerr<<"generateCartesianSubdomain: need at least two nodes along each axis"<<endl;
        MPI_Abort(PROTEUS_COMM_WORLD,1);
      }
  const vector<vector<int> > slots = elementBoundarySlots(nNodes_element), pairs = elementEdges(nNodes_element);
  const int nElementBoundaries_element = int(slots.size()), nNodes_elementBoundary = int(slots[0].size());
  BrickNumbering nodes(g, false), cells(g, true);
  EntityFamily faceFamily(g, typesFromElement(g, slots)), edgeFamily(g, typesFromElement(g, pairs));
  StageTimer timer(PROTEUS_COMM_WORLD,"generateCartesianSubdomain");

  //this rank's brick, the elements with a node in it, and those elements' nodes
  int lo[3], hi[3], cellLo[3], cellHi[3], nodeLo[3], nodeHi[3];
  g.brick(rank, lo, hi);
  bool empty = CartesianGrid::boxSize(lo, hi) == 0;
  for (int d = 0; d < 3; d++)
    {
      cellLo[d] = d < g.dim ? max(lo[d]-1, 0) : 0;
      cellHi[d] = d < g.dim ? min(hi[d], g.nc[d]) : 1;
      if (empty || cellHi[d] <= cellLo[d])
        empty = true;
    }
  for (int d = 0; d < 3; d++)
    {
      if (empty)
        cellLo[d] = cellHi[d] = 0;
      nodeLo[d] = cellLo[d];
      nodeHi[d] = empty ? cellLo[d] : (d < g.dim ? cellHi[d]+1 : 1);
    }
  auto forBox = [](const int blo[3], const int bhi[3], const int order[3], auto f)
    {
      Ijk x;
      const int a = order[0], b = order[1], c = order[2];
      for (x[a] = blo[a]; x[a] < bhi[a]; x[a]++)
        for (x[b] = blo[b]; x[b] < bhi[b]; x[b]++)
          for (x[c] = blo[c]; x[c] < bhi[c]; x[c]++)
            f(x);
    };
  auto inBox = [](const Ijk& x, const int blo[3], const int bhi[3])
    {
      for (int d = 0; d < 3; d++)
        if (x[d] < blo[d] || x[d] >= bhi[d])
          return false;
      return true;
    };

  //nodes: the brick in old order, then ghosts in new order
  vector<Ijk> sdNodes;
  forBox(lo, hi, g.nodeOrder, [&](const Ijk& x) { if (!empty) sdNodes.push_back(x); });
  const int nNodes_owned = int(sdNodes.size());
  {
    vector<pair<Index,Ijk> > ghosts;
    forBox(nodeLo, nodeHi, g.nodeOrder, [&](const Ijk& x) {
        if (!inBox(x, lo, hi))
          ghosts.push_back(make_pair(nodes.newNumber(x), x));
      });
    sort(ghosts.begin(), ghosts.end());
    for (auto& gh : ghosts)
      sdNodes.push_back(gh.second);
  }
  unordered_map<Index,int> nodeLocal;
  for (size_t i = 0; i < sdNodes.size(); i++)
    nodeLocal[g.nodeOld(sdNodes[i])] = int(i);

  //elements: the owned ones in old order, then ghosts in new order
  int ownLo[3], ownHi[3];
  g.cellBrick(rank, ownLo, ownHi);
  vector<Ijk> sdCells;
  forBox(ownLo, ownHi, g.cellOrder, [&](const Ijk& x) { if (!empty) sdCells.push_back(x); });
  const int nElements_owned = int(sdCells.size());
  {
    vector<pair<Index,Ijk> > ghosts;
    forBox(cellLo, cellHi, g.cellOrder, [&](const Ijk& x) {
        if (!inBox(x, ownLo, ownHi))
          ghosts.push_back(make_pair(cells.newNumber(x), x));
      });
    sort(ghosts.begin(), ghosts.end());
    for (auto& gh : ghosts)
      sdCells.push_back(gh.second);
  }
  timer.mark("nodes_and_elements");

  //element boundaries and edges of the subdomain's elements, keyed by old number
  map<Index,Entity> faces, edges;
  vector<Index> elementFaceOld(sdCells.size()*nElementBoundaries_element);
  for (size_t eN = 0; eN < sdCells.size(); eN++)
    {
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
        {
          vector<Ijk> fnodes;
          for (int nN : slots[ebN])
            fnodes.push_back(g.elementNode(sdCells[eN], nN));
          Entity e;
          faceFamily.identify(fnodes, e.type, e.origin);
          const Index old = faceFamily.oldNumber(e.type, e.origin);
          elementFaceOld[eN*nElementBoundaries_element+ebN] = old;
          if (faces.count(old))
            continue;
          /* the lowest rank that reaches it: through each element on either
             side, from the nodes partitionNodes selects for that element's
             local element boundary */
          e.owner = size;
          int normal = -1;
          for (int d = 0; d < g.dim; d++)
            if (faceFamily.type(e.type).lo[d] == 0 && faceFamily.type(e.type).hi[d] == 0)
              normal = d;
          for (int side = 0; side < 2; side++)
            {
              Ijk c = e.origin;
              c[normal] -= side;
              if (!g.cellExists(c))
                continue;
              for (int s = 0; s < nElementBoundaries_element; s++)
                {
                  vector<Ijk> snodes;
                  for (int nN : slots[s])
                    snodes.push_back(g.elementNode(c, nN));
                  sort(snodes.begin(), snodes.end());
                  vector<Ijk> sorted = fnodes;
                  sort(sorted.begin(), sorted.end());
                  if (snodes != sorted)
                    continue;
                  for (int nN : selectingNodes(nNodes_element, s))
                    e.owner = min(e.owner, g.nodeOwner(g.elementNode(c, nN)));
                }
            }
          faces[old] = e;
        }
      for (const vector<int>& pr : pairs)
        {
          vector<Ijk> enodes = {g.elementNode(sdCells[eN], pr[0]), g.elementNode(sdCells[eN], pr[1])};
          Entity e;
          edgeFamily.identify(enodes, e.type, e.origin);
          const Index old = edgeFamily.oldNumber(e.type, e.origin);
          if (edges.count(old))
            continue;
          //the owner of the node with the lower new number
          const Index n0 = nodes.newNumber(enodes[0]), n1 = nodes.newNumber(enodes[1]);
          e.owner = nodes.owner(n0 < n1 ? enodes[0] : enodes[1]);
          edges[old] = e;
        }
    }
  timer.mark("enumerate_boundaries_and_edges");
  vector<Index> faceOffsets, edgeOffsets;
  numberCountedEntities(PROTEUS_COMM_WORLD, rank, size, faces, faceOffsets);
  numberCountedEntities(PROTEUS_COMM_WORLD, rank, size, edges, edgeOffsets);
  const vector<Index> faceOrder = subdomainOrder(faces, rank), edgeOrder = subdomainOrder(edges, rank);
  timer.mark("number_boundaries_and_edges");
  unordered_map<Index,int> faceLocal;
  for (size_t i = 0; i < faceOrder.size(); i++)
    faceLocal[faceOrder[i]] = int(i);

  //the subdomain mesh, set up as partitionNodes sets it up
  if (mesh.subdomainp == NULL)
    mesh.subdomainp = new Mesh();
  Mesh& sd = *mesh.subdomainp;
  sd.nElements_global = int(sdCells.size());
  sd.nNodes_global = int(sdNodes.size());
  sd.nElementBoundaries_global = int(faceOrder.size());
  sd.nEdges_global = int(edgeOrder.size());
  sd.nNodes_element = nNodes_element;
  sd.nNodes_elementBoundary = nNodes_elementBoundary;
  sd.nElementBoundaries_element = nElementBoundaries_element;
  sd.nodeArray = new double[sd.nNodes_global*3];
  sd.nodeMaterialTypes = new int[sd.nNodes_global];
  for (int nN = 0; nN < sd.nNodes_global; nN++)
    {
      //the generators' own expressions, so the coordinates agree to the bit
      for (int d = 0; d < 3; d++)
        sd.nodeArray[nN*3+d] = d < g.dim ? sdNodes[nN][d]*g.h[d] : 0.0;
      sd.nodeMaterialTypes[nN] = nodeMaterial(g, sdNodes[nN]);
    }
  sd.elementNodesArray = new int[sd.nElements_global*nNodes_element];
  sd.elementMaterialTypes = new int[sd.nElements_global];
  sd.elementBoundariesArray = new int[sd.nElements_global*nElementBoundaries_element];
  for (int eN = 0; eN < sd.nElements_global; eN++)
    {
      sd.elementMaterialTypes[eN] = 0;//DEFAULT_ELEMENT_MATERIAL
      for (int nN = 0; nN < nNodes_element; nN++)
        sd.elementNodesArray[eN*nNodes_element+nN] = nodeLocal.at(g.nodeOld(g.elementNode(sdCells[eN], nN)));
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
        sd.elementBoundariesArray[eN*nElementBoundaries_element+ebN] = faceLocal.at(elementFaceOld[eN*nElementBoundaries_element+ebN]);
    }
  sd.edgeNodesArray = new int[sd.nEdges_global*2];
  for (int i = 0; i < sd.nEdges_global; i++)
    {
      //the global edge's nodes in its own order: the lower old number first
      const Entity& e = edges.at(edgeOrder[i]);
      const Ijk& o = edgeFamily.type(e.type).offsets[0], &o1 = edgeFamily.type(e.type).offsets[1];
      const Ijk a{e.origin[0]+o[0], e.origin[1]+o[1], e.origin[2]+o[2]};
      const Ijk b{e.origin[0]+o1[0], e.origin[1]+o1[1], e.origin[2]+o1[2]};
      const bool aFirst = g.nodeOld(a) < g.nodeOld(b);
      sd.edgeNodesArray[2*i+0] = nodeLocal.at(g.nodeOld(aFirst ? a : b));
      sd.edgeNodesArray[2*i+1] = nodeLocal.at(g.nodeOld(aFirst ? b : a));
    }
  sd.px = mesh.px;
  sd.py = mesh.py;
  sd.pz = mesh.pz;
  if (nNodes_element == 4)
    {
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_quadrilateral(sd);
      allocateGeometricInfo_quadrilateral(sd);
      computeGeometricInfo_quadrilateral(sd);
    }
  else
    {
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_hexahedron(sd);
      allocateGeometricInfo_hexahedron(sd);
      computeGeometricInfo_hexahedron(sd);
    }
  //the global mesh's element boundary materials
  for (int eN = 0; eN < sd.nElements_global; eN++)
    for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
      {
        const Entity& e = faces.at(elementFaceOld[eN*nElementBoundaries_element+ebN]);
        sd.elementBoundaryMaterialTypes[sd.elementBoundariesArray[eN*nElementBoundaries_element+ebN]] =
          elementBoundaryMaterial(g, faceFamily.type(e.type), e.origin);
      }
  timer.mark("subdomain_mesh");

  //the global mesh stands in for itself only through its counts
  mesh.nNodes_global = checkedInt(PROTEUS_COMM_WORLD, Index(g.nn[0])*g.nn[1]*g.nn[2]);
  mesh.nElements_global = checkedInt(PROTEUS_COMM_WORLD, Index(g.nc[0])*g.nc[1]*g.nc[2]);
  mesh.nElementBoundaries_global = checkedInt(PROTEUS_COMM_WORLD, faceFamily.total());
  mesh.nEdges_global = checkedInt(PROTEUS_COMM_WORLD, edgeFamily.total());
  mesh.nNodes_element = nNodes_element;
  mesh.nNodes_elementBoundary = nNodes_elementBoundary;
  mesh.nElementBoundaries_element = nElementBoundaries_element;
  delete [] mesh.nodeOffsets_subdomain_owned;
  delete [] mesh.elementOffsets_subdomain_owned;
  delete [] mesh.elementBoundaryOffsets_subdomain_owned;
  delete [] mesh.edgeOffsets_subdomain_owned;
  mesh.nodeOffsets_subdomain_owned = new int[size+1];
  mesh.elementOffsets_subdomain_owned = new int[size+1];
  mesh.elementBoundaryOffsets_subdomain_owned = new int[size+1];
  mesh.edgeOffsets_subdomain_owned = new int[size+1];
  for (int r = 0; r <= size; r++)
    {
      mesh.nodeOffsets_subdomain_owned[r] = checkedInt(PROTEUS_COMM_WORLD, r < size ? nodes.offset(r) : mesh.nNodes_global);
      mesh.elementOffsets_subdomain_owned[r] = checkedInt(PROTEUS_COMM_WORLD, r < size ? cells.offset(r) : mesh.nElements_global);
      mesh.elementBoundaryOffsets_subdomain_owned[r] = checkedInt(PROTEUS_COMM_WORLD, faceOffsets[r]);
      mesh.edgeOffsets_subdomain_owned[r] = checkedInt(PROTEUS_COMM_WORLD, edgeOffsets[r]);
    }
  delete [] mesh.nodeNumbering_subdomain2global;
  delete [] mesh.elementNumbering_subdomain2global;
  delete [] mesh.elementBoundaryNumbering_subdomain2global;
  delete [] mesh.edgeNumbering_subdomain2global;
  mesh.nodeNumbering_subdomain2global = new int[sd.nNodes_global];
  mesh.elementNumbering_subdomain2global = new int[sd.nElements_global];
  mesh.elementBoundaryNumbering_subdomain2global = new int[sd.nElementBoundaries_global];
  mesh.edgeNumbering_subdomain2global = new int[sd.nEdges_global];
  for (int nN = 0; nN < sd.nNodes_global; nN++)
    mesh.nodeNumbering_subdomain2global[nN] = checkedInt(PROTEUS_COMM_WORLD, nodes.newNumber(sdNodes[nN]));
  for (int eN = 0; eN < sd.nElements_global; eN++)
    mesh.elementNumbering_subdomain2global[eN] = checkedInt(PROTEUS_COMM_WORLD, cells.newNumber(sdCells[eN]));
  for (int i = 0; i < sd.nElementBoundaries_global; i++)
    mesh.elementBoundaryNumbering_subdomain2global[i] = checkedInt(PROTEUS_COMM_WORLD, faces.at(faceOrder[i]).newNumber);
  for (int i = 0; i < sd.nEdges_global; i++)
    mesh.edgeNumbering_subdomain2global[i] = checkedInt(PROTEUS_COMM_WORLD, edges.at(edgeOrder[i]).newNumber);
  (void)nNodes_owned;
  (void)nElements_owned;
  timer.mark("maps");
  timer.report();
  return 0;
}
}//proteus
