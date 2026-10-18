#include "partitioning.h"
#include <algorithm>
#include <array>
#include <map>
#include <unordered_map>
#include <vector>

/* A regular grid generated directly in pieces: each rank builds its own
   subdomain mesh, with the numbering and the maps partitionNodes produces when
   it is given the global grid from one of the rectangular-grid generators
   (generateQuadrilateralMeshFromRectangularGrid,
   generateHexahedralMeshFromRectangularGrid,
   generateTriangularMeshFromRectangularGrid with any triangleFlag,
   generateTetrahedralMeshFromRectangularGrid) and MeshTools.cartesianNodeOwners
   for the same rank grid -- entity for entity, without the global mesh, a graph
   partitioner or PETSc. See CARTESIAN_PARTITIONER_CAMPAIGN.md, Phases 2 and 3.

   What has to be reproduced, and how:

   - Old numbers are the generators'. Nodes: quadrilaterals y-fastest
     (i*nny + j), the others x-fastest. Elements: the grid's cells x-fastest,
     and within a cell its elements in the generator's order (two triangles, six
     tetrahedra, one quadrilateral or hexahedron). Element boundaries and edges
     are numbered by the std::map/std::set over sorted node tuples that the
     constructElementBoundaryElementsArray_* functions build, so an entity's old
     number is the number of entities whose sorted tuple of old node numbers is
     lexicographically smaller. On a grid that is a count over boxes
     (EntityFamily::oldNumber). "Edges" are every node pair of an element for
     quadrilaterals, triangles and tetrahedra (so quadrilateral diagonals
     count), and the twelve cell edges for hexahedra. The union-jack triangles
     (triangleFlag 1) alternate their diagonal with the cell's parity, so the
     diagonals exist only at every other origin; the count allows for that.
   - Owners are partitionNodes's: a node's owner is its brick; an element or
     element boundary belongs to the lowest rank that reaches it from an owned
     node through the node's element star -- an element through any of its
     nodes, an element boundary through the nodes partitionNodes selects (the
     face's own nodes for hexahedra, every node of the element but the one with
     the face's local number otherwise, which for a simplex is the face's own
     nodes again) -- and an edge to the owner of its node with the lower new
     number.
   - New numbers: each rank's owned entities in increasing old number, ranks in
     order. Nodes are bricks, so theirs are arithmetic, and so are elements'
     when every element contains its cell's lowest corner (quadrilaterals,
     hexahedra, tetrahedra, triangleFlag 0). Otherwise elements, and always
     element boundaries and edges, are counted per rank (MPI_Allgather) and the
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

enum Kind { QUAD = 0, HEX = 1, TRI = 2, TET = 3 };

/* A type of element boundary or edge: its node offsets from its node with the
   lowest old number (its origin). parity >= 0 means it exists only at origins
   with (i + j) % 2 == parity (union-jack diagonals). */
struct EntityType
{
  std::vector<Ijk> offsets;
  int lo[3], hi[3];
  int parity;
};

class CartesianGrid
{
public:
  int kind, triangleFlag, dim, size, nNodes_element, nSub;
  int nn[3], nc[3], p[3];
  std::vector<int> starts[3];
  Index nodeStride[3], cellStride[3];
  int nodeOrder[3], cellOrder[3]; //axes from slowest to fastest
  double h[3];
  //the elements of a cell, node offsets in element order: patterns[0] and,
  //for union-jack triangles, patterns[1] for cells with odd i + j
  std::vector<std::vector<Ijk> > patterns[2];
  bool cornerInEveryElement;

  CartesianGrid(int kindIn, int triangleFlagIn, const int* nnIn, const double* L, const int* rankGrid)
  {
    kind = kindIn;
    triangleFlag = triangleFlagIn;
    dim = (kind == HEX || kind == TET) ? 3 : 2;
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
    if (kind == QUAD)
      {
        //regularMeshNodes2D: y-fastest
        nodeStride[0] = nn[1]; nodeStride[1] = 1; nodeStride[2] = 0;
        nodeOrder[0] = 2; nodeOrder[1] = 0; nodeOrder[2] = 1;
      }
    else
      {
        nodeStride[0] = 1; nodeStride[1] = nn[0]; nodeStride[2] = Index(nn[0])*nn[1];
        nodeOrder[0] = 2; nodeOrder[1] = 1; nodeOrder[2] = 0;
      }
    cellStride[0] = 1; cellStride[1] = nc[0]; cellStride[2] = Index(nc[0])*nc[1];
    cellOrder[0] = 2; cellOrder[1] = 1; cellOrder[2] = 0;
    if (kind == QUAD) //regularQuadrilateralMeshElements
      patterns[0] = {{{0,0,0},{0,1,0},{1,1,0},{1,0,0}}};
    else if (kind == HEX) //regularHexahedralMeshElements
      patterns[0] = {{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}}};
    else if (kind == TRI) //regularRectangularToTriangularMeshElements
      {
        const Ijk n0{0,0,0}, n1{1,0,0}, n2{0,1,0}, n3{1,1,0};
        const std::vector<std::vector<Ijk> > right = {{n0,n1,n3},{n0,n3,n2}}, left = {{n0,n1,n2},{n2,n1,n3}};
        patterns[0] = triangleFlag == 2 ? left : right;
        patterns[1] = triangleFlag == 1 ? left : patterns[0];
      }
    else //regularHexahedralToTetrahedralMeshElements
      {
        const Ijk n0{0,0,0}, n1{1,0,0}, n2{0,1,0}, n3{1,1,0}, n4{0,0,1}, n5{1,0,1}, n6{0,1,1}, n7{1,1,1};
        patterns[0] = {{n0,n1,n3,n7},{n0,n1,n5,n7},{n0,n2,n3,n7},{n0,n2,n6,n7},{n0,n4,n5,n7},{n0,n4,n6,n7}};
      }
    if (patterns[1].empty())
      patterns[1] = patterns[0];
    nSub = int(patterns[0].size());
    nNodes_element = int(patterns[0][0].size());
    cornerInEveryElement = true;
    for (int q = 0; q < 2; q++)
      for (const std::vector<Ijk>& e : patterns[q])
        if (std::find(e.begin(), e.end(), Ijk{0,0,0}) == e.end())
          cornerInEveryElement = false;
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
  //the cells whose lowest node is in r's brick
  void cellBrick(int r, int lo[3], int hi[3]) const
  {
    brick(r, lo, hi);
    for (int d = 0; d < 3; d++)
      hi[d] = std::max(lo[d], std::min(hi[d], nc[d]));
  }
  //the cells with a node in r's brick
  void touchingCells(int r, int lo[3], int hi[3]) const
  {
    int blo[3], bhi[3];
    brick(r, blo, bhi);
    bool empty = false;
    for (int d = 0; d < 3; d++)
      {
        lo[d] = d < dim ? std::max(blo[d]-1, 0) : 0;
        hi[d] = d < dim ? std::min(bhi[d], nc[d]) : 1;
        if (bhi[d] <= blo[d] || hi[d] <= lo[d])
          empty = true;
      }
    if (empty)
      for (int d = 0; d < 3; d++)
        lo[d] = hi[d] = 0;
  }
  const std::vector<std::vector<Ijk> >& cellElements(const Ijk& c) const
  {
    return patterns[(c[0] + c[1]) % 2];
  }
  std::vector<Ijk> elementNodes(const Ijk& c, int sub) const
  {
    std::vector<Ijk> nodes;
    for (const Ijk& o : cellElements(c)[sub])
      nodes.push_back(Ijk{c[0]+o[0], c[1]+o[1], c[2]+o[2]});
    return nodes;
  }
  int elementOwner(const Ijk& c, int sub) const
  {
    int owner = size;
    for (const Ijk& n : elementNodes(c, sub))
      owner = std::min(owner, nodeOwner(n));
    return owner;
  }
  Index nodeOld(const Ijk& n) const { return n[0]*nodeStride[0] + n[1]*nodeStride[1] + n[2]*nodeStride[2]; }
  Index cellOld(const Ijk& c) const { return c[0]*cellStride[0] + c[1]*cellStride[1] + c[2]*cellStride[2]; }
  Index elementOld(const Ijk& c, int sub) const { return cellOld(c)*nSub + sub; }
  void elementFromOld(Index old, Ijk& c, int& sub) const
  {
    sub = int(old % nSub);
    const Index cell = old / nSub;
    c[0] = int(cell % nc[0]);
    c[1] = int((cell / nc[0]) % nc[1]);
    c[2] = int(cell / (Index(nc[0])*nc[1]));
  }
  Ijk nodeFromOld(Index old) const
  {
    if (kind == QUAD)
      return Ijk{int(old / nn[1]), int(old % nn[1]), 0};
    return Ijk{int(old % nn[0]), int((old / nn[0]) % nn[1]), int(old / (Index(nn[0])*nn[1]))};
  }
  bool cellExists(const Ijk& c) const
  {
    for (int d = 0; d < 3; d++)
      if (c[d] < 0 || c[d] >= nc[d])
        return false;
    return true;
  }
  static bool inBox(const Ijk& x, const int lo[3], const int hi[3])
  {
    for (int d = 0; d < 3; d++)
      if (x[d] < lo[d] || x[d] >= hi[d])
        return false;
    return true;
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
  template <class F>
  static void forBox(const int lo[3], const int hi[3], const int order[3], F f)
  {
    Ijk x;
    const int a = order[0], b = order[1], c = order[2];
    for (x[a] = lo[a]; x[a] < hi[a]; x[a]++)
      for (x[b] = lo[b]; x[b] < hi[b]; x[b]++)
        for (x[c] = lo[c]; x[c] < hi[c]; x[c]++)
          f(x);
  }
  //the local element boundaries of an element, as the constructors number them
  std::vector<std::vector<int> > boundarySlots() const
  {
    if (kind == QUAD)
      return {{0,1},{1,2},{2,3},{3,0}};
    if (kind == HEX)
      return {{0,1,2,3},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7},{4,5,6,7}};
    if (kind == TRI)
      return {{1,2},{2,0},{0,1}};
    return {{1,2,3},{2,3,0},{3,0,1},{0,1,2}};
  }
  //the node pairs of an element the constructors call edges
  std::vector<std::vector<int> > edgePairs() const
  {
    if (kind == HEX)
      return {{0,1},{1,2},{2,3},{3,0},{0,4},{1,5},{2,6},{3,7},{4,5},{5,6},{6,7},{7,4}};
    std::vector<std::vector<int> > pairs;
    for (int a = 0; a < nNodes_element; a++)
      for (int b = a+1; b < nNodes_element; b++)
        pairs.push_back({a,b});
    return pairs;
  }
  //the nodes from which partitionNodes reaches local element boundary s
  std::vector<int> selectingNodes(int s) const
  {
    if (kind == HEX)
      return boundarySlots()[s];
    std::vector<int> nodes;
    for (int nN = 0; nN < nNodes_element; nN++)
      if (nN != s)
        nodes.push_back(nN);
    return nodes;
  }
};

/* Node numbering, and element numbering when every element contains its cell's
   lowest corner: bricks of nodes, or of cells, so arithmetic. */
class BrickNumbering
{
public:
  BrickNumbering(const CartesianGrid& g, bool cells) : g_(g), cells_(cells), offsets_(g.size+1)
  {
    const Index perCell = cells ? g.nSub : 1;
    offsets_[0] = 0;
    for (int r = 0; r < g.size; r++)
      {
        int lo[3], hi[3];
        box(r, lo, hi);
        offsets_[r+1] = offsets_[r] + perCell*CartesianGrid::boxSize(lo, hi);
      }
  }
  void box(int r, int lo[3], int hi[3]) const
  {
    if (cells_)
      g_.cellBrick(r, lo, hi);
    else
      g_.brick(r, lo, hi);
  }
  Index node(const Ijk& x) const
  {
    const int r = g_.nodeOwner(x);
    int lo[3], hi[3];
    box(r, lo, hi);
    return offsets_[r] + CartesianGrid::boxIndex(x, lo, hi, g_.nodeOrder);
  }
  Index element(const Ijk& c, int sub) const
  {
    const int r = g_.nodeOwner(c);
    int lo[3], hi[3];
    box(r, lo, hi);
    return offsets_[r] + CartesianGrid::boxIndex(c, lo, hi, g_.cellOrder)*g_.nSub + sub;
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
  //nodes(c, sub) gives, for every element of every cell pattern, its entities' local node lists
  EntityFamily(const CartesianGrid& g, const std::vector<std::vector<int> >& local) : g_(g)
  {
    //every distinct offset pattern, and the parities of the cells it occurs in
    std::vector<std::vector<Ijk> > found;
    std::vector<std::vector<std::pair<int,Ijk> > > occurrences;
    for (int q = 0; q < 2; q++)
      for (const std::vector<Ijk>& element : g.patterns[q])
        for (const std::vector<int>& e : local)
          {
            std::vector<Ijk> nodes;
            for (int nN : e)
              nodes.push_back(element[nN]);
            Ijk origin;
            std::vector<Ijk> offsets = offsetsOf(nodes, origin);
            size_t t = std::find(found.begin(), found.end(), offsets) - found.begin();
            if (t == found.size())
              {
                found.push_back(offsets);
                occurrences.push_back(std::vector<std::pair<int,Ijk> >());
              }
            occurrences[t].push_back(std::make_pair(q, origin));
          }
    std::vector<std::pair<std::vector<Index>,EntityType> > keyed;
    for (size_t t = 0; t < found.size(); t++)
      {
        EntityType et;
        et.offsets = found[t];
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
        /* With two cell patterns (union jack), a type found in only one of them
           exists only where a cell of that parity puts it: at origins of parity
           (cell parity + offset of the origin in the cell). A type found in
           both exists wherever its nodes do, as with one pattern. */
        et.parity = -1;
        if (g.kind == TRI && g.triangleFlag == 1)
          {
            bool inPattern[2] = {false,false};
            int parity = -1;
            for (auto& oc : occurrences[t])
              {
                inPattern[oc.first] = true;
                const int pt = (oc.first + oc.second[0] + oc.second[1]) % 2;
                if (parity >= 0 && pt != parity)
                  parity = -2;
                else if (parity != -2)
                  parity = pt;
              }
            if (!(inPattern[0] && inPattern[1]))
              {
                if (parity < 0)
                  {
                    std::cerr<<"generateCartesianSubdomain: an entity type's parity is ambiguous"<<std::endl;
                    std::abort();
                  }
                et.parity = parity;
              }
          }
        std::vector<Index> key;
        for (const Ijk& o : et.offsets)
          key.push_back(o[0]*g.nodeStride[0] + o[1]*g.nodeStride[1] + o[2]*g.nodeStride[2]);
        std::sort(key.begin(), key.end());
        keyed.push_back(std::make_pair(key, et));
      }
    std::sort(keyed.begin(), keyed.end(),
              [](const std::pair<std::vector<Index>,EntityType>& a, const std::pair<std::vector<Index>,EntityType>& b)
              { return a.first < b.first; });
    for (auto& k : keyed)
      types_.push_back(k.second);
  }
  int nTypes() const { return int(types_.size()); }
  const EntityType& type(int t) const { return types_[t]; }
  //the type of the entity with these nodes, and its origin
  void identify(const std::vector<Ijk>& nodes, int& t, Ijk& origin) const
  {
    std::vector<Ijk> offsets = offsetsOf(nodes, origin);
    for (t = 0; t < nTypes(); t++)
      if (types_[t].offsets == offsets)
        return;
    std::cerr<<"generateCartesianSubdomain: an entity matches no type"<<std::endl;
    std::abort();
  }
  bool exists(int t, const Ijk& origin) const
  {
    for (int d = 0; d < 3; d++)
      if (origin[d] + types_[t].lo[d] < 0 || origin[d] + types_[t].hi[d] >= g_.nn[d])
        return false;
    return types_[t].parity < 0 || (origin[0] + origin[1]) % 2 == types_[t].parity;
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
  std::vector<Ijk> offsetsOf(const std::vector<Ijk>& nodes, Ijk& origin) const
  {
    origin = nodes[0];
    for (const Ijk& n : nodes)
      if (g_.nodeOld(n) < g_.nodeOld(origin))
        origin = n;
    std::vector<Ijk> offsets;
    for (const Ijk& n : nodes)
      offsets.push_back(Ijk{n[0]-origin[0], n[1]-origin[1], n[2]-origin[2]});
    std::sort(offsets.begin(), offsets.end());
    return offsets;
  }
  Index countInBox(const int lo[3], const int hi[3]) const
  {
    Index count = 0;
    for (const EntityType& et : types_)
      {
        Index n[3], even[3], odd[3];
        for (int d = 0; d < 3; d++)
          {
            const int first = std::max(lo[d], -et.lo[d]), last = std::min(hi[d], g_.nn[d] - et.hi[d]);
            n[d] = std::max(0, last - first);
            //the even numbers in [first, last)
            even[d] = n[d] > 0 ? (last - first + (first % 2 == 0 ? 1 : 0))/2 : 0;
            odd[d] = n[d] - even[d];
          }
        if (et.parity < 0)
          count += n[0]*n[1]*n[2];
        else if (et.parity == 0)
          count += (even[0]*even[1] + odd[0]*odd[1])*n[2];
        else
          count += (even[0]*odd[1] + odd[0]*even[1])*n[2];
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

struct Entity
{
  int type;    //element boundaries and edges: the type; elements: the element in its cell
  Ijk origin;  //element boundaries and edges: the origin; elements: the cell
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

/* The elements rank r owns, by old number: they have a node in its brick. */
std::vector<Index> ownedElements(const CartesianGrid& g, int r)
{
  int lo[3], hi[3];
  g.touchingCells(r, lo, hi);
  std::vector<Index> owned;
  CartesianGrid::forBox(lo, hi, g.cellOrder, [&](const Ijk& c) {
      for (int sub = 0; sub < g.nSub; sub++)
        if (g.elementOwner(c, sub) == r)
          owned.push_back(g.elementOld(c, sub));
    });
  std::sort(owned.begin(), owned.end());
  return owned;
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
  //regularMeshNodes2D, regularRectangularToTriangularMeshNodes, regularMeshNodes
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
   inside -- including on the cut between owned and ghost elements. Only an
   element boundary in a coordinate plane can be on the boundary. */
int elementBoundaryMaterial(const CartesianGrid& g, const EntityType& et, const Ijk& origin)
{
  int normal = -1;
  for (int d = 0; d < g.dim; d++)
    if (et.lo[d] == 0 && et.hi[d] == 0)
      normal = d;
  if (normal < 0)
    return 0;
  const bool atLow = origin[normal] == 0, atHigh = origin[normal] == g.nn[normal]-1;
  if (!atLow && !atHigh)
    return 0;
  if (g.dim == 2)
    {
      //regularRectangularToTriangularElementBoundaryMaterials, for quadrilaterals too
      if (normal == 1) return atLow ? 1 : 3;
      return atLow ? 4 : 2;
    }
  //regularHexahedralMeshElementBoundaryMaterials, regularHexahedralToTetrahedralElementBoundaryMaterials
  if (normal == 2) return atLow ? 1 : 4;
  if (normal == 1) return atLow ? 2 : 6;
  return atLow ? 3 : 5;
}

/* The lowest rank that reaches an element boundary: through each element that
   has it, from the nodes partitionNodes selects for that element's local
   element boundary. For a simplex those are the boundary's own nodes; for
   quadrilaterals and hexahedra the elements on either side are found through
   the boundary's normal. */
int elementBoundaryOwner(const CartesianGrid& g, const EntityType& et, const Ijk& origin,
                         const std::vector<Ijk>& nodes)
{
  int owner = g.size;
  if (g.kind == TRI || g.kind == TET)
    {
      for (const Ijk& n : nodes)
        owner = std::min(owner, g.nodeOwner(n));
      return owner;
    }
  const std::vector<std::vector<int> > slots = g.boundarySlots();
  int normal = -1;
  for (int d = 0; d < g.dim; d++)
    if (et.lo[d] == 0 && et.hi[d] == 0)
      normal = d;
  std::vector<Ijk> sorted = nodes;
  std::sort(sorted.begin(), sorted.end());
  for (int side = 0; side < 2; side++)
    {
      Ijk c = origin;
      c[normal] -= side;
      if (!g.cellExists(c))
        continue;
      const std::vector<Ijk> cellNodes = g.elementNodes(c, 0);
      for (size_t s = 0; s < slots.size(); s++)
        {
          std::vector<Ijk> snodes;
          for (int nN : slots[s])
            snodes.push_back(cellNodes[nN]);
          std::sort(snodes.begin(), snodes.end());
          if (snodes != sorted)
            continue;
          for (int nN : g.selectingNodes(int(s)))
            owner = std::min(owner, g.nodeOwner(cellNodes[nN]));
        }
    }
  return owner;
}
}//namespace

int cartesianGlobalCounts(int kind, int triangleFlag, const int* nn, int* counts)
{
  const int grid[3] = {1,1,1};
  const double L[3] = {1.0,1.0,1.0};
  CartesianGrid g(kind, triangleFlag, nn, L, grid);
  EntityFamily faces(g, g.boundarySlots()), edges(g, g.edgePairs());
  counts[0] = int(Index(g.nn[0])*g.nn[1]*g.nn[2]);
  counts[1] = int(Index(g.nc[0])*g.nc[1]*g.nc[2]*g.nSub);
  counts[2] = int(faces.total());
  counts[3] = int(edges.total());
  return 0;
}

int cartesianNumbering(int kind, int triangleFlag, const int* nn, const int* rankGrid, int elements,
                       int n, const int* oldNumbers, int* newNumbers)
{
  const double L[3] = {1.0,1.0,1.0};
  CartesianGrid g(kind, triangleFlag, nn, L, rankGrid);
  if (!elements || g.cornerInEveryElement)
    {
      BrickNumbering numbering(g, elements != 0);
      for (int i = 0; i < n; i++)
        {
          if (elements)
            {
              Ijk c;
              int sub;
              g.elementFromOld(oldNumbers[i], c, sub);
              newNumbers[i] = int(numbering.element(c, sub));
            }
          else
            newNumbers[i] = int(numbering.node(g.nodeFromOld(oldNumbers[i])));
        }
      return 0;
    }
  /* Elements a rank owns need not fill a box of cells (a union-jack or
     left-leaning triangle may miss its cell's lowest corner), so they are
     counted: every rank's, which touches the whole grid. A utility for
     checking against partitionNodes's permutation, not for production runs. */
  std::vector<std::vector<Index> > owned(g.size);
  std::vector<Index> offsets(g.size+1,0);
  for (int r = 0; r < g.size; r++)
    {
      owned[r] = ownedElements(g, r);
      offsets[r+1] = offsets[r] + Index(owned[r].size());
    }
  for (int i = 0; i < n; i++)
    {
      Ijk c;
      int sub;
      g.elementFromOld(oldNumbers[i], c, sub);
      const int r = g.elementOwner(c, sub);
      newNumbers[i] = int(offsets[r] + (std::lower_bound(owned[r].begin(), owned[r].end(), Index(oldNumbers[i])) - owned[r].begin()));
    }
  return 0;
}

int generateCartesianSubdomain(const MPI_Comm& PROTEUS_COMM_WORLD, int kind, int triangleFlag, const int* nn,
                               const double* L, const int* rankGrid, int nLayersOfOverlap, Mesh& mesh)
{
  using namespace std;
  int size, rank;
  MPI_Comm_size(PROTEUS_COMM_WORLD,&size);
  MPI_Comm_rank(PROTEUS_COMM_WORLD,&rank);
  if (kind < QUAD || kind > TET)
    {
      cerr<<"generateCartesianSubdomain: kind "<<kind<<" is not quadrilateral, hexahedron, triangle or tetrahedron"<<endl;
      MPI_Abort(PROTEUS_COMM_WORLD,1);
    }
  const bool flat = kind == QUAD || kind == TRI;
  if (rankGrid[0]*rankGrid[1]*rankGrid[2] != size || (flat && rankGrid[2] != 1))
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
  CartesianGrid g(kind, triangleFlag, nn, L, rankGrid);
  for (int d = 0; d < g.dim; d++)
    if (g.nn[d] < 2)
      {
        cerr<<"generateCartesianSubdomain: need at least two nodes along each axis"<<endl;
        MPI_Abort(PROTEUS_COMM_WORLD,1);
      }
  const vector<vector<int> > slots = g.boundarySlots(), pairs = g.edgePairs();
  const int nElementBoundaries_element = int(slots.size()), nNodes_elementBoundary = int(slots[0].size());
  const int nNodes_element = g.nNodes_element;
  BrickNumbering nodes(g, false), cells(g, true);
  EntityFamily faceFamily(g, slots), edgeFamily(g, pairs);
  StageTimer timer(PROTEUS_COMM_WORLD,"generateCartesianSubdomain");

  //the elements with a node in this rank's brick
  int lo[3], hi[3], cellLo[3], cellHi[3];
  g.brick(rank, lo, hi);
  g.touchingCells(rank, cellLo, cellHi);
  map<Index,Entity> elements;
  CartesianGrid::forBox(cellLo, cellHi, g.cellOrder, [&](const Ijk& c) {
      for (int sub = 0; sub < g.nSub; sub++)
        {
          bool touches = false;
          for (const Ijk& n : g.elementNodes(c, sub))
            if (CartesianGrid::inBox(n, lo, hi))
              touches = true;
          if (!touches)
            continue;
          Entity e;
          e.type = sub;
          e.origin = c;
          e.owner = g.elementOwner(c, sub);
          e.newNumber = g.cornerInEveryElement ? cells.element(c, sub) : -1;
          elements[g.elementOld(c, sub)] = e;
        }
    });
  vector<Index> elementOffsets;
  if (g.cornerInEveryElement)
    {
      elementOffsets.resize(size+1);
      for (int r = 0; r <= size; r++)
        elementOffsets[r] = r < size ? cells.offset(r) : Index(g.nc[0])*g.nc[1]*g.nc[2]*g.nSub;
    }
  else
    numberCountedEntities(PROTEUS_COMM_WORLD, rank, size, elements, elementOffsets);
  const vector<Index> elementOrder = subdomainOrder(elements, rank);

  //their nodes: the brick in old order, then ghosts in new order
  vector<Ijk> sdNodes;
  if (!elements.empty())
    CartesianGrid::forBox(lo, hi, g.nodeOrder, [&](const Ijk& x) { sdNodes.push_back(x); });
  {
    map<Index,Ijk> ghosts;
    for (const auto& e : elements)
      for (const Ijk& n : g.elementNodes(e.second.origin, e.second.type))
        if (!CartesianGrid::inBox(n, lo, hi))
          ghosts[nodes.node(n)] = n;
    for (auto& gh : ghosts)
      sdNodes.push_back(gh.second);
  }
  unordered_map<Index,int> nodeLocal;
  for (size_t i = 0; i < sdNodes.size(); i++)
    nodeLocal[g.nodeOld(sdNodes[i])] = int(i);
  timer.mark("nodes_and_elements");

  //element boundaries and edges of the subdomain's elements, keyed by old number
  map<Index,Entity> faces, edges;
  vector<Index> elementFaceOld(elementOrder.size()*nElementBoundaries_element);
  for (size_t eN = 0; eN < elementOrder.size(); eN++)
    {
      const Entity& el = elements.at(elementOrder[eN]);
      const vector<Ijk> enodes = g.elementNodes(el.origin, el.type);
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
        {
          vector<Ijk> fnodes;
          for (int nN : slots[ebN])
            fnodes.push_back(enodes[nN]);
          Entity e;
          faceFamily.identify(fnodes, e.type, e.origin);
          const Index old = faceFamily.oldNumber(e.type, e.origin);
          elementFaceOld[eN*nElementBoundaries_element+ebN] = old;
          if (faces.count(old))
            continue;
          e.owner = elementBoundaryOwner(g, faceFamily.type(e.type), e.origin, fnodes);
          faces[old] = e;
        }
      for (const vector<int>& pr : pairs)
        {
          vector<Ijk> pnodes = {enodes[pr[0]], enodes[pr[1]]};
          Entity e;
          edgeFamily.identify(pnodes, e.type, e.origin);
          const Index old = edgeFamily.oldNumber(e.type, e.origin);
          if (edges.count(old))
            continue;
          //the owner of the node with the lower new number
          e.owner = g.nodeOwner(nodes.node(pnodes[0]) < nodes.node(pnodes[1]) ? pnodes[0] : pnodes[1]);
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
  sd.nElements_global = int(elementOrder.size());
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
      const Entity& el = elements.at(elementOrder[eN]);
      const vector<Ijk> enodes = g.elementNodes(el.origin, el.type);
      sd.elementMaterialTypes[eN] = 0;//DEFAULT_ELEMENT_MATERIAL
      for (int nN = 0; nN < nNodes_element; nN++)
        sd.elementNodesArray[eN*nNodes_element+nN] = nodeLocal.at(g.nodeOld(enodes[nN]));
      for (int ebN = 0; ebN < nElementBoundaries_element; ebN++)
        sd.elementBoundariesArray[eN*nElementBoundaries_element+ebN] = faceLocal.at(elementFaceOld[eN*nElementBoundaries_element+ebN]);
    }
  sd.edgeNodesArray = new int[sd.nEdges_global*2];
  for (int i = 0; i < sd.nEdges_global; i++)
    {
      //the global edge's nodes in its own order: the lower old number first
      const Entity& e = edges.at(edgeOrder[i]);
      const Ijk& o0 = edgeFamily.type(e.type).offsets[0], &o1 = edgeFamily.type(e.type).offsets[1];
      const Ijk a{e.origin[0]+o0[0], e.origin[1]+o0[1], e.origin[2]+o0[2]};
      const Ijk b{e.origin[0]+o1[0], e.origin[1]+o1[1], e.origin[2]+o1[2]};
      const bool aFirst = g.nodeOld(a) < g.nodeOld(b);
      sd.edgeNodesArray[2*i+0] = nodeLocal.at(g.nodeOld(aFirst ? a : b));
      sd.edgeNodesArray[2*i+1] = nodeLocal.at(g.nodeOld(aFirst ? b : a));
    }
  sd.px = mesh.px;
  sd.py = mesh.py;
  sd.pz = mesh.pz;
  if (kind == QUAD)
    {
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_quadrilateral(sd);
      allocateGeometricInfo_quadrilateral(sd);
      computeGeometricInfo_quadrilateral(sd);
    }
  else if (kind == HEX)
    {
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_hexahedron(sd);
      allocateGeometricInfo_hexahedron(sd);
      computeGeometricInfo_hexahedron(sd);
    }
  else if (kind == TRI)
    {
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_triangle(sd);
      allocateGeometricInfo_triangle(sd);
      computeGeometricInfo_triangle(sd);
    }
  else
    {
      constructElementBoundaryElementsArrayWithGivenElementBoundaryAndEdgeNumbers_tetrahedron(sd);
      allocateGeometricInfo_tetrahedron(sd);
      computeGeometricInfo_tetrahedron(sd);
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
  mesh.nElements_global = checkedInt(PROTEUS_COMM_WORLD, Index(g.nc[0])*g.nc[1]*g.nc[2]*g.nSub);
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
      mesh.elementOffsets_subdomain_owned[r] = checkedInt(PROTEUS_COMM_WORLD, elementOffsets[r]);
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
    mesh.nodeNumbering_subdomain2global[nN] = checkedInt(PROTEUS_COMM_WORLD, nodes.node(sdNodes[nN]));
  for (int eN = 0; eN < sd.nElements_global; eN++)
    mesh.elementNumbering_subdomain2global[eN] = checkedInt(PROTEUS_COMM_WORLD, elements.at(elementOrder[eN]).newNumber);
  for (int i = 0; i < sd.nElementBoundaries_global; i++)
    mesh.elementBoundaryNumbering_subdomain2global[i] = checkedInt(PROTEUS_COMM_WORLD, faces.at(faceOrder[i]).newNumber);
  for (int i = 0; i < sd.nEdges_global; i++)
    mesh.edgeNumbering_subdomain2global[i] = checkedInt(PROTEUS_COMM_WORLD, edges.at(edgeOrder[i]).newNumber);
  timer.mark("maps");
  timer.report();
  return 0;
}
}//proteus
