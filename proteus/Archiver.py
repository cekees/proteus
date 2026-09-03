"""
Classes for archiving numerical solution data

.. inheritance-diagram:: proteus.Archiver
   :parts: 1
"""
from . import Profiling
from .Profiling import logEvent
from . import Comm
import numpy
import os
import h5py
from xml.etree.ElementTree import *

memory = Profiling.memory

def indentXML(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indentXML(e, level+1)
            if not e.tail or not e.tail.strip():
                e.tail = i + "  "
        if not e.tail or not e.tail.strip():
            e.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

class ArchiveFlags(object):
    EVERY_MODEL_STEP     = 0
    EVERY_USER_STEP      = 1
    EVERY_SEQUENCE_STEP  = 2
    UNDEFINED            =-1
#
class AR_base(object):
    def __init__(self,dataDir,filename,
                 useTextArchive=False,
                 gatherAtClose=True,
                 useGlobalXMF=True,
                 hotStart=False,
                 readOnly=False,
                 global_sync=True):
        import os.path
        import copy
        self.useGlobalXMF=useGlobalXMF
        comm=Comm.get()
        self.comm=comm
        self.rank = comm.rank()
        self.size = comm.size()
        self.dataDir=dataDir
        self.filename=filename
        self.readOnly = readOnly
        self.n_datasets = 0
        self.archived_domain = None
        import datetime
        #filename += datetime.datetime.now().isoformat()
        self.global_sync = global_sync
        comm_world = self.comm.comm.tompi4py()
        self.xmlHeader = "<?xml version=\"1.0\" ?>\n<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
        if hotStart:
            if useGlobalXMF:
                xmlFile_old=open(os.path.join(self.dataDir,
                                              filename+".xmf"),
                                 "rb")
            else:
                xmlFile_old=open(os.path.join(self.dataDir,
                                              filename+str(self.rank)+".xmf"),
                                 "rb")
            self.tree=ElementTree(file=xmlFile_old)
            if self.comm.isMaster():
                self.xmlFileGlobal = open(os.path.join(self.dataDir,
                                                       filename+".xmf"),
                                          "ab")
                self.treeGlobal=copy.deepcopy(self.tree)
            if not useGlobalXMF:
                self.xmlFile=open(os.path.join(self.dataDir,
                                               filename+str(self.rank)+".xmf"),
                                  "ab")
            if not useTextArchive:
                self.hdfFilename=filename+".h5"
                self.hdfFile=h5py.File(os.path.join(self.dataDir,self.hdfFilename),
                                       "a",
                                       driver="mpio",
                                       comm = comm_world)
                self.dataItemFormat="HDF"
            else:
                self.textDataDir=filename+"_Data"
                if not os.path.exists(self.textDataDir):
                    try:
                        os.mkdir(self.textDataDir)
                    except:
                        self.textDataDir=""
                self.hdfFile=None
                self.dataItemFormat="XML"
        elif readOnly:
            if useGlobalXMF:
                self.xmlFile=open(os.path.join(self.dataDir,
                                               filename+".xmf"),
                                  "rb")
            else:
                self.xmlFile=open(os.path.join(self.dataDir,
                                               filename+str(self.rank)+".xmf"),
                                  "rb")
            self.tree=ElementTree(file=self.xmlFile)
            if not useTextArchive:
                self.hdfFilename=filename+".h5"
                self.hdfFile=h5py.File(os.path.join(self.dataDir,
                                                    self.hdfFilename),
                                       "r",
                                       driver = 'mpio',
                                       comm = comm_world)
                self.dataItemFormat="HDF"
            else:
                self.textDataDir=filename+"_Data"
                assert(os.path.exists(self.textDataDir))
                self.hdfFile=None
                self.dataItemFormat="XML"
        else:
            if not self.useGlobalXMF:
                self.xmlFile=open(os.path.join(self.dataDir,
                                               filename+str(self.rank)+".xmf"),
                                  "wb")
            self.tree=ElementTree(
                Element("Xdmf",
                        {"Version":"2.0",
                         "xmlns:xi":"http://www.w3.org/2001/XInclude"})
            )
            if self.comm.isMaster():
                self.xmlFileGlobal=open(
                    os.path.join(self.dataDir,
                                 filename+".xmf"),
                    "wb")
                self.treeGlobal=ElementTree(
                    Element("Xdmf",
                            {"Version":"2.0",
                             "xmlns:xi":"http://www.w3.org/2001/XInclude"})
                )
            if not useTextArchive:
                self.hdfFilename=filename+".h5"
                self.hdfFile=h5py.File(os.path.join(self.dataDir,
                                                    self.hdfFilename),
                                       "w",
                                       driver = 'mpio',
                                       comm = comm_world)
                self.dataItemFormat="HDF"
                self.comm.barrier()
            else:
                self.textDataDir=filename+"_Data"
                if not os.path.exists(self.textDataDir):
                    try:
                        os.mkdir(self.textDataDir)
                    except:
                        self.textDataDir=""
                self.hdfFile=None
                self.dataItemFormat="XML"
        #
        self.gatherAtClose = gatherAtClose
    #: Bumped when the in-archive metadata layout changes incompatibly.
    #: Written as an attribute on the HDF5 root by
    #: :meth:`allGatherIncremental` and checked by
    #: :meth:`gatherAndWriteTimes`, so reading an archive written by an
    #: incompatible proteus fails with a clear message rather than a
    #: parse error deep inside the reader.
    METADATA_FORMAT_VERSION = 2
    #: HDF5 root attribute holding :data:`METADATA_FORMAT_VERSION`.
    METADATA_VERSION_ATTR = "ymf_archive_metadata_version"
    #: HDF5 root attribute holding the time-collection names, newline
    #: separated, in the order they were written.
    COLLECTIONS_ATTR = "ymf_archive_collections"
    #: HDF5 root attribute recording whether the archive was written as
    #: global arrays (1) or one grid per subdomain (0). Recorded so a
    #: reader need not be told: at one rank the two layouts produce
    #: metadata of the same shape but different XDMF, so it cannot be
    #: inferred from the data.
    GLOBAL_SYNC_ATTR = "ymf_archive_global_sync"

    def _metadata_version(self):
        """Which metadata layout this archive uses.

        Version 1 is XDMF ``<Grid>`` fragments, written by proteus up to
        and including 1.9.x. Version 2 is YAML describing a ymf grid dict.
        Both live in datasets named ``<collection>_<step>``, and a version 1
        archive carries no version attribute at all -- its absence is the
        marker.

        Version 1 is **read**, not refused: an existing XDMF archive must
        stay hot-startable and stay usable with the scripts. Only version 2
        is written.
        """
        found = self.hdfFile.attrs.get(self.METADATA_VERSION_ATTR)
        if found is None:
            return 1
        version = int(found)
        if version > self.METADATA_FORMAT_VERSION:
            raise ValueError(
                "%s holds grid metadata in format version %d; this proteus "
                "reads up to version %d"
                % (self.hdfFilename, version, self.METADATA_FORMAT_VERSION))
        return version

    def _check_metadata_version(self):
        """Kept for callers that only want the refusal on a future version."""
        self._metadata_version()

    def _discover_metadata_datasets(self):
        """Group the metadata datasets by collection, ordered by step.

        Needed for a version 1 archive, which records no collection names.
        Rather than pattern-matching names -- a collection name can itself
        contain underscores and digits, as Mesh_c0p2_Lagrange does -- a
        candidate is confirmed by what it *is*: a one-dimensional array of
        byte strings. Field data is numeric, so nothing else in the file
        looks like this.

        Returns ``{collection_name: [dataset_name_by_step, ...]}``.
        """
        import re

        pattern = re.compile(r"^(.+)_(\d+)$")
        found = {}
        for key, value in self.hdfFile.items():
            if getattr(value, "ndim", None) != 1 or value.dtype.kind != "S":
                continue
            match = pattern.match(key)
            if match is None:
                continue
            found.setdefault(match.group(1), []).append(
                (int(match.group(2)), key))
        return {name: [key for _, key in sorted(steps)]
                for name, steps in found.items()}

    @property
    def archived_times(self):
        """The times recorded in the archive, in the order written.

        Reads the assembled ymf domain rather than the in-memory XML tree.
        Before Phase 2 of the YMF campaign, ``gatherAndWriteTimes`` filled
        ``self.treeGlobal`` from the HDF5 metadata and callers counted
        ``<Time>`` elements in it; the domain dict is the representation
        now, and that tree is on its way out.

        Empty until :meth:`gatherAndWriteTimes` has run, i.e. until the
        archive is closed.
        """
        domain = self.load_archived_domain() if self.hdfFile is not None \
            else getattr(self, "archived_domain", None)
        if domain is None:
            return []
        times = []
        for collection in domain.get("TimeCollections", []):
            for step in collection["Data"]:
                times.append(step["Time"])
            #every collection covers the same instants, so one is enough
            break
        return times

    def assemble_domain(self, n_steps=None):
        """Build the ymf domain for the whole archive from the HDF5 metadata.

        Used by both ends of the archive's life: :meth:`gatherAndWriteTimes`
        calls it at close to produce the document it writes, and readers
        call it to get the archive's structure without walking XML. That
        shared use is the point -- there is one definition of what the
        archive contains, rather than a writer's idea and a reader's idea
        that can drift.

        ``n_steps`` bounds the search when the caller knows it (the writer
        does, from ``self.n_datasets``). A reader does not, so it is
        discovered: steps are written consecutively from 0, so counting up
        until a step is missing finds them all.

        Returns ``None`` when there is no HDF5 metadata to assemble from,
        which is the text-archive case.
        """
        from ymf.archive import (add_collection, add_spatial_step,
                                 add_uniform_step, load_grid, new_domain)

        if self.hdfFile is None:
            return None
        version = self._metadata_version()

        raw = self.hdfFile.attrs.get(self.COLLECTIONS_ATTR)
        if raw is not None:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            collection_names = [name for name in raw.split("\n") if name]
            datasets = {name: None for name in collection_names}
        else:
            #A version 1 archive records no names; find them from the file.
            datasets = self._discover_metadata_datasets()
            collection_names = sorted(datasets)

        #Prefer the mode the archive recorded. A version 2 reader has no way
        #to infer it: at one rank a global write and a per-subdomain write
        #both leave metadata of shape (1,), but the first means a uniform
        #step and the second a spatial collection holding one grid. Version 1
        #records nothing, so there it *is* inferred -- see _v1_grids.
        recorded = self.hdfFile.attrs.get(self.GLOBAL_SYNC_ATTR)
        global_sync = self.global_sync if recorded is None else bool(int(recorded))

        domain = None
        for ci, collection_name in enumerate(collection_names):
            if domain is None:
                domain = new_domain(collection_name)
            else:
                add_collection(domain, collection_name)
            keys = datasets.get(collection_name)
            step = 0
            while n_steps is None or step < n_steps:
                if keys is not None:
                    if step >= len(keys):
                        break
                    dataset_name = keys[step]
                else:
                    dataset_name = self._metadata_dataset_name(
                        collection_name, step)
                    if dataset_name not in self.hdfFile:
                        if n_steps is None:
                            break      # discovered the end
                        step += 1
                        continue       # writer knows the count; tolerate a gap
                grid_array = self.hdfFile[dataset_name]
                if version == 1:
                    t, grids, step_is_global = self._v1_grids(grid_array)
                else:
                    t = float(grid_array.attrs['Time'])
                    grids = [load_grid(grid_array[j].decode("utf-8"))
                             for j in range(grid_array.shape[0])]
                    step_is_global = global_sync
                if step_is_global:
                    #one already-global grid: its pieces are the archive
                    g = grids[0]
                    add_uniform_step(domain, t, g["Topology"], g["Geometry"],
                                     g.get("Attributes", []), collection=ci)
                else:
                    #one grid per rank, presented as a single instant
                    add_spatial_step(domain, t, grids, collection=ci)
                step += 1
        return domain

    def _v1_grids(self, grid_array):
        """Read one step out of a version 1 (XDMF fragment) archive.

        Returns ``(time, grids, global_sync)``. Both the time and the
        layout are recovered from the fragments themselves, because
        proteus <= 1.9.x recorded neither reliably:

        * The dataset's ``Time`` attribute was set only on the
          per-subdomain path, so for a global archive the time has to come
          from the ``<Time>`` element inside the fragment.
        * That same asymmetry identifies the layout. The per-subdomain
          writer moved ``<Time>`` out of each grid before storing it, so a
          fragment that still has one was written globally. Shape alone is
          not enough -- both layouts give shape (1,) at a single rank.
        """
        from xml.etree.ElementTree import fromstring

        from ymf.xdmf import parse_grid_element

        elements = [fromstring(grid_array[j])
                    for j in range(grid_array.shape[0])]
        first_time = elements[0].find("Time")
        global_sync = first_time is not None
        if first_time is not None:
            t = float(first_time.attrib["Value"])
        else:
            t = float(grid_array.attrs["Time"])
        return t, [parse_grid_element(e) for e in elements], global_sync

    def load_archived_domain(self):
        """The archive's domain, assembling it from HDF5 on first use.

        For readers -- hot start, mesh readers -- which open an existing
        archive and need its structure. Cached, since assembling walks
        every step's metadata.
        """
        if getattr(self, "archived_domain", None) is None:
            self.archived_domain = self.assemble_domain()
        return self.archived_domain

    def field_dataset(self, name, tCount):
        """The HDF5 dataset holding one field at one step, for reading back.

        Hot start supports exactly the two modes the archive can write:

        * **global** -- one assembled array per field, named
          ``<name>_t<step>``. Readable at any number of MPI tasks, since
          the array does not encode the decomposition.
        * **per-rank** -- one array per subdomain, named
          ``<name>_p<rank>_t<step>``. Readable only with the same number
          of tasks that wrote it, because rank *i* reads subdomain *i*.

        A third path used to exist: a separate ``<name>global.h5`` opened
        as ``hdfFileGlb`` and read with ``get_node``, which is a PyTables
        method that ``h5py.File`` does not have. It had been dead since the
        PyTables-to-h5py migration, and silently so -- the open was wrapped
        in a bare ``except: pass``, so it only surfaced when the file
        actually existed. Removed rather than repaired: a user writing
        either supported mode can hot start from it.

        Raises ``KeyError`` naming the mismatch rather than letting a
        missing dataset surface as a bare key error, since the usual cause
        is a per-rank archive being read at the wrong task count.
        """
        if self.global_sync:
            key = "{0:s}_t{1:d}".format(name, tCount)
        else:
            key = "{0:s}_p{1:s}_t{2:d}".format(
                name, repr(self.comm.rank()), tCount)
        try:
            return self.hdfFile["/" + key]
        except KeyError:
            pass
        if self.global_sync:
            raise KeyError(
                "%s has no dataset %r. This run is hot starting in global "
                "mode; the archive may have been written per-subdomain "
                "instead, in which case it must be read with global_sync "
                "off and the same number of MPI tasks."
                % (self.hdfFilename, key))
        import re
        pattern = re.compile(r"^%s_p(\d+)_t%d$" % (re.escape(name), tCount))
        written_by = sorted(int(m.group(1)) for m in
                            (pattern.match(k) for k in self.hdfFile) if m)
        raise KeyError(
            "%s has no dataset %r. A per-subdomain hot start needs the same "
            "number of MPI tasks that wrote the archive: this one holds %d "
            "subdomain(s) %s and this run has %d task(s). Either run with %d "
            "tasks or write the archive in global mode, which any task count "
            "can read."
            % (self.hdfFilename, key, len(written_by),
               written_by if len(written_by) < 8 else
               "0..%d" % (written_by[-1],),
               self.size, len(written_by) or self.size))

    def _metadata_dataset_name(self, collection_name, index):
        """Name of the HDF5 dataset holding one collection's step metadata."""
        return (collection_name + "_" + str(index)).replace(" ", "_")

    def gatherAndWriteTimes(self):
        """Assemble the whole archive document and write it out.

        Reads back the per-step, per-rank grid metadata that
        :meth:`allGatherIncremental` stashed in the HDF5 file as YAML,
        builds a single ymf domain from it, and writes that domain as both
        the ``.ymf`` archive of record and an ``.xmf`` for viewers.

        This is where the campaign's "one dict, three serializations" shape
        actually lands: the domain dict is assembled once here and both
        output files are derived from it, rather than the XML tree being
        the thing that gets written and the data model being an
        afterthought.
        """
        from ymf.archive import write_ymf
        from ymf.xdmf import build_xdmf_tree, XDMF_HEADER

        domain = self.assemble_domain()
        #keep the assembled domain: it is the archive's representation now,
        #and callers that used to inspect self.treeGlobal want this instead
        self.archived_domain = domain

        self.clear_xml()
        if domain is not None:
            ymf_path = os.path.join(self.dataDir, self.filename + ".ymf")
            write_ymf(domain, ymf_path)
            logEvent("Wrote YMF archive " + ymf_path)
            #The .xmf is derived from the same domain, for viewers, and is
            #written through the handle opened in __init__ -- the legacy
            #path wrote self.treeGlobal here, and since the grids no longer
            #accumulate in that tree it would truncate this file back to an
            #empty collection. Phase 4 replaces this with an on-demand
            #converter and drops the .xmf from the write path entirely.
            tree = build_xdmf_tree(domain)
            indentXML(tree.getroot())
            self.xmlFileGlobal.write(XDMF_HEADER)
            tree.write(self.xmlFileGlobal, encoding="utf-8")
        else:
            #no HDF5 metadata to assemble from (text-archive mode): fall
            #back to whatever the in-memory tree holds
            self.xmlFileGlobal.write(bytes(self.xmlHeader,"utf-8"))
            indentXML(self.treeGlobal.getroot())
            self.treeGlobal.write(self.xmlFileGlobal,encoding="utf-8")
    def clear_xml(self):
        if not self.useGlobalXMF:
            self.xmlFile.seek(0)
            self.xmlFile.truncate()
        if self.comm.isMaster():
            self.xmlFileGlobal.seek(0)
            self.xmlFileGlobal.truncate()
    def close(self):
        logEvent("Closing Archive")
        if not self.useGlobalXMF:
            self.xmlFile.close()
        if self.comm.isMaster() and self.useGlobalXMF:
            self.gatherAndWriteTimes()
            self.xmlFileGlobal.close()
        if self.hdfFile is not None:
            self.hdfFile.close()
        logEvent("Done Closing Archive")
        try:
            if not self.useGlobalXMF:
                if self.gatherAtClose:
                    self.allGather()
        except:
            pass
    def allGather(self):
        logEvent("Gathering Archive")
        self.comm.barrier()
        if self.rank==0:
            #replace the bottom level grid with a spatial collection
            XDMF_all=self.tree.getroot()
            Domain_all=XDMF_all[-1]
            for TemporalGridCollection in Domain_all:
                Grids = TemporalGridCollection[:]
                del TemporalGridCollection[:]
                for Grid in Grids:
                    SpatialCollection=SubElement(TemporalGridCollection,"Grid",{"GridType":"Collection",
                                                                                "CollectionType":"Spatial"})
                    SpatialCollection.append(Grid[0])#append Time in Spatial Collection
                    del Grid[0]#delete Time in grid
                    SpatialCollection.append(Grid) #append Grid without Time
            for i in range(1,self.size):
                xmlFile=open(os.path.join(self.dataDir,self.filename+str(i)+".xmf"),"rb")
                tree = ElementTree(file=xmlFile)
                XDMF=tree.getroot()
                Domain=XDMF[-1]
                for TemporalGridCollection,TemporalGridCollection_all in zip(Domain,Domain_all):
                    SpatialGridCollections = TemporalGridCollection_all.findall('Grid')
                    for Grid,Grid_all in zip(TemporalGridCollection,SpatialGridCollections):
                        del Grid[0]#Time
                        Grid_all.append(Grid)
                xmlFile.close()
            f = open(os.path.join(self.dataDir,self.filename+".xmf"),"wb")
            indentXML(self.tree.getroot())
            self.tree.write(f)
            f.close()
        logEvent("Done Gathering Archive")
    def allGatherIncremental(self):
        """Stash this timestep's grid metadata in the HDF5 file.

        Each rank turns its own ``<Grid>`` element into a ymf grid dict,
        serializes it to YAML, and the collection of them is written into a
        dataset named ``<collection>_<step>``.
        :meth:`gatherAndWriteTimes` reads them back at close and assembles
        the whole document.

        Three things changed here relative to the XML version:

        * The payload is YAML describing a ymf grid dict, not an XDMF
          fragment. Parsing an element into a dict at this boundary is a
          bridge -- the writers still build elements (Phase 1 of the
          campaign is unfinished), and when they build dicts directly this
          parse disappears.
        * Grids cross MPI as dicts rather than as pickled
          ``ElementTree.Element`` objects.
        * The collective agreement on the dataset width is an
          ``allreduce(MAX)`` over each rank's own encoded length, not a
          ``Bcast`` of a maximum only master could compute. Master no
          longer needs every grid in hand before the dataset can be sized.
          A fixed width is still required: parallel HDF5 rejects
          variable-length datatypes outright ("Parallel IO does not support
          writing VL or region reference datatypes yet"), so
          ``h5py.string_dtype()`` is not an option here.
        """
        import copy
        from mpi4py import MPI
        from ymf.archive import dump_grid
        from ymf.xdmf import parse_grid_element

        logEvent("Gathering Archive Time step")
        self.comm.barrier()
        XDMF =self.tree.getroot()
        Domain =XDMF[-1]
        #initialize Domain and grid collections on master if necessary
        if self.comm.isMaster():
            XDMFGlobal =self.treeGlobal.getroot()
            if len(XDMFGlobal) == 0:
                XDMFGlobal.append(copy.deepcopy(Domain))
                DomainGlobal = XDMFGlobal[-1]
                #delete any actual grids in the temporal  collections
                for TemporalGridCollectionGlobal in DomainGlobal:
                    del TemporalGridCollectionGlobal[:]
            else:
                DomainGlobal = XDMFGlobal[-1]

        comm_world = self.comm.comm.tompi4py()
        if self.hdfFile is not None:
            self.hdfFile.attrs[self.METADATA_VERSION_ATTR] = \
                self.METADATA_FORMAT_VERSION
            #Record the collection names, in order, so a reader can find the
            #metadata without inferring names from dataset spellings. A
            #collection name may itself contain underscores and digits
            #(Mesh_c0p2_Lagrange), so pattern-matching <name>_<step> against
            #the file's keys is guessy where this is not. Written by every
            #rank with the same value, since attribute writes are collective.
            self.hdfFile.attrs[self.COLLECTIONS_ATTR] = "\n".join(
                c.attrib['Name'] for c in Domain)
            self.hdfFile.attrs[self.GLOBAL_SYNC_ATTR] = \
                1 if self.global_sync else 0

        for i, TemporalGridCollection in enumerate(Domain):
            GridLocal = TemporalGridCollection[-1]
            time_elem = GridLocal.find("Time")
            TimeAttrib = time_elem.attrib['Value'] if time_elem is not None else "0.0"
            local_grid = parse_grid_element(GridLocal)

            if not self.global_sync:
                grid_dicts = comm_world.gather(local_grid)
                #the master's own XML tree still gets the spatial collection,
                #so the per-rank .xmf files keep their current shape
                if self.comm.isMaster():
                    TemporalGridCollectionGlobal = DomainGlobal[i]
                    SpatialCollection=SubElement(TemporalGridCollectionGlobal,"Grid",
                                                 {"GridType":"Collection",
                                                  "CollectionType":"Spatial"})
                    SpatialCollection.append(GridLocal[0])#append Time in Spatial Collection
                payloads = [dump_grid(g).encode("utf-8")
                            for g in grid_dicts] if self.comm.isMaster() else []
                n_rows = self.size
            else:
                if self.comm.isMaster():
                    TemporalGridCollectionGlobal = DomainGlobal[i]
                    TemporalGridCollectionGlobal.append(GridLocal)
                payloads = [dump_grid(local_grid).encode("utf-8")] \
                    if self.comm.isMaster() else []
                n_rows = 1

            #every rank must create the dataset with the same width, so agree
            #on it collectively. Each rank contributes the length it knows.
            local_width = max((len(pl) for pl in payloads), default=0)
            width = comm_world.allreduce(local_width, op=MPI.MAX)
            width = max(int(width), 1)

            dataset_name = self._metadata_dataset_name(
                TemporalGridCollection.attrib['Name'], self.n_datasets)
            if self.hdfFile is not None:
                try:
                    grid_data = self.hdfFile.create_dataset(
                        name  = dataset_name,
                        shape = (n_rows,),
                        dtype = '|S'+str(width))
                except (ValueError, RuntimeError, OSError):
                    grid_data = self.hdfFile[dataset_name]
                grid_data.attrs['Time'] = TimeAttrib
                if self.comm.isMaster():
                    for j, payload in enumerate(payloads):
                        grid_data[j] = payload
        self.n_datasets += 1
        logEvent("Done Gathering Archive Time Step")
    def sync(self):
        logEvent("Syncing Archive",level=3)
        memory()
        self.allGatherIncremental()
        self.clear_xml()
        if not self.useGlobalXMF:
            self.xmlFile.write(bytes(self.xmlHeader,"utf-8"))
            indentXML(self.tree.getroot())
            self.tree.write(self.xmlFile, encoding="utf-8")
            self.xmlFile.flush()
        #delete grids for step from tree
        XDMF =self.tree.getroot()
        Domain = XDMF[-1]
        for TemporalGridCollection in Domain:
            del TemporalGridCollection[:]
        if self.comm.isMaster():
            self.xmlFileGlobal.write(bytes(self.xmlHeader,"utf-8"))
            indentXML(self.treeGlobal.getroot())
            self.treeGlobal.write(self.xmlFileGlobal, encoding="utf-8")
            self.xmlFileGlobal.flush()
            #delete grids for step from tree
            XDMF = self.treeGlobal.getroot()
            Domain = XDMF[-1]
            for TemporalGridCollection in Domain:
                del TemporalGridCollection[:]
        if self.hdfFile is not None:
            self.comm.barrier()
            self.hdfFile.flush()
            self.comm.barrier()
        logEvent("Done Syncing Archive",level=3)
        logEvent(memory("Syncing Archive"),level=4)
    def create_dataset_async(self,name,data):
        comm_world = self.comm.comm.tompi4py()
        metadata = comm_world.allgather((name,data.shape,data.dtype))
        for i,m in enumerate(metadata):
            dataset = self.hdfFile.create_dataset(name  = m[0],
                                                  shape = m[1],
                                                  dtype = m[2])
            if i == self.rank:
                dataset[:] = data
    def create_dataset_sync(self,name,offsets,data):
        try:
            dataset = self.hdfFile.create_dataset(name  = name,
                                                  shape = tuple([offsets[-1]]+list(data.shape[1:])),
                                                  dtype = data.dtype)
        except:
            try:
                dataset = self.hdfFile[name]
            except Exception as e:
                raise e
        dataset[offsets[self.rank]:offsets[self.rank+1]] = data

    def write_field(self, grid, name, data, tCount,
                    center="Node", rank="Scalar", dimensions=None,
                    sync_offsets=None, sync_data=None, dataset=None,
                    text_stem=None):
        """Attach one field to ``grid`` and write its array. Returns the DataItem.

        This is the single place an ``Attribute`` + ``DataItem`` pair gets
        built. Before it existed, that eight-line block was written out by
        hand at roughly fifty call sites across Archiver.py, FemTools.py and
        MeshTools.py, each repeating the same four-way branch on
        ``global_sync`` and ``hdfFile``. Two consequences of that
        duplication worth knowing about:

        * ``DataType`` and ``Precision`` were hardcoded per call site --
          usually ``Float``/``8``. A ``float32`` field was therefore
          described to consumers as 8-byte, which a viewer reads as
          garbage. Here they come from the array's own dtype via
          :func:`ymf.archive.data_item_for`, so they cannot disagree with
          the data.
        * The ``xi:include`` for the text fallback had to be attached to the
          right ``DataItem`` by hand, and at ``Archiver.py:1475`` it wasn't
          -- the reference for one field landed on another field's
          DataItem. Here the include is attached to the DataItem this call
          just created, so that class of bug is not expressible.

        Parameters
        ----------
        grid : Element
            The XDMF ``Grid`` element to attach the Attribute to.
        name : str
            Field name, used for both the Attribute and the dataset.
        data : numpy.ndarray
            The values to write. In the ``global_sync`` case this is the
            full local array and ``sync_data`` is the owned slice actually
            written; ``data`` is still used for its dtype.
        dimensions : sequence of int, optional
            The logical shape to declare. Required when it differs from
            ``data.shape``, which it does for ``global_sync`` writes: the
            DataItem describes the *global* array while each rank
            contributes only the part it owns.
        sync_offsets, sync_data :
            Passed to :meth:`create_dataset_sync` for ``global_sync``
            writes.
        dataset : str, optional
            Overrides the HDF5 dataset name. The default follows the
            convention most call sites use -- ``<name>_t<tCount>`` when
            synchronized, ``<name>_p<rank>_t<tCount>`` otherwise -- but a
            few writers predate it and use their own. Those pass their name
            explicitly so that converting them does not rename datasets
            inside existing archives.
        text_stem : str, optional
            Overrides the sidecar filename stem for the text fallback,
            which defaults to ``<name><tCount>``.
        """
        from ymf.archive import data_item_for

        if dataset is not None:
            dataset_name = dataset
        elif self.global_sync:
            dataset_name = "{0:s}_t{1:d}".format(name, tCount)
        else:
            dataset_name = "{0:s}_p{1:s}_t{2:d}".format(
                name, repr(self.comm.rank()), tCount)

        if text_stem is None:
            text_stem = "{0:s}{1:d}".format(name, tCount)
        #textDataDir only exists when the archive was opened with
        #useTextArchive=True, so it must not be touched on the HDF5 path
        text_path = ("{0:s}/{1:s}.txt".format(self.textDataDir, text_stem)
                     if self.hdfFile is None else None)

        item = data_item_for(
            data,
            data="{0:s}:/{1:s}".format(self.hdfFilename, dataset_name)
            if self.hdfFile is not None else None,
            include="./" + text_path if self.hdfFile is None else None,
            dimensions=list(data.shape) if dimensions is None else dimensions,
            #On a genuinely parallel collective write the DataItem
            #describes the assembled global array while `data` is this
            #rank's slice, present only for its dtype, so the declared
            #dimensions are larger than the array by design and checking
            #them would reject a correct write.
            #
            #The check stays on for size==1, where global and local
            #coincide: that is the case where a declared size that does not
            #match the data means a genuinely malformed archive, and it is
            #how the phi_s corruption was found. Narrowing rather than
            #disabling keeps that safety net for the common serial run.
            check=not (self.global_sync and self.size > 1),
        )

        attribute = SubElement(grid, "Attribute",
                               {"Name": name,
                                "AttributeType": rank,
                                "Center": center})
        values = SubElement(attribute, "DataItem",
                            {"Format": self.dataItemFormat,
                             "DataType": item["DataType"],
                             "Precision": str(item["Precision"]),
                             "Dimensions": " ".join(
                                 str(d) for d in item["Dimensions"])})

        if self.hdfFile is not None:
            values.text = item["Data"]
            if self.global_sync:
                self.create_dataset_sync(dataset_name,
                                         offsets=sync_offsets,
                                         data=sync_data)
            else:
                self.create_dataset_async(dataset_name, data=data)
        else:
            assert not self.global_sync, \
                "global_sync is not supported with text heavy data"
            numpy.savetxt(text_path, data)
            # Attached to the DataItem this call created -- see the note above.
            SubElement(values, "xi:include",
                       {"parse": "text", "href": item["Include"]})
        return values

    def write_grid(self, collection, grid_name, t, tCount):
        """Create the ``Grid``/``Time`` pair for one timestep of a mesh.

        Returns ``(grid, time)``. The ten ``writeMeshXdmf_*`` methods in
        :class:`XdmfWriter` each open with this same pair, spelled either
        ``"%e" % (t,)``/``str(tCount)`` or
        ``"{0:e}".format(t)``/``"{0:d}".format(tCount)`` depending on the
        method -- both produce identical strings, so one spelling serves.
        """
        grid = SubElement(collection, "Grid",
                          {"Name": grid_name, "GridType": "Uniform"})
        time = SubElement(grid, "Time",
                          {"Value": "{0:e}".format(t),
                           "Name": "{0:d}".format(tCount)})
        return grid, time

    def write_topology(self, grid, topology_type, n_elements, dimensions,
                       dataset, text_stem, nodes_per_element=None,
                       data_type="Int", precision=None):
        """Write a ``Topology`` and its ``DataItem``. Returns the DataItem.

        The array reference is filled in here, but the array itself is
        **not** written: mesh datasets are only created when ``init or
        meshChanged``, while the reference is written on every pass. The
        caller keeps that decision, and the data preparation that goes with
        it, because it differs at every call site (element-to-node maps,
        DG node duplication, particle index ranges).
        """
        attrs = {"Type": topology_type, "NumberOfElements": str(n_elements)}
        if nodes_per_element is not None:
            attrs["NodesPerElement"] = str(nodes_per_element)
        topology = SubElement(grid, "Topology", attrs)
        return self._mesh_data_item(topology, dimensions, dataset, text_stem,
                                    data_type, precision)

    def write_geometry(self, grid, dimensions, dataset, text_stem,
                       geometry_type="XYZ", data_type="Float", precision=8):
        """Write a ``Geometry`` and its ``DataItem``. Returns the DataItem.

        Same division of labour as :meth:`write_topology`.
        """
        geometry = SubElement(grid, "Geometry", {"Type": geometry_type})
        return self._mesh_data_item(geometry, dimensions, dataset, text_stem,
                                    data_type, precision)

    def _mesh_data_item(self, parent, dimensions, dataset, text_stem,
                        data_type, precision):
        """The DataItem shared by :meth:`write_topology`/:meth:`write_geometry`.

        Unlike :meth:`write_field` this cannot read the dtype off an array,
        because the array often does not exist yet at this point -- the
        reference is written on every pass and the data only when the mesh
        changed. So ``data_type``/``precision`` are stated by the caller.
        """
        attrs = {"Format": self.dataItemFormat,
                 "DataType": data_type,
                 "Dimensions": " ".join(str(d) for d in dimensions)}
        if precision is not None:
            attrs["Precision"] = str(precision)
        item = SubElement(parent, "DataItem", attrs)
        if self.hdfFile is not None:
            item.text = "{0:s}:/{1:s}".format(self.hdfFilename, dataset)
        else:
            SubElement(item, "xi:include",
                       {"parse": "text",
                        "href": "./{0:s}/{1:s}.txt".format(self.textDataDir,
                                                           text_stem)})
        return item

def readArchiveDomain(filename, dataDir='.'):
    """The ymf domain of an existing archive, read from its HDF5 file.

    For tools that hold a filename rather than a live archive object. The
    ``.h5`` is self-describing -- it carries the metadata format version,
    the time-collection names and whether the run wrote global arrays or
    one grid per subdomain -- so no ``.xmf`` or ``.ymf`` sidecar is needed
    and no flags have to be supplied.

    That is what makes the sidecars recoverable: an archive whose ``.ymf``
    or ``.xmf`` was lost or truncated can be rebuilt from the ``.h5``
    alone.
    """
    import h5py

    base = os.path.join(dataDir, filename)
    archive = AR_base.__new__(AR_base)
    archive.hdfFilename = filename + ".h5"
    archive.global_sync = True          # overridden by the recorded value
    archive.archived_domain = None
    with h5py.File(base + ".h5", "r") as hdfFile:
        archive.hdfFile = hdfFile
        domain = archive.assemble_domain()
    archive.hdfFile = None
    return domain


XdmfArchive=AR_base

########################################################################
#for writing out various quantities in Xdmf format
#import xml.etree.ElementTree as ElementTree
#from xml.etree.ElementTree import SubElement
import numpy
class XdmfWriter(object):
    """
    collect functionality for writing data to Xdmf format

    Writer is supposed to keep track of grid collection (temporal collection)

    as well as current grid under grid collection where data belong,
    since data are associated with a grid of specific type
    (e.g., P1 Lagrange, P2 Lagrange, elementQuadrature dictionary, ...)
    """
    def __init__(self,shareSingleGrid=True,arGridCollection=None,arGrid=None,arTime=None):
        self.arGridCollection = arGridCollection #collection of "grids" (at least one per time level)
        self.arGrid           = arGrid #grid in collection that data should be associated with
        self.arTime           = arTime #time level for data
        self.shareSingleGrid  = shareSingleGrid

    def setGridCollectionAndGridElements(self,init,ar,arGrid,t,spaceSuffix):
        """
        attempt at boiler plate code to grab current arGridCollection and grid for
        a given type and time level t
        returns gridName to use in writing mesh
        """
        if init:
            #ar should have domain now and mesh should have gridCollection
            #but want own grid collection to start
            self.arGridCollection = SubElement(ar.domain,"Grid",{"Name":"Mesh"+spaceSuffix,
                                                                 "GridType":"Collection",
                                                                 "CollectionType":"Temporal"})
        elif self.arGridCollection is None:#try to get existing grid collection
            for child in ar.domain:
                if child.tag == "Grid" and child.attrib["Name"] == "Mesh"+spaceSuffix:
                    self.arGridCollection = child
        assert self.arGridCollection is not None

        #see if dgp1 grid exists with current time?

        if self.shareSingleGrid:
            gridName = "Grid"+spaceSuffix
            if arGrid is not None:
                self.arGrid = arGrid
                gt = arGrid.find("Time")
                self.arTime = gt
            #brute force search through child grids of arGridCollection
            #grids = self.arGridCollection.findall("Grid")
            #for g in grids:
            #    for gt in g:
            #        if gt.tag == "Time" and gt.attrib["Value"] == str(t):
            #            self.arTime = gt
            #            self.arGrid = g
            #            break
            #end brute force search

        else:
            #`name` was not in scope here; this branch is only reachable
            #with shareSingleGrid=False, which no caller sets.
            gridName = "Grid"+spaceSuffix
        return gridName

    def writeMeshXdmf_elementQuadrature(self,ar,mesh,spaceDim,x,t=0.0,
                                       init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.writeMeshXdmf_quadraturePoints(ar,mesh,spaceDim,x,t=t,quadratureType="q",
                                                   init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)
    def writeMeshXdmf_elementBoundaryQuadrature(self,ar,mesh,spaceDim,x,t=0.0,
                                                init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.writeMeshXdmf_quadraturePoints(ar,mesh,spaceDim,x,t=t,quadratureType="ebq_global",
                                                   init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)
    def writeMeshXdmf_exteriorElementBoundaryQuadrature(self,ar,mesh,spaceDim,x,t=0.0,
                                                        init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.writeMeshXdmf_quadraturePoints(ar,mesh,spaceDim,x,t=t,quadratureType="ebqe",
                                                   init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)

    def writeScalarXdmf_elementQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementQuadrature"
        return self.writeScalarXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeVectorXdmf_elementQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementQuadrature"
        return self.writeVectorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeTensorXdmf_elementQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementQuadrature"
        return self.writeTensorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    #
    def writeScalarXdmf_elementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementBoundaryQuadrature"
        return self.writeScalarXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeVectorXdmf_elementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementBoundaryQuadrature"
        return self.writeVectorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeTensorXdmf_elementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementBoundaryQuadrature"
        return self.writeTensorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    #
    def writeScalarXdmf_exteriorElementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_exteriorElementBoundaryQuadrature"
        return self.writeScalarXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeVectorXdmf_exteriorElementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_exteriorElementBoundaryQuadrature"
        return self.writeVectorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeTensorXdmf_exteriorElementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_exteriorElementBoundaryQuadrature"
        return self.writeTensorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)


    def writeMeshXdmf_quadraturePoints(self,ar,mesh,spaceDim,x,t=0.0,quadratureType="q",
                                       init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        write out quadrature point mesh for quadrature points that are unique to
        either elements or elementBoundaries

        quadrature type
           q  --> element
           ebq_global --> element boundary
           ebqe       --> exterior element boundary
        """
        if quadratureType == "q":
            spaceSuffix = "_elementQuadrature"
        elif quadratureType == "ebq_global":
            spaceSuffix = "_elementBoundaryQuadrature"
        elif quadratureType == "ebqe":
            spaceSuffix = "_exteriorElementBoundaryQuadrature"
        else:
            raise RuntimeError("quadratureType = %s not recognized" % quadratureType)

        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that is a collection of points per element

        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        assert len(x.shape) == 3 #make sure have right type of dictionary
        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if ar.global_sync:
                self.mesh = mesh
                Xdmf_ElementTopology = "Polyvertex"
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                Xdmf_NodesPerElement = x.shape[1]
                Xdmf_NodesGlobal     = Xdmf_NumberOfElements*Xdmf_NodesPerElement

                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":"%i" % (tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements),
                                          "NodesPerElement":str(Xdmf_NodesPerElement)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (Xdmf_NodesGlobal,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #this will fail if elements_dgp1 already exists
                        #q_l2g = numpy.zeros((Xdmf_NumberOfElements,Xdmf_NodesPerElement),'i')
                        #brute force to start
                        #for eN in range(Xdmf_NumberOfElements):
                        #    for nN in range(Xdmf_NodesPerElement):
                        #        q_l2g[eN,nN] = eN*Xdmf_NodesPerElement + nN
                        #
                        from proteus import Comm
                        comm = Comm.get()
                        q_l2g = numpy.arange(mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank]*Xdmf_NodesPerElement,
                                             mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]*Xdmf_NodesPerElement,
                                             dtype='i').reshape((mesh.nElements_owned,Xdmf_NodesPerElement))
                        ar.create_dataset_sync('elements'+spaceSuffix+str(tCount),
                                               offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = q_l2g)
                        ar.create_dataset_sync('nodes'+spaceSuffix+str(tCount),
                                               offsets=mesh.globalMesh.elementOffsets_subdomain_owned*Xdmf_NodesPerElement,
                                               data = x[:mesh.nElements_owned].reshape((mesh.nElements_owned*Xdmf_NodesPerElement,3)))
                else:
                    assert False, "global_sync not supported with text  heavy data"
            else:
                Xdmf_ElementTopology = "Polyvertex"
                Xdmf_NumberOfElements= x.shape[0]
                Xdmf_NodesPerElement = x.shape[1]
                Xdmf_NodesGlobal     = Xdmf_NumberOfElements*Xdmf_NodesPerElement

                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":"%i" % (tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements),
                                          "NodesPerElement":str(Xdmf_NodesPerElement)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (Xdmf_NodesGlobal,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #this will fail if elements_dgp1 already exists
                        #q_l2g = numpy.zeros((Xdmf_NumberOfElements,Xdmf_NodesPerElement),'i')
                        #brute force to start
                        #for eN in range(Xdmf_NumberOfElements):
                        #    for nN in range(Xdmf_NodesPerElement):
                        #        q_l2g[eN,nN] = eN*Xdmf_NodesPerElement + nN
                        #
                        q_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = q_l2g)
                        ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = x.flat[:])
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        #this will fail if elements_dgp1 already exists
                        #q_l2g = numpy.zeros((Xdmf_NumberOfElements,Xdmf_NodesPerElement),'i')
                        #brute force to start
                        #for eN in range(Xdmf_NumberOfElements):
                        #    for nN in range(Xdmf_NodesPerElement):
                        #        q_l2g[eN,nN] = eN*Xdmf_NodesPerElement + nN
                        #
                        q_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",q_l2g,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",x.flat[:])

                    #
                #hdfile
        #need to write a grid
        return self.arGrid
    #def
    def writeScalarXdmf_quadrature(self,ar,u,name,tCount=0,init=True):
        assert len(u.shape) == 2
        if ar.global_sync:
            #the DataItem covers every element's quadrature points globally;
            #this rank contributes only the elements it owns
            n_owned = self.mesh.nElements_owned
            ar.write_field(self.arGrid, name, u, tCount,
                           dimensions=[self.mesh.globalMesh.nElements_global*u.shape[1]],
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*u.shape[1],
                           sync_data=u[:n_owned].reshape((n_owned*u.shape[1],)))
        else:
            ar.write_field(self.arGrid, name, u.flat[:], tCount,
                           dimensions=[u.shape[0]*u.shape[1]])

    def writeVectorXdmf_quadrature(self,ar,u,name,tCount=0,init=True):
        assert len(u.shape) == 3
        #XDMF vectors are 3-component because the points are 3D, so the
        #components the field doesn't have stay zero
        Xdmf_StorageDim = 3
        Xdmf_NumberOfComponents = u.shape[2]
        if ar.global_sync:
            n_owned = self.mesh.nElements_owned
            n_local = n_owned*u.shape[1]
            tmp = numpy.zeros((n_local,Xdmf_StorageDim),'d')
            tmp[:,:Xdmf_NumberOfComponents] = numpy.reshape(
                u[:n_owned].flat,(n_local,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[self.mesh.globalMesh.nElements_global*u.shape[1],
                                       Xdmf_StorageDim],
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*u.shape[1],
                           sync_data=tmp)
        else:
            Xdmf_NodesGlobal = u.shape[0]*u.shape[1]
            tmp = numpy.zeros((Xdmf_NodesGlobal,Xdmf_StorageDim),'d')
            tmp[:,:Xdmf_NumberOfComponents] = numpy.reshape(
                u.flat,(Xdmf_NodesGlobal,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[Xdmf_NodesGlobal,Xdmf_StorageDim])

    def writeTensorXdmf_quadrature(self,ar,u,name,tCount=0,init=True):
        """
        TODO make faster tmp creation
        """
        assert len(u.shape) == 4
        #XDMF tensors are 9-component; a 2x2 tensor occupies the leading
        #corner of a 3x3 and the rest stays zero
        Xdmf_NumberOfComponents = u.shape[2]*u.shape[3]

        def pack(rows, source):
            tmp = numpy.zeros((rows,9),'d')
            for k in range(rows):
                for i in range(u.shape[2]):
                    for j in range(u.shape[3]):
                        tmp.flat[k*9 + i*3 + j] = source.flat[
                            k*Xdmf_NumberOfComponents + i*u.shape[2] + j]
            return tmp

        if ar.global_sync:
            #sized by the elements this rank owns, matching
            #writeVectorXdmf_quadrature. The previous version sized tmp by
            #the *global* element count while reading from the local array,
            #which indexes past the end of u.
            n_owned = self.mesh.nElements_owned
            tmp = pack(n_owned*u.shape[1], u[:n_owned])
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Tensor",
                           dimensions=[self.mesh.globalMesh.nElements_global*u.shape[1],9],
                           #.globalMesh was missing here; every other call
                           #site in this file reads the offsets off the
                           #global mesh, and a subdomain mesh has no such
                           #attribute
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*u.shape[1]*9,
                           sync_data=tmp)
        else:
            Xdmf_NodesGlobal = u.shape[0]*u.shape[1]
            tmp = pack(Xdmf_NodesGlobal, u)
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Tensor",
                           dimensions=[Xdmf_NodesGlobal,9])


    def writeMeshXdmf_DGP1Lagrange(self,ar,name,mesh,spaceDim,dofMap,CGDOFMap,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        TODO Not tested yet
        """
        comm = Comm.get()
        #assert False, "Not tested"
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches DG P1 layout

        spaceSuffix = "_dgp1_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Polyline"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Triangle"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tetrahedron"
            if ar.global_sync:
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"{0:e}".format(t),"Name":"{0:d}".format(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements": "{0:d}".format(mesh.globalMesh.nElements_global)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"{0:d} {1:d}".format(mesh.globalMesh.nElements_global,mesh.nNodes_element)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"{0:d} {1:d}".format(dofMap.nDOF_all_processes,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #this will fail if elements_dgp1 already exists
                        ar.create_dataset_sync('elements{0}{1:d}'.format(spaceSuffix,tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = dofMap.dof_offsets_subdomain_owned[ar.rank]+dofMap.l2g[:mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]
                                                                                                             -mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank]])
                        #bad
                        dgnodes = numpy.zeros((dofMap.dof_offsets_subdomain_owned[ar.rank+1]-dofMap.dof_offsets_subdomain_owned[ar.rank],3),'d')
                        for eN in range(mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]
                                        -mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank]):
                            for nN in range(mesh.nNodes_element):
                                dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                        #make more pythonic loop
                        ar.create_dataset_sync('nodes{0}{1:d}'.format(spaceSuffix,tCount),
                                               offsets=dofMap.dof_offsets_subdomain_owned,
                                               data = dgnodes)
                else:
                    assert False, "Global sync  not implemented for text heavy data"
            else:
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(mesh.nElements_global)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (mesh.nElements_global,mesh.nNodes_element)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (dofMap.nDOF,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #this will fail if elements_dgp1 already exists
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = dofMap.l2g)
                        #bad
                        dgnodes = numpy.zeros((dofMap.nDOF,3),'d')
                        for eN in range(mesh.nElements_global):
                            for nN in range(mesh.nNodes_element):
                                dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                        ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = dgnodes)
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",dofMap.l2g,fmt='%d')
                        #bad
                        dgnodes = numpy.zeros((dofMap.nDOF,3),'d')
                        for eN in range(mesh.nElements_global):
                            for nN in range(mesh.nNodes_element):
                                dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                        #make more pythonic loop
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",dgnodes)

                    #
                #hdfile
            #need to write a grid
        return self.arGrid
    #def
    def writeMeshXdmf_DGP2Lagrange(self,ar,name,mesh,spaceDim,dofMap,CGDOFMap,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches DG P1 layout
        #first duplicate geometry points etc then try to save space
        spaceSuffix = "_dgp2_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Edge_3"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Tri_6"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tet_10"
            if ar.global_sync:
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"{0:e}".format(t),"Name":"{0:d}".format(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":"{0:d}".format(mesh.globalMesh.nElements_global)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"{0:d} {1:d}".format(mesh.globalMesh.nElements_global,dofMap.l2g.shape[-1])})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                #try to use fancy functions later
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"{0:d} {1:d}".format(dofMap.nDOF_all_processes,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #this will fail if elements_dgp1 already exists
                        ar.create_dataset_sync('elements'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = dofMap.dof_offsets_subdomain_owned[ar.rank]+dofMap.l2g[:mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]-
                                                                                                             mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank]])
                        #bad
                        dgnodes = numpy.zeros((dofMap.dof_offsets_subdomain_owned[ar.rank+1]-dofMap.dof_offsets_subdomain_owned[ar.rank],3),'d')
                        for eN in range(mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]-mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank]):
                            for i in range(dofMap.l2g.shape[1]):
                                #for nN in range(mesh.nNodes_element):
                                #dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                                #now changed lagrange nodes to hold all nodes
                                dgnodes[dofMap.l2g[eN,i],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,i],:]
                            #next loop over extra dofs on element and write out
                            #for nN in range(mesh.nNodes_element,dofMap.l2g.shape[1]):
                            #    dgnodes[dofMap.l2g[eN,nN],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,nN]-mesh.nNodes_global,:]

                        #make more pythonic loop
                        ar.create_dataset_sync('nodes'+spaceSuffix+str(tCount),
                                               offsets = dofMap.dof_offsets_subdomain_owned,
                                               data = dgnodes)
                else:
                    assert False, "global_sync not implemented for  text heavy data"
            else:
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(mesh.nElements_global)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % dofMap.l2g.shape})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                #try to use fancy functions later
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (dofMap.nDOF,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #this will fail if elements_dgp1 already exists
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = dofMap.l2g)
                        #bad
                        dgnodes = numpy.zeros((dofMap.nDOF,3),'d')
                        for eN in range(mesh.nElements_global):
                            for i in range(dofMap.l2g.shape[1]):
                                #for nN in range(mesh.nNodes_element):
                                #dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                                #now changed lagrange nodes to hold all nodes
                                dgnodes[dofMap.l2g[eN,i],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,i],:]
                            #next loop over extra dofs on element and write out
                            #for nN in range(mesh.nNodes_element,dofMap.l2g.shape[1]):
                            #    dgnodes[dofMap.l2g[eN,nN],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,nN]-mesh.nNodes_global,:]

                        ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = dgnodes)
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",dofMap.l2g,fmt='%d')
                        #bad
                        dgnodes = numpy.zeros((dofMap.nDOF,3),'d')
                        for eN in range(mesh.nElements_global):
                            for i in range(dofMap.l2g.shape[1]):
                                #for nN in range(mesh.nNodes_element):
                                #dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                                dgnodes[dofMap.l2g[eN,i],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,i],:]
                            #next loop over extra dofs on element and write out
                            #for nN in range(mesh.nNodes_element,dofMap.l2g.shape[1]):
                            #    dgnodes[dofMap.l2g[eN,nN],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,nN]-mesh.nNodes_global,:]

                        #make more pythonic loop
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",dgnodes)

                    #
                #hdfile
            #need to write a grid
        return self.arGrid

    def writeMeshXdmf_C0P2Lagrange(self,ar,name,mesh,spaceDim,dofMap,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        TODO: test new lagrangeNodes convention for 2d,3d
        """
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        spaceSuffix = "_c0p2_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)
        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Edge_3"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Tri_6"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tet_10"
            lagrangeNodesArray = dofMap.lagrangeNodesArray

            #the synchronized case describes the assembled global arrays and
            #names its datasets without a rank; the per-rank case describes
            #this subdomain and carries the rank in the dataset name. Sidecar
            #names omit the rank in both.
            if ar.global_sync:
                n_elements       = mesh.globalMesh.nElements_global
                elements_dims    = [n_elements, dofMap.l2g.shape[-1]]
                nodes_dims       = [dofMap.nDOF_all_processes, 3]
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
            else:
                n_elements       = mesh.nElements_global
                elements_dims    = list(dofMap.l2g.shape)
                nodes_dims       = [dofMap.nDOF, 3]
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid, self.arTime = ar.write_grid(self.arGridCollection,
                                                     gridName, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              elements_dims, elements_dataset, elements_stem)
            ar.write_geometry(self.arGrid, nodes_dims,
                              nodes_dataset, nodes_stem)

            if ar.hdfFile is None:
                assert not ar.global_sync, \
                    "global_sync is not implemented for text heavy data"
                if init or meshChanged:
                    numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",
                                  dofMap.l2g,fmt='%d')
                    numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",
                                  lagrangeNodesArray)
                return self.arGrid

            if spaceDim == 3:
                #proteus stores 3d dof as
                #|n0,n1,n2,n3|(n0,n1),(n1,n2),(n2,n3)|(n0,n2),(n1,n3)|(n0,n3)|
                #xdmf wants them as
                #|n0,n1,n2,n3|(n0,n1),(n1,n2),(n0,n2) (n0,n3),(n1,n3) (n2,n3)|
                import copy
                element_nodes = copy.deepcopy(dofMap.l2g)
                for eN in range(mesh.nElements_global):
                    element_nodes[eN,4+2] = dofMap.l2g[eN,4+3]
                    element_nodes[eN,4+3] = dofMap.l2g[eN,4+5]
                    element_nodes[eN,4+5] = dofMap.l2g[eN,4+2]
            else:
                element_nodes = dofMap.l2g

            #references were written above on every pass; the arrays only when
            #the mesh actually changed
            if init or meshChanged:
                if ar.global_sync:
                    owned = dofMap.dof_offsets_subdomain_owned
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = dofMap.subdomain2global[element_nodes[:mesh.nElements_owned]])
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = owned,
                                           data = lagrangeNodesArray[:owned[ar.rank+1]-owned[ar.rank]])
                else:
                    ar.create_dataset_async(elements_dataset, data = element_nodes)
                    ar.create_dataset_async(nodes_dataset, data = lagrangeNodesArray)
        return self.arGrid

    def writeMeshXdmf_C0Q2Lagrange(self,ar,name,mesh,spaceDim,dofMap,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        TODO: test new lagrangeNodes convention for 2d,3d
        """
        #write out basic geometry if not already done?
        #mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        spaceSuffix = "_c0q2_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)
        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if spaceDim == 1:
                print("No writeMeshXdmf_C0Q2Lagrange for 1D")
                return 0
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Quadrilateral"
                e2s=[ [0,4,8,7], [7,8,6,3],  [4,1,5,8], [8,5,2,6] ]
                nsubelements=4

                l2g = numpy.zeros((4*mesh.nElements_global,4),'i')
                for eN in range(mesh.nElements_global):
                    dofs=dofMap.l2g[eN,:]
                    for i in range(4): #loop over subelements
                        for j in range(4): # loop over nodes of subelements
                            l2g[4*eN+i,j] = dofs[e2s[i][j]]
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Hexahedron"

                e2s=[[0,8,20,11,12,21,26,24], [8,1,9,20,21,13,22,26], [11,20,10,3,24,26,23,15], [20,9,2,10,26,22,14,23],
                     [12,21,26,24,4,16,25,19], [21,13,22,26,16,5,17,25], [24,26,23,15,19,25,18,7], [26,22,14,23,25,17,6,18] ]

                l2g = numpy.zeros((8*mesh.nElements_global,8),'i')
                nsubelements=8
                for eN in range(mesh.nElements_global):
                    dofs=dofMap.l2g[eN,:]
                    for i in range(8): #loop over subelements
                        for j in range(8): # loop over nodes of subelement
                            l2g[8*eN+i,j] = dofs[e2s[i][j]]
            if ar.global_sync:
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})

                lagrangeNodesArray = dofMap.lagrangeNodesArray

                topology = SubElement(self.arGrid,"Topology",
                                      {"Type":Xdmf_ElementTopology,
                                       "NumberOfElements":str(mesh.globalMesh.nElements_global*nsubelements)})
                elements = SubElement(topology,"DataItem",
                                      {"Format":ar.dataItemFormat,
                                       "DataType":"Int",
                                       "Dimensions":"%i %i" % (mesh.globalMesh.nElements_global*nsubelements, l2g.shape[-1])})
                geometry = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})

                allNodes = SubElement(geometry,"DataItem",
                                      {"Format":ar.dataItemFormat,
                                       "DataType":"Float",
                                       "Precision":"8",
                                       "Dimensions":"%i %i" % (dofMap.nDOF_all_processes, lagrangeNodesArray.shape[-1])})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    allNodes.text = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    import copy
                    if init or meshChanged:
                        ar.create_dataset_sync('elements'+spaceSuffix+str(tCount),
                                               offsets = [x*nsubelements for x in mesh.globalMesh.elementOffsets_subdomain_owned],
                                               data = dofMap.subdomain2global[l2g[:mesh.nElements_owned*nsubelements]])
                        ar.create_dataset_sync('nodes'+spaceSuffix+str(tCount),
                                               offsets = dofMap.dof_offsets_subdomain_owned,
                                               data = lagrangeNodesArray[:dofMap.dof_offsets_subdomain_owned[ar.rank+1]-dofMap.dof_offsets_subdomain_owned[ar.rank]])
                else:
                    assert False, "not implemented  for text heavy data"
            else:
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})

                lagrangeNodesArray = dofMap.lagrangeNodesArray

                topology = SubElement(self.arGrid,"Topology",
                                      {"Type":Xdmf_ElementTopology,
                                       "NumberOfElements":str(l2g.shape[0])})
                elements = SubElement(topology,"DataItem",
                                      {"Format":ar.dataItemFormat,
                                       "DataType":"Int",
                                       "Dimensions":"%i %i" % l2g.shape})
                geometry = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})

                allNodes = SubElement(geometry,"DataItem",
                                      {"Format":ar.dataItemFormat,
                                       "DataType":"Float",
                                       "Precision":"8",
                                       "Dimensions":"%i %i" % lagrangeNodesArray.shape})

                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    allNodes.text = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    import copy
                    if init or meshChanged:
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = l2g)
                        ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = lagrangeNodesArray)
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(allNodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",l2g,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",lagrangeNodesArray)
        return self.arGrid

    def writeMeshXdmf_CrouzeixRaviartP1(self,ar,mesh,spaceDim,dofMap,t=0.0,
                                        init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        Write out nonconforming P1 approximation
        Write out as a (discontinuous) Lagrange P1 function to make visualization easier
        and dof's as face centered data on original grid
        """
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches DG P1 layout

        spaceSuffix = "_ncp1_CrouzeixRaviart"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Polyline"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Triangle"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tetrahedron"
            if ar.global_sync:
                Xdmf_NodesPerElement = spaceDim+1
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements*Xdmf_NodesPerElement,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #simple dg l2g mapping
                        dg_l2g = numpy.arange(mesh.nElements_owned*Xdmf_NodesPerElement,dtype='i').reshape((mesh.nElements_owned,Xdmf_NodesPerElement))
                        ar.create_dataset_sync('elements'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = dg_l2g)
                        dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray[:mesh.nElements_owned]],
                                                (mesh.nElements_owned*Xdmf_NodesPerElement,3))
                        ar.create_dataset_sync('nodes'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned*Xdmf_NodesPerElement,
                                               data = dgnodes)
                else:
                    assert False, "global_sync not implemented for text heavy data"
            else:
                Xdmf_NodesPerElement = spaceDim+1
                Xdmf_NumberOfElements= mesh.nElements_global
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements*Xdmf_NodesPerElement,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #simple dg l2g mapping
                        dg_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = dg_l2g)

                        dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray],(Xdmf_NumberOfElements*Xdmf_NodesPerElement,3))
                        ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = dgnodes)
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        dg_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",dg_l2g,fmt='%d')

                        dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray],(Xdmf_NumberOfElements*Xdmf_NodesPerElement,3))
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",dgnodes)

                    #
                #hdfile
            #need to write a grid
        return self.arGrid
    #def
    def writeFunctionXdmf_DGP1Lagrange(self,ar,u,tCount=0,init=True, dofMap=None):
        if ar.global_sync:
            assert(dofMap)
            owned = dofMap.dof_offsets_subdomain_owned
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dimensions=[dofMap.nDOF_all_processes],
                           sync_offsets=owned,
                           sync_data=u.dof[:owned[ar.rank+1] - owned[ar.rank]])
        else:
            #Dimensions comes from the array rather than u.nDOF_global: the
            #DataItem describes exactly what is written, and the two differ
            #when the caller passed a foreign array through the residual
            #adapter (phi_s is a vertex field, not a DOF vector of this space)
            ar.write_field(self.arGrid, u.name, u.dof, tCount)

    def writeFunctionXdmf_DGP2Lagrange(self,ar,u,tCount=0,init=True, dofMap=None):
        #this writer predates the <name>_p<rank>_t<tCount> dataset
        #convention and names its dataset <name><tCount> in both the
        #synchronized and per-rank cases. Passed explicitly so converting
        #it does not rename datasets inside existing archives.
        dataset = u.name + str(tCount)
        if ar.global_sync:
            owned = dofMap.dof_offsets_subdomain_owned
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dataset=dataset,
                           dimensions=[dofMap.nDOF_all_processes],
                           sync_offsets=owned,
                           sync_data=u.dof[:owned[ar.rank+1] - owned[ar.rank]])
        else:
            #Dimensions from the array, not u.nDOF_global -- see
            #writeFunctionXdmf_DGP1Lagrange
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dataset=dataset)

    def writeFunctionXdmf_CrouzeixRaviartP1(self,ar,u,tCount=0,init=True, dofMap=None):
        if ar.global_sync:
            Xdmf_NumberOfElements = u.femSpace.mesh.globalMesh.nElements_global
            Xdmf_NodesPerElement  = u.femSpace.mesh.nNodes_element
            name = u.name.replace(' ','_')
            #if writing as dgp1
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Scalar",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i" % (Xdmf_NumberOfElements*Xdmf_NodesPerElement,)})
            nElements_owned = u.femSpace.mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1] - u.femSpace.mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank]
            u_tmp = numpy.zeros((nElements_owned*Xdmf_NodesPerElement,),'d')
            if u.femSpace.nSpace_global == 1:
                for eN in range(nElements_owned):
                    #dof associated with face id, so opposite usual C0P1 ordering here
                    u_tmp[eN*Xdmf_NodesPerElement + 0] = u.dof[u.femSpace.dofMap.l2g[eN,1]]
                    u_tmp[eN*Xdmf_NodesPerElement + 1] = u.dof[u.femSpace.dofMap.l2g[eN,0]]
            elif u.femSpace.nSpace_global == 2:
                for eN in range(nElements_owned):
                    #assume vertex associated with face across from it
                    u_tmp[eN*Xdmf_NodesPerElement + 0] = u.dof[u.femSpace.dofMap.l2g[eN,1]]
                    u_tmp[eN*Xdmf_NodesPerElement + 0]+= u.dof[u.femSpace.dofMap.l2g[eN,2]]
                    u_tmp[eN*Xdmf_NodesPerElement + 0]-= u.dof[u.femSpace.dofMap.l2g[eN,0]]

                    u_tmp[eN*Xdmf_NodesPerElement + 1] = u.dof[u.femSpace.dofMap.l2g[eN,0]]
                    u_tmp[eN*Xdmf_NodesPerElement + 1]+= u.dof[u.femSpace.dofMap.l2g[eN,2]]
                    u_tmp[eN*Xdmf_NodesPerElement + 1]-= u.dof[u.femSpace.dofMap.l2g[eN,1]]

                    u_tmp[eN*Xdmf_NodesPerElement + 2] = u.dof[u.femSpace.dofMap.l2g[eN,0]]
                    u_tmp[eN*Xdmf_NodesPerElement + 2]+= u.dof[u.femSpace.dofMap.l2g[eN,1]]
                    u_tmp[eN*Xdmf_NodesPerElement + 2]-= u.dof[u.femSpace.dofMap.l2g[eN,2]]
            else:
                for eN in range(nElements_owned):
                    for i in range(Xdmf_NodesPerElement):
                        u_tmp[eN*Xdmf_NodesPerElement + i] = u.dof[u.femSpace.dofMap.l2g[eN,i]]*(1.0-float(u.femSpace.nSpace_global)) + \
                                                             sum([u.dof[u.femSpace.dofMap.l2g[eN,j]] for j in range(Xdmf_NodesPerElement) if j != i])


            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_t"+str(tCount)
                ar.create_dataset_sync(name+"_t"+str(tCount),
                                       offsets = u.femSpace.mesh.globalMesh.elementOffsets_subdomain_owned*Xdmf_NodesPerElement,
                                       data = u_tmp)
            else:
                assert False, "global_sync not implemented for text heavy data"
            #if writing as face centered (true nc p1)
            #could be under self.arGrid or mesh.arGrid
            grid = u.femSpace.mesh.arGrid #self.arGrid
            attribute_dof = SubElement(grid,"Attribute",{"Name":name+"_dof",
                                                         "AttributeType":"Scalar",
                                                         "Center":"Face"})
            values_dof    = SubElement(attribute_dof,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i" % (dofMap.nDOF_all_processes,)})
            if ar.hdfFile is not None:
                values_dof.text = ar.hdfFilename+":/"+name+"_dof"+"_t"+str(tCount)
                ar.create_dataset_sync(name+"_dof"+"_t"+str(tCount),
                                       offsets = dofMap.dof_offsets_subdomain_owned,
                                       data = u.dof)
            else:
                assert False, "global_sync not implemented for text heavy data"
        else:
            Xdmf_NumberOfElements = u.femSpace.mesh.nElements_global
            Xdmf_NodesPerElement  = u.femSpace.mesh.nNodes_element

            name = u.name.replace(' ','_')
            #if writing as dgp1
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Scalar",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i" % (Xdmf_NumberOfElements*Xdmf_NodesPerElement,)})
            u_tmp = numpy.zeros((Xdmf_NumberOfElements*Xdmf_NodesPerElement,),'d')
            if u.femSpace.nSpace_global == 1:
                for eN in range(Xdmf_NumberOfElements):
                    #dof associated with face id, so opposite usual C0P1 ordering here
                    u_tmp[eN*Xdmf_NodesPerElement + 0] = u.dof[u.femSpace.dofMap.l2g[eN,1]]
                    u_tmp[eN*Xdmf_NodesPerElement + 1] = u.dof[u.femSpace.dofMap.l2g[eN,0]]
            elif u.femSpace.nSpace_global == 2:
                for eN in range(Xdmf_NumberOfElements):
                    #assume vertex associated with face across from it
                    u_tmp[eN*Xdmf_NodesPerElement + 0] = u.dof[u.femSpace.dofMap.l2g[eN,1]]
                    u_tmp[eN*Xdmf_NodesPerElement + 0]+= u.dof[u.femSpace.dofMap.l2g[eN,2]]
                    u_tmp[eN*Xdmf_NodesPerElement + 0]-= u.dof[u.femSpace.dofMap.l2g[eN,0]]

                    u_tmp[eN*Xdmf_NodesPerElement + 1] = u.dof[u.femSpace.dofMap.l2g[eN,0]]
                    u_tmp[eN*Xdmf_NodesPerElement + 1]+= u.dof[u.femSpace.dofMap.l2g[eN,2]]
                    u_tmp[eN*Xdmf_NodesPerElement + 1]-= u.dof[u.femSpace.dofMap.l2g[eN,1]]

                    u_tmp[eN*Xdmf_NodesPerElement + 2] = u.dof[u.femSpace.dofMap.l2g[eN,0]]
                    u_tmp[eN*Xdmf_NodesPerElement + 2]+= u.dof[u.femSpace.dofMap.l2g[eN,1]]
                    u_tmp[eN*Xdmf_NodesPerElement + 2]-= u.dof[u.femSpace.dofMap.l2g[eN,2]]
            else:
                for eN in range(Xdmf_NumberOfElements):
                    for i in range(Xdmf_NodesPerElement):
                        u_tmp[eN*Xdmf_NodesPerElement + i] = u.dof[u.femSpace.dofMap.l2g[eN,i]]*(1.0-float(u.femSpace.nSpace_global)) + \
                                                             sum([u.dof[u.femSpace.dofMap.l2g[eN,j]] for j in range(Xdmf_NodesPerElement) if j != i])


            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_p"+str(ar.rank)+"_t"+str(tCount)
                ar.create_dataset_async(name+"_p"+str(ar.rank)+"_t"+str(tCount), data = u_tmp)
            else:
                numpy.savetxt(ar.textDataDir+"/"+name+str(tCount)+".txt",u_tmp)
                SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+name+str(tCount)+".txt"})


            #if writing as face centered (true nc p1)
            #could be under self.arGrid or mesh.arGrid
            grid = u.femSpace.mesh.arGrid #self.arGrid
            attribute_dof = SubElement(grid,"Attribute",{"Name":name+"_dof",
                                                         "AttributeType":"Scalar",
                                                         "Center":"Face"})
            values_dof    = SubElement(attribute_dof,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i" % (u.nDOF_global,)})
            if ar.hdfFile is not None:
                values_dof.text = ar.hdfFilename+":/"+name+"_dof"+"_p"+str(ar.rank)+"_t"+str(tCount)
                ar.create_dataset_async(name+"_dof"+"_p"+str(ar.rank)+"_t"+str(tCount), data = u.dof)
            else:
                numpy.savetxt(ar.textDataDir+"/"+name+"_dof"+str(tCount)+".txt",u.dof)
                SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+name+"_dof"+str(tCount)+".txt"})

    def writeVectorFunctionXdmf_nodal(self,ar,uList,components,vectorName,spaceSuffix,tCount=0,init=True):
        nDOF_global = uList[components[0]].nDOF_global
        if ar.global_sync:
            attribute = SubElement(self.arGrid,"Attribute",{"Name":vectorName,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i %i" % (uList[components[0]].femSpace.dofMap.nDOF_all_processes,3)})
            u_dof = uList[components[0]].dof
            if len(components) < 2:
                v_dof = numpy.zeros(u_dof.shape,dtype='d')
            else:
                v_dof = uList[components[1]].dof
            if len(components) < 3:
                w_dof = numpy.zeros(u_dof.shape,dtype='d')
            else:
                w_dof = uList[components[2]].dof
            velocity = numpy.column_stack((u_dof,v_dof,w_dof))
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+vectorName+"_t"+str(tCount)
                nDOF_owned = (uList[components[0]].femSpace.dofMap.dof_offsets_subdomain_owned[ar.rank+1] -
                              uList[components[0]].femSpace.dofMap.dof_offsets_subdomain_owned[ar.rank] )
                ar.create_dataset_sync(vectorName+"_t"+str(tCount),
                                       offsets = uList[components[0]].femSpace.dofMap.dof_offsets_subdomain_owned,
                                       data = velocity[:nDOF_owned])
        else:
            attribute = SubElement(self.arGrid,"Attribute",{"Name":vectorName,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i %i" % (nDOF_global,3)})
            u_dof = uList[components[0]].dof
            if len(components) < 2:
                v_dof = numpy.zeros(u_dof.shape,dtype='d')
            else:
                v_dof = uList[components[1]].dof
            if len(components) < 3:
                w_dof = numpy.zeros(u_dof.shape,dtype='d')
            else:
                w_dof = uList[components[2]].dof
            velocity = numpy.column_stack((u_dof,v_dof,w_dof))
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+vectorName+"_p"+str(ar.rank)+"_t"+str(tCount)
                ar.create_dataset_async(vectorName+"_p"+str(ar.rank)+"_t"+str(tCount), data = velocity)

    def writeVectorFunctionXdmf_CrouzeixRaviartP1(self,ar,uList,components,spaceSuffix,tCount=0,init=True):
        return self.writeVectorFunctionXdmf_nodal(ar,uList,components,"_ncp1_CrouzeixRaviart",
                                                  tCount=tCount,init=init)

    def writeMeshXdmf_MonomialDGPK(self,ar,mesh,spaceDim,interpolationPoints,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        as a first cut, archive DG monomial spaces using same approach as for element quadrature
        arrays using x = interpolation points (which are just Gaussian quadrature points) as values
        """

        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that is a collection of points per element
        spaceSuffix = "_dgpk_Monomial"

        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        assert len(interpolationPoints.shape) == 3 #make sure have right type of dictionary
        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            Xdmf_ElementTopology = "Polyvertex"
            if ar.global_sync:
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                Xdmf_NodesPerElement = interpolationPoints.shape[1]
                Xdmf_NodesGlobal     = Xdmf_NumberOfElements*Xdmf_NodesPerElement

                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements),
                                          "NodesPerElement":str(Xdmf_NodesPerElement)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (Xdmf_NodesGlobal,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        q_l2g = numpy.arange(mesh.nElements_owned*Xdmf_NodesPerElement,dtype='i').reshape((mesh.nElements_owned,Xdmf_NodesPerElement))
                        ar.create_dataset_sync('elements'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = q_l2g)
                        ar.create_dataset_sync('nodes'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = interpolationPoints[:mesh.nElements_owned])
                else:
                    assert False, "global_sync not implementedf or text heavy data"
            else:
                Xdmf_NumberOfElements= interpolationPoints.shape[0]
                Xdmf_NodesPerElement = interpolationPoints.shape[1]
                Xdmf_NodesGlobal     = Xdmf_NumberOfElements*Xdmf_NodesPerElement

                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements),
                                          "NodesPerElement":str(Xdmf_NodesPerElement)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (Xdmf_NodesGlobal,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        q_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = q_l2g)
                        ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = interpolationPoints.flat[:])
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        q_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",q_l2g,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",interpolationPoints.flat[:])

                    #
                #hdfile
            #need to write a grid
        return self.arGrid
    #def
    def writeFunctionXdmf_MonomialDGPK(self,ar,interpolationValues,name,tCount=0,init=True, mesh=None):
        """
        Different than usual FemFunction Write routines since saves values at interpolation points
        need to check on way to save dofs as well
        """
        assert len(interpolationValues.shape) == 2
        if ar.global_sync:
            ar.write_field(self.arGrid, name, interpolationValues, tCount,
                           dimensions=[mesh.globalMesh.nElements_global
                                       *interpolationValues.shape[1]],
                           sync_offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                           sync_data=interpolationValues[:mesh.nElements_owned])
        else:
            ar.write_field(self.arGrid, name, interpolationValues.flat[:], tCount,
                           dimensions=[interpolationValues.shape[0]
                                       *interpolationValues.shape[1]])

    def writeVectorFunctionXdmf_MonomialDGPK(self,ar,interpolationValues,name,tCount=0,init=True):
        """
        Different than usual FemFunction Write routines since saves values at interpolation points
        need to check on way to save dofs as well
        """
        assert len(interpolationValues.shape) == 3
        if ar.global_sync:
            Xdmf_NodesGlobal = self.mesh.globalMesh.nElements_global*interpolationValues.shape[1]
            Xdmf_NumberOfComponents = interpolationValues.shape[2]
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i %i" % (Xdmf_NodesGlobal,3)})#force 3d vector since points 3d
            #mwf brute force
            tmp = numpy.zeros((Xdmf_NodesGlobal,3),'d')
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(interpolationValues.flat,(interpolationValues.shape[0]*interpolationValues.shape[1],
                                                                                    Xdmf_NumberOfComponents))

            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_t"+str(tCount)
                ar.create_dataset_sync(name+"_t"+str(tCount),
                                       offsets = self.mesh.globalMesh.elementOffsets_subdomain_owned*interpolationValues.shape[1],
                                       data = tmp)
            else:
                assert False, "global_sync  not  implemented for text  heavy data"
        else:
            Xdmf_NodesGlobal = interpolationValues.shape[0]*interpolationValues.shape[1]
            Xdmf_NumberOfComponents = interpolationValues.shape[2]
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i %i" % (Xdmf_NodesGlobal,3)})#force 3d vector since points 3d
            #mwf brute force
            tmp = numpy.zeros((Xdmf_NodesGlobal,3),'d')
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(interpolationValues.flat,(Xdmf_NodesGlobal,Xdmf_NumberOfComponents))

            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_p"+str(ar.rank)+"_t"+str(tCount)
                ar.create_dataset_async(name+"_p"+str(ar.rank)+"_t"+str(tCount), data = tmp)
            else:
                numpy.savetxt(ar.textDataDir+"/"+name+str(tCount)+".txt",tmp)
                SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+name+str(tCount)+".txt"})


    def writeMeshXdmf_DGP0(self,ar,mesh,spaceDim,
                           t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        as a first cut, archive piecewise constant space
        """

        #write out basic geometry if not already done?
        meshSpaceSuffix = "Spatial_Domain"
        mesh.writeMeshXdmf(ar,meshSpaceSuffix,t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that a constant per element
        spaceSuffix = "_dgp0"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if ar.global_sync:
            if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
                #mwf hack
                #allow for other types of topologies if the mesh has specified one
                if 'elementTopologyName' in dir(mesh):
                    Xdmf_ElementTopology = mesh.elementTopologyName
                else:
                    if spaceDim == 1:
                        Xdmf_ElementTopology = "Polyline"
                    elif spaceDim == 2:
                        Xdmf_ElementTopology = "Triangle"
                    elif spaceDim == 3:
                        Xdmf_ElementTopology = "Tetrahedron"
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"{0:e}".format(t),"Name":"{0:d}".format(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":"{0:d}".format(mesh.nElements_global)})
                #mwf hack, allow for a mixed element mesh
                if mesh.nNodes_element is None:
                    assert 'xdmf_topology' in dir(mesh)
                    elements = SubElement(topology,"DataItem",
                                          {"Format":ar.dataItemFormat,
                                           "DataType":"Int",
                                           "Dimensions":"{0:d}".format(len(self.xdmf_topology))})
                else:
                    elements    = SubElement(topology,"DataItem",
                                             {"Format":ar.dataItemFormat,
                                              "DataType":"Int",
                                              "Dimensions":"{0:d} {1:d}".format(mesh.nElements_global,mesh.nNodes_element)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"{0:d} {1:d}".format(mesh.nNodes_global,3)})
                #just reuse spatial mesh entries
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+meshSpaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+meshSpaceSuffix+str(tCount)
                else:
                    assert False, "global_sync not implemented  for text heavy data"
                #hdfile
            #need to write a grid
        else:
            if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
                #mwf hack
                #allow for other types of topologies if the mesh has specified one
                if 'elementTopologyName' in dir(mesh):
                    Xdmf_ElementTopology = mesh.elementTopologyName
                else:
                    if spaceDim == 1:
                        Xdmf_ElementTopology = "Polyline"
                    elif spaceDim == 2:
                        Xdmf_ElementTopology = "Triangle"
                    elif spaceDim == 3:
                        Xdmf_ElementTopology = "Tetrahedron"
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(mesh.nElements_global)})
                #mwf hack, allow for a mixed element mesh
                if mesh.nNodes_element is None:
                    assert 'xdmf_topology' in dir(mesh)
                    elements = SubElement(topology,"DataItem",
                                          {"Format":ar.dataItemFormat,
                                           "DataType":"Int",
                                           "Dimensions":"%i" % len(self.xdmf_topology)})
                else:
                    elements    = SubElement(topology,"DataItem",
                                             {"Format":ar.dataItemFormat,
                                              "DataType":"Int",
                                              "Dimensions":"%i %i" % (mesh.nElements_global,mesh.nNodes_element)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (mesh.nNodes_global,3)})
                #just reuse spatial mesh entries
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+meshSpaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+meshSpaceSuffix+str(tCount)
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+meshSpaceSuffix+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+meshSpaceSuffix+".txt"})
                #hdfile
            #need to write a grid
        return self.arGrid
    #def
    def writeFunctionXdmf_DGP0(self,ar,u,tCount=0,init=True):
        name = u.name.replace(' ','_')
        mesh = u.femSpace.elementMaps.mesh
        if ar.global_sync:
            ar.write_field(self.arGrid, name, u.dof, tCount, center="Cell",
                           dimensions=[mesh.globalMesh.nElements_global],
                           sync_offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                           sync_data=u.dof[:mesh.nElements_owned])
        else:
            ar.write_field(self.arGrid, name, u.dof, tCount, center="Cell",
                           dimensions=[mesh.nElements_global])

    def writeVectorFunctionXdmf_DGP0(self,ar,uList,components,vectorName,tCount=0,init=True):
        #this referred to a bare `u` that is not a parameter of this
        #function, so every call raised NameError -- including in the
        #default global_sync configuration. The mesh comes from the first
        #component, matching u_dof below.
        mesh = uList[components[0]].femSpace.elementMaps.mesh
        u_dof = uList[components[0]].dof
        if len(components) < 2:
            v_dof = numpy.zeros(u_dof.shape,dtype='d')
        else:
            v_dof = uList[components[1]].dof
        if len(components) < 3:
            w_dof = numpy.zeros(u_dof.shape,dtype='d')
        else:
            w_dof = uList[components[2]].dof
        velocity = numpy.column_stack((u_dof,v_dof,w_dof))

        if ar.global_sync:
            ar.write_field(self.arGrid, vectorName, velocity, tCount,
                           center="Cell", rank="Vector",
                           dimensions=[mesh.globalMesh.nElements_global,3],
                           sync_offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                           sync_data=velocity[:mesh.nElements_owned])
        else:
            #the non-HDF path had no else branch at all, so with
            #--useTextArchive this Attribute got a DataItem containing no
            #data reference. write_field writes a sidecar instead.
            ar.write_field(self.arGrid, vectorName, velocity, tCount,
                           center="Cell", rank="Vector",
                           dimensions=[mesh.nElements_global,3])

    def writeMeshXdmf_P1Bubble(self,ar,mesh,spaceDim,dofMap,t=0.0,
                               init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        represent P1 bubble space using just vertices for now
        """
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        spaceSuffix = "_c0p1_Bubble%s" % tCount
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Polyline"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Triangle"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tetrahedron"
            Xdmf_NodesPerElement = spaceDim+1 #just handle vertices for now
            Xdmf_NumberOfNodes   = mesh.nNodes_global
            if  ar.global_sync:
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfNodes,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #c0p1 mapping for now
                        ar.create_dataset_sync('elements'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = mesh.elementNodesArray)
                        ar.create_dataset_sync('nodes'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.nodeOffsets_subdomain_owned,
                                               data = mesh.nodeArray)
                else:
                    assert False, "global_sync not implemented for text heavy data"
            else:
                Xdmf_NumberOfElements= mesh.nElements_global
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Precision":"8",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfNodes,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #c0p1 mapping for now
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = mesh.elementNodesArray)
                        ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = mesh.nodeArray)
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",mesh.elementNodesArray,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",mesh.nodeArray)

                    #
                #hdfile
            #need to write a grid
        return self.arGrid
    #def
    def writeFunctionXdmf_P1Bubble(self,ar,u,tCount=0,init=True):
        #this read ar.sync_global, which does not exist -- the attribute
        #is global_sync -- so every call raised AttributeError
        if ar.global_sync:
            #just write out nodal part right now
            Xdmf_NumberOfElements = u.femSpace.mesh.globalMesh.nElements_global
            Xdmf_NumberOfNodes    = u.femSpace.mesh.globalMesh.nNodes_global
            name = u.name.replace(' ','_')

            #if writing as dgp1
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Scalar",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i" % (Xdmf_NumberOfNodes,)})
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_t"+str(tCount)
                ar.create_dataset_sync(name+"_t"+str(tCount),
                                       offsets = u.femSpace.mesh.nodeOffsets_subdomain_owned,
                                       data = u.dof[0:u.femSpace.mesh.nNodes_owned])
            else:
                assert False, "global_sync not implemented for text heavy data"
        else:
            #just write out nodal part right now
            Xdmf_NumberOfElements = u.femSpace.mesh.nElements_global
            Xdmf_NumberOfNodes    = u.femSpace.mesh.nNodes_global
            name = u.name.replace(' ','_')

            #if writing as dgp1
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Scalar",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i" % (Xdmf_NumberOfNodes,)})
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_p"+str(ar.rank)+"_t"+str(tCount)
                ar.create_dataset_async(name+"_p"+str(ar.rank)+"_t"+str(tCount), data = u.dof[0:Xdmf_NumberOfNodes])
            else:
                numpy.savetxt(ar.textDataDir+"/"+name+str(tCount)+".txt",u.dof[0:Xdmf_NumberOfNodes])
                SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+name+str(tCount)+".txt"})

    def writeFunctionXdmf_C0P2Lagrange(self,ar,u,tCount=0,init=True):
        #the text branch here read `assert Fasle` -- a typo for False, so it
        #raised NameError rather than the intended AssertionError.
        #write_field raises the assertion itself.
        if ar.global_sync:
            owned = u.femSpace.dofMap.dof_offsets_subdomain_owned
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dimensions=[u.femSpace.dofMap.nDOF_all_processes],
                           sync_offsets=owned,
                           sync_data=u.dof[:(owned[ar.rank+1] - owned[ar.rank])])
        else:
            #Dimensions comes from the array rather than u.nDOF_global: the
            #DataItem describes exactly what is written, and the two differ
            #when the caller passed a foreign array through the residual
            #adapter (phi_s is a vertex field, not a DOF vector of this space)
            ar.write_field(self.arGrid, u.name, u.dof, tCount)

    def writeVectorFunctionXdmf_P1Bubble(self,ar,uList,components,vectorName,spaceSuffix,tCount=0,init=True):
        if ar.global_sync:
            nDOF_global = uList[components[0]].femSpace.mesh.globalMesh.nNodes_global
            nDOF_local = (uList[components[0]].femSpace.mesh.globalMesh.nodeOffsets_subdomain_owned[ar.rank+1] -
                          uList[components[0]].femSpace.mesh.globalMesh.nodeOffsets_subdomain_owned[ar.rank])
            attribute = SubElement(self.arGrid,"Attribute",{"Name":vectorName,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i %i" % (nDOF_global,3)})
            u_dof = uList[components[0]].dof[0:nDOF_local]
            if len(components) < 2:
                v_dof = numpy.zeros((nDOF_global,),dtype='d')
            else:
                v_dof = uList[components[1]].dof[0:nDOF_local]
            if len(components) < 3:
                w_dof = numpy.zeros((nDOF_global,),dtype='d')
            else:
                w_dof = uList[components[2]].dof[0:nDOF_local]
            velocity = numpy.column_stack((u_dof,v_dof,w_dof))
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+vectorName+"_t"+str(tCount)
                ar.create_dataset_sync(vectorName+"_t"+str(tCount),
                                       offsets = uList[components[0]].femSpace.mesh.globalMesh.nodeOffsets_subdomain_owned,
                                       data = velocity)
        else:
            nDOF_global = uList[components[0]].femSpace.mesh.nNodes_global
            attribute = SubElement(self.arGrid,"Attribute",{"Name":vectorName,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Dimensions":"%i %i" % (nDOF_global,3)})
            u_dof = uList[components[0]].dof[0:nDOF_global]
            if len(components) < 2:
                v_dof = numpy.zeros((nDOF_global,),dtype='d')
            else:
                v_dof = uList[components[1]].dof[0:nDOF_global]
            if len(components) < 3:
                w_dof = numpy.zeros((nDOF_global,),dtype='d')
            else:
                w_dof = uList[components[2]].dof[0:nDOF_global]
            velocity = numpy.column_stack((u_dof,v_dof,w_dof))
            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+vectorName+"_p"+str(ar.rank)+"_t"+str(tCount)
                ar.create_dataset_async(vectorName+"_p"+str(ar.rank)+"_t"+str(tCount), data = velocity)

    def writeMeshXdmf_particles(self,ar,mesh,spaceDim,x,t=0.0,
                                init=False,meshChanged=False,arGrid=None,tCount=0,
                                spaceSuffix = "_particles"):
        """
        write out arbitrary set of points on a mesh
        """
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that is a collection of points per element

        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)
        nPoints = numpy.cumprod(x.shape)[-2]
        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            Xdmf_NodesPerElement = 1
            #the dataset names carry the rank before the space suffix while the
            #sidecar names omit it; both predate write_field's convention
            elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
            nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
            elements_stem    = 'elements'+spaceSuffix+str(tCount)
            nodes_stem       = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid, self.arTime = ar.write_grid(self.arGridCollection,
                                                     gridName, t, tCount)
            ar.write_topology(self.arGrid, "Polyvertex", nPoints,
                              [nPoints, Xdmf_NodesPerElement],
                              elements_dataset, elements_stem,
                              nodes_per_element=Xdmf_NodesPerElement)
            ar.write_geometry(self.arGrid, [nPoints, 3],
                              nodes_dataset, nodes_stem)

            #the references above are written every pass; the arrays only when
            #the mesh actually changed
            if init or meshChanged:
                q_l2g = numpy.arange(nPoints*Xdmf_NodesPerElement,
                                     dtype='i').reshape((nPoints,Xdmf_NodesPerElement))
                if ar.hdfFile is not None:
                    ar.create_dataset_async(elements_dataset, data = q_l2g)
                    ar.create_dataset_async(nodes_dataset, data = x.flat[:])
                else:
                    numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",q_l2g,fmt='%d')
                    numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",x.flat[:])
        return self.arGrid
    #def
    def writeScalarXdmf_particles(self,ar,u,name,tCount=0,init=True):
        nPoints = numpy.cumprod(u.shape)[-1]
        ar.write_field(self.arGrid, name, u.flat[:], tCount,
                       dimensions=[nPoints])

    def writeVectorXdmf_particles(self,ar,u,name,tCount=0,init=True):
        nPoints = numpy.cumprod(u.shape)[-2]
        Xdmf_NumberOfComponents = u.shape[-1]
        #force a 3-component vector since the points are 3D
        tmp = numpy.zeros((nPoints,3),'d')
        tmp[:,:Xdmf_NumberOfComponents] = numpy.reshape(
            u.flat,(nPoints,Xdmf_NumberOfComponents))
        ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                       dimensions=[nPoints,3])

    def writeMeshXdmf_LowestOrderMixed(self,ar,mesh,spaceDim,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0,
                                       spaceSuffix = "_RT0"):
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches RT0 velocity as dgp1 lagrange
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self.arGrid is None or self.arTime.get('Value') != "{0:e}".format(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Polyline"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Triangle"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tetrahedron"
            Xdmf_NodesPerElement = spaceDim+1
            if ar.global_sync:
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements*Xdmf_NodesPerElement,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #simple dg l2g mapping
                        dg_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((mesh.nElements_global,
                                                                                                             Xdmf_NodesPerElement))
                        ar.create_dataset_sync('elements'+spaceSuffix+str(tCount),
                                               offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                               data = dg_l2g)
                        
                    dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray[:mesh.nElements_owned]],(mesh.nElements_owned*Xdmf_NodesPerElement,3))
                    ar.create_dataset_sync('nodes'+spaceSuffix+str(tCount),
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned*Xdmf_NodesPerElement,
                                           data = dgnodes)
                else:
                    assert False, "global_sync not implemented for text heavy data"
            else:
                Xdmf_NumberOfElements= mesh.nElements_global
                self.arGrid = SubElement(self.arGridCollection,"Grid",{"Name":gridName,"GridType":"Uniform"})
                self.arTime = SubElement(self.arGrid,"Time",{"Value":"%e" % (t,),"Name":str(tCount)})
                topology    = SubElement(self.arGrid,"Topology",
                                         {"Type":Xdmf_ElementTopology,
                                          "NumberOfElements":str(Xdmf_NumberOfElements)})
                elements    = SubElement(topology,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Int",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements,Xdmf_NodesPerElement)})
                geometry    = SubElement(self.arGrid,"Geometry",{"Type":"XYZ"})
                nodes       = SubElement(geometry,"DataItem",
                                         {"Format":ar.dataItemFormat,
                                          "DataType":"Float",
                                          "Dimensions":"%i %i" % (Xdmf_NumberOfElements*Xdmf_NodesPerElement,3)})
                if ar.hdfFile is not None:
                    elements.text = ar.hdfFilename+":/elements"+str(ar.rank)+spaceSuffix+str(tCount)
                    nodes.text    = ar.hdfFilename+":/nodes"+str(ar.rank)+spaceSuffix+str(tCount)
                    if init or meshChanged:
                        #simple dg l2g mapping
                        dg_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        ar.create_dataset_async('elements'+str(ar.rank)+spaceSuffix+str(tCount), data = dg_l2g)
                    
                    dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray],(Xdmf_NumberOfElements*Xdmf_NodesPerElement,3))
                    ar.create_dataset_async('nodes'+str(ar.rank)+spaceSuffix+str(tCount), data = dgnodes)
                else:
                    SubElement(elements,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt"})
                    SubElement(nodes,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt"})
                    if init or meshChanged:
                        dg_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype='i').reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                        numpy.savetxt(ar.textDataDir+"/elements"+spaceSuffix+str(tCount)+".txt",dg_l2g,fmt='%d')

                        dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray],(Xdmf_NumberOfElements*Xdmf_NodesPerElement,3))
                        numpy.savetxt(ar.textDataDir+"/nodes"+spaceSuffix+str(tCount)+".txt",dgnodes)

                    #
                #hdfile
            #need to write a grid
        return self.arGrid
    #def
    def writeVectorFunctionXdmf_LowestOrderMixed(self,ar,u,tCount=0,init=True,spaceSuffix="_RT0",name="velocity"):
        if ar.global_sync:
            Xdmf_NodesGlobal = self.mesh.globalMesh.nElements_global*u.shape[1]
            Xdmf_NumberOfComponents = u.shape[2]
            Xdmf_StorageDim = 3

            #if writing as dgp1
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i %i" % (Xdmf_NodesGlobal,Xdmf_StorageDim)})#force 3d vector since points 3d
            tmp = numpy.zeros((u.shape[0]*u.shape[1],Xdmf_StorageDim),'d')
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(u.flat,(u.shape[0]*u.shape[1],Xdmf_NumberOfComponents))

            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_t"+str(tCount)
                ar.create_dataset_async(name+"_t"+str(tCount), data = tmp)
            else:
                assert False, "global_sync not implemented for text heavy data"
        else:
            Xdmf_NodesGlobal = u.shape[0]*u.shape[1]
            Xdmf_NumberOfComponents = u.shape[2]
            Xdmf_StorageDim = 3

            #if writing as dgp1
            attribute = SubElement(self.arGrid,"Attribute",{"Name":name,
                                                            "AttributeType":"Vector",
                                                            "Center":"Node"})
            values    = SubElement(attribute,"DataItem",
                                   {"Format":ar.dataItemFormat,
                                    "DataType":"Float",
                                    "Precision":"8",
                                    "Dimensions":"%i %i" % (Xdmf_NodesGlobal,Xdmf_StorageDim)})#force 3d vector since points 3d
            tmp = numpy.zeros((Xdmf_NodesGlobal,Xdmf_StorageDim),'d')
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(u.flat,(Xdmf_NodesGlobal,Xdmf_NumberOfComponents))

            if ar.hdfFile is not None:
                values.text = ar.hdfFilename+":/"+name+"_p"+str(ar.rank)+"_t"+str(tCount)
                ar.create_dataset_async(name+"_p"+str(ar.rank)+"_t"+str(tCount), data = tmp)
            else:
                numpy.savetxt(ar.textDataDir+"/"+name+str(tCount)+".txt",tmp)
                SubElement(values,"xi:include",{"parse":"text","href":"./"+ar.textDataDir+"/"+name+str(tCount)+".txt"})