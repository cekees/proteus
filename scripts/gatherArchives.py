#!/usr/bin/env python
import numpy
import os
from xml.etree.ElementTree import *
#rather than import proteus
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

def timeStepGrids(tree, collectionName="Mesh Spatial_Domain"):
    """The per-timestep Grid elements of a temporal collection.

    Replaces ``tree.getroot()[-1][-1]``, which took the root's last child
    as the Domain, that Domain's last child as the collection, and then
    indexed into it. An <Information> element ahead of <Domain>, or a
    second grid collection for a higher-order space -- which a real
    proteus archive commonly has -- silently moved either position.
    """
    domain = tree.getroot().find("Domain")
    assert domain is not None, "no <Domain> in the XDMF document"
    collections = domain.findall("Grid")
    assert collections, "no <Grid> under <Domain>"
    for collection in collections:
        if collectionName in collection.attrib.get("Name", ""):
            return collection.findall("Grid")
    return collections[0].findall("Grid")


def asXML(elem):
    """Serialize an element as text.

    ``tostring`` returns bytes by default, and these files are opened in
    text mode, so every write raised
    ``TypeError: string argument expected, got 'bytes'``.
    """
    return tostring(elem, encoding="unicode")


def gatherXDMFfiles(size,filename,dataDir='.',addname="_all"):
    """
    in case archiving failed to collect results from various processors in parallel simulation

    size -- nprocessors

    """
    xmlFile = []; tree = []
    xmlFile.append(open(filename+str(0)+".xmf","r"))
    tree.append(ElementTree(file=xmlFile[0]))
    xmlFile[-1].close()
    Domain_all = tree[0].getroot().find("Domain")
    assert Domain_all is not None, "no <Domain> in %s0.xmf" % (filename,)
    for TemporalGridCollection in Domain_all.findall("Grid"):
        Grids = TemporalGridCollection.findall("Grid")
        del TemporalGridCollection[:]
        for Grid in Grids:
            SpatialCollection=SubElement(TemporalGridCollection,"Grid",{"GridType":"Collection",
                                                                        "CollectionType":"Spatial"})
            #move the Time out of the grid and onto the spatial collection,
            #found by tag rather than taken as the grid's first child
            time = Grid.find("Time")
            assert time is not None, "a grid has no <Time>"
            SpatialCollection.append(time)
            Grid.remove(time)
            SpatialCollection.append(Grid) #append Grid without Time
    for i in range(1,size):
        xmlFile.append(open(os.path.join(dataDir,filename+str(i)+".xmf"),"r"))
        tree.append(ElementTree(file=xmlFile[-1]))
        Domain=tree[-1].getroot().find("Domain")
        assert Domain is not None, "no <Domain> in %s%d.xmf" % (filename, i)
        for TemporalGridCollection,TemporalGridCollection_all in zip(
                Domain.findall("Grid"), Domain_all.findall("Grid")):
            SpatialGridCollections = TemporalGridCollection_all.findall('Grid')
            for Grid,Grid_all in zip(TemporalGridCollection.findall("Grid"),
                                     SpatialGridCollections):
                time = Grid.find("Time")
                if time is not None:
                    Grid.remove(time)
                Grid_all.append(Grid)
        xmlFile[-1].close()
    f = open(os.path.join(dataDir,filename+addname+str(size)+".xmf"),"w")
    indentXML(tree[0].getroot())
    #encoding="unicode" makes write() emit str; without it ElementTree
    #writes bytes and this text-mode file rejects them
    tree[0].write(f, encoding="unicode")
    f.close()

def gatherXDMFfilesOpt(size,filename,dataDir='.',addname="_all",nStepsOnly=None,stride=1):
    """
    in case archiving failed to collect results from various processors in parallel simulation

    size -- nprocessors

    """

    xmlFile = open(filename+str(0)+".xmf","r")
    tree = ElementTree(file=xmlFile)
    xmlFile.close()
    nSteps = len(timeStepGrids(tree))
    if nStepsOnly != None:
        nSteps = nStepsOnly
    print("nSteps",nSteps)
    #stepsToGather=[i*stride for i in range(nSteps/stride)]
    stepsToGather = list(range(0,nSteps,stride))
    fAll = open(os.path.join(dataDir,filename+addname+str(size)+".xmf"),"w")
    fAll.write(r"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid CollectionType="Temporal" GridType="Collection" Name="Mesh Spatial_Domain">
""")
    for tn  in stepsToGather:
        print("time step",tn)
        print("subdomain",0)
        fAll.write(r"""      <Grid CollectionType="Spatial" GridType="Collection">
        """)
        xmlFile = open(os.path.join(dataDir,filename+str(0)+".xmf"),"r")
        tree = ElementTree(file=xmlFile)
        xmlFile.close()
        Grid=timeStepGrids(tree)[tn]
        time = Grid.find("Time")
        assert time is not None, "step %d has no <Time>" % (tn,)
        fAll.write(asXML(time))
        Grid.remove(time)
        fAll.write(asXML(Grid))
        for i in range(1,size):
            print("subdomain",i)
            xmlFile = open(os.path.join(dataDir,filename+str(i)+".xmf"),"r")
            tree = ElementTree(file=xmlFile)
            xmlFile.close()
            Grid=timeStepGrids(tree)[tn]
            time = Grid.find("Time")
            if time is not None:
                Grid.remove(time)
            fAll.write(asXML(Grid))
        fAll.write(r"""      </Grid>
""")
    fAll.write(r"""    </Grid>
  </Domain>
</Xdmf>
""")
    fAll.close()


def gatherSplitTimeStepXDMFfilesOpt(size,filename,dataDir='.',addname="_all",nStepsOnly=None,stride=1):
    """
    in case archiving failed to collect results from various processors in parallel simulation

    size -- nprocessors

    """
    xmlFile = open(filename+str(0)+".xmf","r")
    tree = ElementTree(file=xmlFile)
    xmlFile.close()
    nSteps = len(timeStepGrids(tree))
    if nStepsOnly != None:
        nSteps = nStepsOnly
    print("nSteps",nSteps)
    stepsToGather=[i*stride for i in range(nSteps//stride)]
    for tn  in stepsToGather:
        fAll = open(os.path.join(dataDir,filename+"_t"+str(tn) + addname+str(size)+".xmf"),"w")
        fAll.write(r"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid CollectionType="Temporal" GridType="Collection" Name="Mesh Spatial_Domain">
""")

        print("time step",tn)
        print("subdomain",0)
        fAll.write(r"""      <Grid CollectionType="Spatial" GridType="Collection">
        """)
        xmlFile = open(os.path.join(dataDir,filename+str(0)+".xmf"),"r")
        tree = ElementTree(file=xmlFile)
        xmlFile.close()
        Grid=timeStepGrids(tree)[tn]
        time = Grid.find("Time")
        assert time is not None, "step %d has no <Time>" % (tn,)
        fAll.write(asXML(time))
        Grid.remove(time)
        fAll.write(asXML(Grid))
        for i in range(1,size):
            print("subdomain",i)
            xmlFile = open(os.path.join(dataDir,filename+str(i)+".xmf"),"r")
            tree = ElementTree(file=xmlFile)
            xmlFile.close()
            Grid=timeStepGrids(tree)[tn]
            time = Grid.find("Time")
            if time is not None:
                Grid.remove(time)
            fAll.write(asXML(Grid))
        fAll.write(r"""      </Grid>
""")
        fAll.write(r"""    </Grid>
  </Domain>
</Xdmf>
""")
        fAll.close()

if __name__ == '__main__':
    from optparse import OptionParser
    usage = ""
    parser = OptionParser(usage=usage)
    parser.add_option("-s","--size",
                      help="number of processors for run",
                      action="store",
                      type="int",
                      dest="size",
                      default=1)

    parser.add_option("-d","--dataDir",
                      help="location of xdmf files",
                      action="store",
                      type="string",
                      dest="dataDir",
                      default=".")

    parser.add_option("-f","--filebase",
                      help="base name for storage files",
                      action="store",
                      type="string",
                      dest="filebase",
                      default="simulation")

    parser.add_option("-a","--append",
                      help="what to append for combined file",
                      action="store",
                      type="string",

                      dest="append",
                      default="_all")
    parser.add_option("-n","--nSteps",
                      help="take first nSteps",
                      action="store",
                      type="int",
                      dest="nStepsOnly",
                      default=None)
    parser.add_option("-m","--mStride",
                      help="take every mStride-th step",
                      action="store",
                      type="int",
                      dest="mStride",
                      default=1)
    parser.add_option("-M","--lotsOfMemory",
                      help="don't worry about memory usage",
                      action="store_true",
                      dest="lotsOfMemory",
                      default=False)
    parser.add_option("-i","--splitTimeSteps",
                      help="don't worry about memory usage",
                      action="store_true",
                      dest="splitTimeSteps",
                      default=False)


    (opts,args) = parser.parse_args()

    if opts.lotsOfMemory:
        gatherXDMFfiles(opts.size,opts.filebase,opts.dataDir,opts.append)
    elif opts.splitTimeSteps:
        gatherSplitTimeStepXDMFfilesOpt(opts.size,opts.filebase,opts.dataDir,opts.append,opts.nStepsOnly,opts.mStride)
    else:
        gatherXDMFfilesOpt(opts.size,opts.filebase,opts.dataDir,opts.append,opts.nStepsOnly,opts.mStride)