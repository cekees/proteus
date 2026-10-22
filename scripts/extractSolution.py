#!/usr/bin/env python
# Converted from PyTables to h5py. This script used tables.open_file,
# createArray and createGroup: the camelCase names were removed in PyTables
# 3.0, so every write raised AttributeError, and PyTables is not declared in
# any of proteus's environment files, so the import was an undeclared
# dependency besides. h5py is what the rest of proteus uses.
import h5py
import os
import sys

def splitH5all(basename1,basename2,size,start,finaltime,stride):

    print("==================")
    print("   Extracting")
    print("==================")

    for proc in range(0,size):
        print("Processor", proc)
        splitH5single(basename1,basename2,proc,start,finaltime,stride)

    print("==================")
    print("   Composing")
    print("==================")
    H5toXMF("solution",size,start,finaltime,stride)

def splitH5single(basename1,basename2,proc,start,finaltime,stride):

# Loop over entries and put in appropriate file
    filename=basename1+str(proc)+".h5"
    print(" Open:",filename)
    f1 = h5py.File(filename, "r")
    filename=basename2+str(proc)+".h5"
    print(" Open:",filename)
    f2 = h5py.File(filename, "r")

    print("   Step:", end=' ')

    for step in range(start,finaltime+1,stride):
        print(step, end=' ')
        sys.stdout.flush()

        filename="sol.p"+str(proc)+"."+str(step)+".h5"
        hdfFile = h5py.File(filename, "w")


        name =  "elementsSpatial_Domain"+str(step)
        hdfFile.create_dataset("elements", data=f1[name][:])

        name =  "nodesSpatial_Domain"+str(step)
        hdfFile.create_dataset("nodes", data=f1[name][:])

        name =  "u"+str(step)
        hdfFile.create_dataset("u", data=f1[name][:])

        name =  "v"+str(step)
        hdfFile.create_dataset("v", data=f1[name][:])

        name =  "w"+str(step)
        hdfFile.create_dataset("w", data=f1[name][:])

        name =  "p"+str(step)
        hdfFile.create_dataset("p", data=f1[name][:])

        name =  "phid"+str(step)
        hdfFile.create_dataset("phid", data=f2[name][:])

        hdfFile.close()

    f1.close()
    f2.close()

    print("finished")

def H5toXMF(basename,size,start,finaltime,stride):

# Open XMF files

    t1="  "
    t2=t1+t1
    t3=t2+t1
    t4=t3+t1
    t5=t4+t1

    XMFfile1 = open(basename+".xmf","w")
    XMFfile1.write('<?xml version="1.0" ?>'+"\n")
    XMFfile1.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'+"\n")
    XMFfile1.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">'+"\n")
    XMFfile1.write(t1 + '<Domain>'+"\n")
    XMFfile1.write(t1 + '<Grid GridType="Collection"   CollectionType="Temporal">'+"\n")

    string=""
    print("   Step:", end=' ')
    for step in range(start,finaltime+1,stride):
        print(step, end=' ')
        sys.stdout.flush()

        filename = basename+"."+str(step)+".h5"
        hdfFile = h5py.File(filename, "w")

        XMFfile2 = open(basename+"."+str(step)+".xmf","w")
        XMFfile2.write('<?xml version="1.0" ?>'+"\n")
        XMFfile2.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>'+"\n")
        XMFfile2.write('<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">'+"\n")
        XMFfile2.write(t1 + '<Domain>'+"\n")

        string = t2 + '<Grid GridType="Collection" CollectionType="Spatial">'+"\n"
        string = string + t3 + '<Time Value="'+str(step)+'" />'+"\n"

        for proc in range(0,size):
            group = hdfFile.create_group('p'+str(proc))

            solname="sol.p"+str(proc)+"."+str(step)+".h5"
            f1 = h5py.File(solname, "r")

            string = string + t3+'<Grid GridType="Uniform">'+"\n"

            data=f1["elements"][:]
            group.create_dataset("elements", data=data)

            string = string + t4 + '<Topology NumberOfElements="' +str(len(data))+ '" Type="Tetrahedron">'+"\n"
            string = string + t5 + '<DataItem DataType="Int" Dimensions="' +str(len(data))+ ' 4" Format="HDF">'+"\n"
            string = string + t5 + filename + ':/p'+str(proc)+'/elements'+"\n"
            string = string + t5 +'</DataItem>'+"\n"
            string = string + t4 + '</Topology>'+"\n"

            data=f1["nodes"][:]
            group.create_dataset("nodes", data=data)

            string = string + t4 + '<Geometry Type="XYZ">'+"\n"
            string = string + t5 + '<DataItem DataType="Float" Dimensions="' +str(len(data))+ ' 3" Format="HDF" Precision="8">' + "\n"
            string = string + t5 + filename + ':/p'+str(proc)+'/nodes'+"\n"
            string = string + t5 + '</DataItem>'+"\n"
            string = string + t4 + '</Geometry>'+"\n"

            data=f1["u"][:]
            group.create_dataset("u", data=data)

            string = string + t4 + '<Attribute AttributeType="Scalar" Center="Node" Name="u">'+"\n"
            string = string + t5 + '<DataItem DataType="Float" Dimensions="' +str(len(data))+ '" Format="HDF" Precision="8">' + "\n"
            string = string + t5 + filename + ':/p'+str(proc)+'/u'+"\n"
            string = string + t5 + '</DataItem>'+"\n"
            string = string + t4 + '</Attribute>'+"\n"

            data=f1["v"][:]
            group.create_dataset("v", data=data)

            string = string + t4 + '<Attribute AttributeType="Scalar" Center="Node" Name="v">'+"\n"
            string = string + t5 +'<DataItem DataType="Float" Dimensions="' +str(len(data))+ '" Format="HDF" Precision="8">' + "\n"
            string = string + t5 + filename + ':/p'+str(proc)+'/v'+"\n"
            string = string + t5 + '</DataItem>'+"\n"
            string = string + t4 + '</Attribute>'+"\n"

            data=f1["w"][:]
            group.create_dataset("w", data=data)

            string = string + t4 + '<Attribute AttributeType="Scalar" Center="Node" Name="w">'+"\n"
            string = string + t5 + '<DataItem DataType="Float" Dimensions="' +str(len(data))+ '" Format="HDF" Precision="8">' + "\n"
            string = string + t5 + filename + ':/p'+str(proc)+'/w'+"\n"
            string = string + t5 + '</DataItem>'+"\n"
            string = string + t4 + '</Attribute>'+"\n"

            data=f1["p"][:]
            group.create_dataset("p", data=data)

            string = string + t4 + '<Attribute AttributeType="Scalar" Center="Node" Name="p">'+"\n"
            string = string + t5 + '<DataItem DataType="Float" Dimensions="' +str(len(data))+ '" Format="HDF" Precision="8">' + "\n"
            string = string + t5 + filename + ':/p'+str(proc)+'/p'+"\n"
            string = string + t5 + '</DataItem>'+"\n"
            string = string + t4 + '</Attribute>'+"\n"

            data=f1["phid"][:]
            group.create_dataset("phid", data=data)

            string = string + t4 + '<Attribute AttributeType="Scalar" Center="Node" Name="phid">'+"\n"
            string = string + t5 + '<DataItem DataType="Float" Dimensions="' +str(len(data))+ '" Format="HDF" Precision="8">' + "\n"
            string = string + t5 + filename + ':/p'+str(proc)+'/phid'+"\n"
            string = string + t5 + '</DataItem>'+"\n"
            string = string + t4 + '</Attribute>'+"\n"

            string = string + t3+'</Grid>'+"\n"

            f1.close()
            os.remove(solname)

        string = string + t2 + '</Grid>'+"\n"

        XMFfile1.write(string)

        XMFfile2.write(string)
        XMFfile2.write(t1 + '</Domain>'+"\n")
        XMFfile2.write('</Xdmf>'+"\n")
        XMFfile2.close()

        hdfFile.close()



    XMFfile1.write(t1 + '</Grid>'+"\n")
    XMFfile1.write(t1 + '</Domain>'+"\n")
    XMFfile1.write('</Xdmf>'+"\n")
    XMFfile1.close()

if __name__ == '__main__':
    from optparse import OptionParser

    usage = ""
    parser = OptionParser(usage=usage)

    parser.add_option("-n","--size",
                      help="number of processors for run",
                      action="store",
                      type="int",
                      dest="size",
                      default=1)

    parser.add_option("-s","--stride",
                      help="stride for solution output",
                      action="store",
                      type="int",
                      dest="stride",
                      default=0)

    parser.add_option("-t","--finaltime",
                      help="finaltime",
                      action="store",
                      type="int",
                      dest="finaltime",
                      default=1000)

    parser.add_option("-f","--filebase_flow",
                      help="base name for storage files",
                      action="store",
                      type="string",
                      dest="filebase1",
                      default="twp_navier_stokes_p")

    parser.add_option("-p","--filebase_phi",
                      help="base name for storage files",
                      action="store",
                      type="string",
                      dest="filebase2",
                      default="redist_p")

    (opts,args) = parser.parse_args()

    start = 0
    if opts.stride == 0 :
        start = opts.finaltime
        opts.stride = 1
    if (opts.size >0) :
        splitH5all(opts.filebase1,opts.filebase2,opts.size,start,opts.finaltime,opts.stride)
    else :
        splitH5single(opts.filebase1,opts.filebase2,-opts.size,start,opts.finaltime,opts.stride)
