#!/bin/bash
echo "fct"
rm -rf fct
parun celia_p.py celia_n.py -l 5 -D fct -C "num='fct' dt=0.0001"
tail -n 100 fct/celia_p.log
echo "implicit-fct"
rm -rf implicit-fct
parun celia_p.py celia_n.py -l 5 -D implicit-fct -C "num='implicit-fct' dt=0.0001"
tail -n 100 implicit-fct/celia_p.log
echo "low-order"
rm -rf low-order
parun celia_p.py celia_n.py -l 5 -D low-order -C "num='low-order' dt=0.0001"
tail -n 100 low-order/celia_p.log
echo "low-order-galerkin"
rm -rf low-order-galerkin
parun celia_p.py celia_n.py -l 5 -D low-order-galerkin -C "num='low-order-galerkin' dt=0.0001"
tail -n 100 low-order-galerkin/celia_p.log
echo "galerkin"
rm -rf galerkin
parun celia_p.py celia_n.py -l 5 -D galerkin -C "num='galerkin' dt=0.0001"
tail -n 100 galerkin/celia_p.log
echo "vms-galerkin"
rm -rf vms-galerkin
parun celia_p.py celia_n.py -l 5 -D vms-galerkin -C "num='vms-galerkin' dt=0.0001"
tail -n 100 vms-galerkin/celia_p.log
echo "vms-sc-galerkin"
rm -rf vms-sc-galerkin
parun celia_p.py celia_n.py -l 5 -D vms-sc-galerkin -C "num='vms-sc-galerkin' dt=0.0001"
tail -n 100 vms-sc-galerkin/celia_p.log
