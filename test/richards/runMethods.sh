#!/bin/bash
echo "fct"
rm -rf fct
parun re_vgm_sand_10m_1d_p.py re_vgm_sand_10m_1d_c0p1_n.py -l 5 -D fct -C "num='fct' dt=0.001"
tail -n 100 fct/re_vgm_sand_10m_1d_p.log
echo "low-order"
rm -rf low-order
parun re_vgm_sand_10m_1d_p.py re_vgm_sand_10m_1d_c0p1_n.py -l 5 -D low-order -C "num='low-order' dt=0.001"
tail -n 100 low-order/re_vgm_sand_10m_1d_p.log
echo "low-order-galerkin"
rm -rf low-order-galerkin
parun re_vgm_sand_10m_1d_p.py re_vgm_sand_10m_1d_c0p1_n.py -l 5 -D low-order-galerkin -C "num='low-order-galerkin' dt=0.001"
tail -n 100 low-order-galerkin/re_vgm_sand_10m_1d_p.log
#echo "galerkin"
#rm -rf galerkin
#parun re_vgm_sand_10m_1d_p.py re_vgm_sand_10m_1d_c0p1_n.py -l 5 -D galerkin -C "num='galerkin' dt=0.001"
#tail -n 100 galerkin/re_vgm_sand_10m_1d_p.log
#echo "vms-galerkin"
#rm -rf vms-galerkin
#parun re_vgm_sand_10m_1d_p.py re_vgm_sand_10m_1d_c0p1_n.py -l 5 -D vms-galerkin -C "num='vms-galerkin' dt=0.001"
#tail -n 100 vms-galerkin/re_vgm_sand_10m_1d_p.log
#echo "vms-sc-galerkin"
#rm -rf vms-sc-galerkin
#parun re_vgm_sand_10m_1d_p.py re_vgm_sand_10m_1d_c0p1_n.py -l 5 -D vms-sc-galerkin -C "num='vms-sc-galerkin' dt=0.001"
#tail -n 100 vms-sc-galerkin/re_vgm_sand_10m_1d_p.log