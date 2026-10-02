# Run from this directory: pymol compare.pml
reinitialize
load receptor.pdb, receptor
load reference.pdb, native
load lowest_vina_top1.pdb, top1
load closest_output_diagnostic_only.pdb, closest
hide everything
show cartoon, receptor
color gray80, receptor
show sticks, native or top1 or closest
color green, native
color orange, top1
color cyan, closest
show sticks, byres (receptor within 4 of native)
set cartoon_transparency, 0.7
zoom native or top1 or closest, 7
bg_color white
