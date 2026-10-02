reinitialize
load "D:/2026-2027_autumn/bioAI_competition/nav18_full_pipeline/outputs/receptors_20261002/nav15/prepared.pdb", receptor
load "D:/2026-2027_autumn/bioAI_competition/nav18_full_pipeline/results/nav15_conformer_redocking_20261002/structures/nav15_qdn_protonated_experimental.pdb", experimental
load "D:/2026-2027_autumn/bioAI_competition/nav18_full_pipeline/results/nav15_conformer_redocking_20261002/structures/nav15_qdn_protonated_pore20_lowest_score_top1.pdb", lowest_score_top1
load "D:/2026-2027_autumn/bioAI_competition/nav18_full_pipeline/results/nav15_conformer_redocking_20261002/structures/nav15_qdn_protonated_pore20_closest_experimental_diagnostic.pdb", closest_experimental_diagnostic
hide everything
show cartoon, receptor
set cartoon_transparency, 0.65
color gray70, receptor
show sticks, experimental or lowest_score_top1 or closest_experimental_diagnostic
color green, experimental
color red, lowest_score_top1
color cyan, closest_experimental_diagnostic
select pocket, byres (receptor within 5 of experimental)
show sticks, pocket
zoom experimental or lowest_score_top1 or closest_experimental_diagnostic, 5
