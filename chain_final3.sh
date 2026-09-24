#!/bin/bash
# Link the box-sequence masks (built at chunk 256, because 512 OOMs on ~1000
# frames) into the directory the battery reads, then launch it.
#
# The parent directory is named cs512 and two of its sequences were built at 256.
# That is deliberate -- the protocol is "one pass where the sequence fits, largest
# feasible chunk otherwise" -- and each sequence's own meta.json records its real
# chunk size, so provenance survives. Say so in the write-up rather than letting
# the directory name imply uniformity.
set -uo pipefail
D=~/data/mask_out
M2=$D/output_dyn_masks_precomputed_cs512_r518_st3_fs1_m6_otsu2_glob
SRC=$D/output_dyn_masks_precomputed_cs256_r518_st3_fs1_m6_otsu2_glob
for S in rgbd_bonn_removing_obstructing_box rgbd_bonn_placing_obstructing_box; do
  [ -d "$M2/$S" ] && continue
  if [ -d "$SRC/$S" ]; then ln -s "$SRC/$S" "$M2/$S" && echo "linked $S"
  else echo "MISSING $SRC/$S -- battery will refuse this sequence"; fi
done
for S in rgbd_bonn_balloon rgbd_bonn_synchronous2 \
         rgbd_bonn_removing_obstructing_box rgbd_bonn_placing_obstructing_box; do
  printf '%-40s %s masks\n' "$S" "$(ls $M2/$S/masks/*.png 2>/dev/null | wc -l)"
done
cd ~/DynamicReconstructionSplat && SERIAL=1 EVAL_DATE=final3 ./submit_final_battery.sh
