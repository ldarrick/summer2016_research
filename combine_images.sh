#! /bin/bash

dir=$(pwd)
outdir="$dir/CombinedImgs/img"
rm -rf "$dir/CombinedImgs/*.png"

for i in $(seq -f "%03g" 7 499)
do
  img1="$dir/Cell_Field_CellField_2D_XY_0/Cell_Field_CellField_2D_XY_0_$i.png"
  img2="$dir/CellularRho_ScalarFieldCellLevel_2D_XY_0/CellularRho_ScalarFieldCellLevel_2D_XY_0_$i.png"
  convert $img1 $img2 -background Black +append $outdir"_"$i".png"
done
