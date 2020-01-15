all:
	gmsh -3 cube.geo -clscale 2 -o cube.msh -smooth 100
