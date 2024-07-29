# SOFAPython3_DynamicLiverModel
Computational Dynamic Liver Model with SOFA Framework
During my internship I have to create an average liver model that can react to external forces in a realistic way. My idea has been divided into several operational steps:
1) selection of a database of N target livers and 1 template liver
2) rigid registration of the template for each target followed by a non-rigid registration using the "Thin Plate Spline" method
3) calculation of the "Principal Component Analysis" to reduce the size of the livers to their eigenvalues ​​and eigenvectors
4) removal of outliers
5) with the remaining livers, use SOFA framework in order to calculate a stiffness matrix for each liver
6) perform an average stiffness matrix that will regulate the behavior of my average liver

I am having problems writing the code on SOFA Framework, in particular:
1) build a liver that behaves inside the volume as TetrahedralCorotationalFEMForceField and behaves on the surface as TriangularBendingSprings i.e. as the Glisson capsule.
2) be able to apply 3 generic forces:
1 compression
1 traction
1 cut
binding the entire posterior part of the liver.
