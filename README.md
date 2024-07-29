# SOFAPython3_DynamicLiverModel
'''Computational Dynamic Liver Model with SOFA Framework
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
binding the entire posterior part of the liver.'''

# Required import for python
import Sofa
import numpy as np
from scipy import sparse
from scipy import linalg
import numpy as np
from matplotlib import pyplot as plt


# Choose in your script to activate or not the GUI
USE_GUI = True
exportCSV = True
showImage = False

def main():
    import SofaRuntime
    import Sofa.Gui

    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)

    if not USE_GUI:
        for iteration in range(10):
            Sofa.Simulation.animate(root, root.dt.value)
    else:
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()


def createScene(root):
    root.gravity=[0, -9.81, 0]
    root.dt=0.02

    root.addObject("RequiredPlugin", pluginName=[
        'Sofa.Component.Collision.Detection.Algorithm',
        'Sofa.Component.Collision.Detection.Intersection',
        'Sofa.Component.Collision.Geometry',
        'Sofa.Component.Collision.Response.Contact',
        'Sofa.Component.Constraint.Projective',
        'Sofa.Component.IO.Mesh',
        'Sofa.Component.LinearSolver.Iterative',
        'Sofa.Component.Mapping.Linear',
        'Sofa.Component.Mass',
        'Sofa.Component.ODESolver.Backward',
        'Sofa.Component.SolidMechanics.FEM.Elastic',
        'Sofa.Component.StateContainer',
        'Sofa.Component.Topology.Container.Dynamic',
        'Sofa.Component.Visual',
        'Sofa.GL.Component.Rendering3D',
        'Sofa.Component.SolidMechanics.Spring',
        'Sofa.Component.MechanicalLoad',
        'Sofa.Component.Topology.Container.Constant'
    ])

    root.addObject('DefaultAnimationLoop')

    root.addObject('VisualStyle', displayFlags="showCollisionModels")
    root.addObject('CollisionPipeline', name="CollisionPipeline")
    root.addObject('BruteForceBroadPhase', name="BroadPhase")
    root.addObject('BVHNarrowPhase', name="NarrowPhase")
    root.addObject('DefaultContactManager', name="CollisionResponse", response="PenalityContactForceField")
    root.addObject('DiscreteIntersection')

    root.addObject("MeshSTLLoader", name="LiverSurface", filename = r"C:\Users\monta\Desktop\Materie\Altro\Tirocinio\Liver_Database\3D_Slicer\liver_sofa\Liver_Target_2.stl")

    liver = root.addChild('Liver')
    liver.addObject('EulerImplicitSolver', name="cg_odesolver", rayleighStiffness="0.1", rayleighMass="0.1")
    liver.addObject('CGLinearSolver', name="linear_solver", iterations="25", tolerance="1e-09", threshold="1e-09")
    liver.addObject("MeshGmshLoader", name="meshLoader", filename = r"C:\Users\monta\Desktop\Materie\Altro\Tirocinio\Liver_Database\3D_Slicer\liver_gmsh\Liver_Target_2.msh")  
    liver.addObject('TetrahedronSetTopologyContainer', name="tetra", src="@meshLoader")
    liver.addObject('MechanicalObject', name="dofs", src="@meshLoader")
    liver.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d", name="TetraGeomAlgo")
    liver.addObject('DiagonalMass', name="Mass", massDensity="1.0")
    FEM = liver.addObject('TetrahedralCorotationalFEMForceField', template="Vec3d", name="TetraFEM", method="large", poissonRatio="0.48", youngModulus="3000", computeGlobalMatrix="0")
    
    # Glisson capsule with TriangularBendingSprings
    liverCapsule = liver.addChild('GlissonCapsule')
    liverCapsule.addObject('MeshSTLLoader', name='CapsuleMesh', filename = r"C:\Users\monta\Desktop\Materie\Altro\Tirocinio\Liver_Database\3D_Slicer\liver_segmentation\Liver_Target_2.stl")
    liverCapsule.addObject('MechanicalObject', name='dofsCapsule', src="@CapsuleMesh")
    liverCapsule.addObject('TriangleSetTopologyContainer', name="tri", src="@CapsuleMesh")
    liverCapsule.addObject('TriangleSetGeometryAlgorithms', template="Vec3d", name="TriGeomAlgo")
    liverCapsule.addObject('TriangularBendingSprings', template="Vec3d", name="BendingSprings", stiffness=10000, damping=0.2)
    liverCapsule.addObject('OglModel', name='visualCapsule', color='0.2 0.8 0.2 1.0', src='@CapsuleMesh')
    liverCapsule.addObject('BarycentricMapping', name="VisualMappingCapsule", input="@dofsCapsule", output="@visualCapsule")
    
    #visualization
    visualization = liver.addChild('visualization')
    visualization.addObject('OglModel', name='VisualModelParenchyma', color='0.8 0.2 0.2 1.0', src='@../meshLoader')
    visualization.addObject('BarycentricMapping', name="VisualMappingParenchyma", input="@../dofs", output="@VisualModelParenchyma")
    
    liver.addObject(MatrixAccessController('MatrixAccessor', name='matrixAccessor', force_field=FEM))

    return root

class MatrixAccessController(Sofa.Core.Controller):


    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.force_field = kwargs.get("force_field")

    def onAnimateEndEvent(self, event):
        stiffness_matrix = self.force_field.assembleKMatrix()

        print('====================================')
        print('Stiffness matrix')
        print('====================================')
        print('dtype: ' + str(stiffness_matrix.dtype))
        print('shape: ' + str(stiffness_matrix.shape))
        print('ndim: ' + str(stiffness_matrix.ndim))
        print('nnz: ' + str(stiffness_matrix.nnz))
        print('norm: ' + str(sparse.linalg.norm(stiffness_matrix)))

        if exportCSV:
            np.savetxt('stiffness.csv', stiffness_matrix.toarray(), delimiter=',')
        if showImage:
            plt.imshow(stiffness_matrix.toarray(), interpolation='nearest', cmap='gist_gray')
            plt.show(block=False)

# Function used only if this script is called from a python environment
if __name__ == '__main__':
    main()
