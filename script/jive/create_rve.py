import sys
import math
import gmsh

from gmshModel.Model import SimpleCubicCell

# Input parameters.

a = None      # cube     size (length)   [mm]
r = 0.005/2.  # particle size (radius) [mm]
volInc = (4.0 / 3.0) * math.pi * (r**3)
volFracs = [10,20,30] # volume fraction [%]

"""
unitario

a=1
volTot = 1

dentro do for
volInc = volTot * (volFrac / 100.0)
r = ((3.0 * volInc) / (4.0 * math.pi)) ** (1.0 / 3.0)
"""
for volFrac in volFracs:

    # 2. Descobre qual deve ser o volume total do cubo para respeitar os 10%
    volTot = volInc / (volFrac / 100.0)

    a = volTot ** (1.0 / 3.0)
    mesh_size = 0.1*a

    print('--- PARÂMETROS DA CÉLULA UNITÁRIA ---')
    print(f'Raio (r)       = {r:.5e} mm')
    print(f'Lado do Cubo (a) = {a:.5e} mm')
    print(f'Volume Esfera  = {volInc:.5e}')
    print(f'Volume Cubo    = {volTot:.5e}')
    print(f'Fração Vol.    = {100. * volInc / volTot:.8f} %')

    print('\n==================================================================================\n')

    # Initialization of the unit cell

    initParameters={                                                                # save all possible parameters in one dict to facilitate the method call
        "numberCells": [1,1,1],                                                     # generate 1 unit cell in every spatial direction
        "radius": r,                                                                # set the inclusion radius to 'r'
        "inclusionType": "Sphere",                                                  # define inclusionType as "Sphere"
        "size": [a, a, a],
        "origin": [0., 0., 0.],                                                     # set cell origin to [0,0,0]
        "periodicityFlags": [1, 1, 1],                                              # define all axis directions as periodic
        "domainGroup": "domain",                                                    # NOTE: it seems like SimpleCubicCell does not take into account other name
        "inclusionGroup": "inclusions",                                             # use "inclusions" as name for the inclusionGroup
        "gmshConfigChanges": {"General.Terminal": 0,                                # deactivate console output by default (only activated for mesh generation)
                              "Mesh.MshFileVersion": 2.2,                           # mesh file format (legacy)
    			                "Mesh.SecondOrderLinear": 1,
                              "Mesh.ElementOrder": 2,
                              "Mesh.CharacteristicLengthExtendFromBoundary": 0,     # do not calculate mesh sizes from the boundary by default (since mesh sizes are specified by fields)
        }
    }
    testCell=SimpleCubicCell(**initParameters)

    # Gmsh model generation

    # NOTE: in this step, 'boundary' node group has been removed. If the user
    #       needs such group, just use 
    #       'testCell.createGmshModel(**modelingParameters)' or 
    #       'testCell.createGmshModel()'

    # define geometric objects and add them to the Gmsh model
    testCell.defineGeometricObjects()
    testCell.addGeometricObjectsToGmshModel()

    # define boolean operations and add them to the Gmsh mode
    testCell.defineBooleanOperations()                                          
    testCell.performBooleanOperationsForGmshModel()                            

    # define DEFAULT physical groups
    testCell.definePhysicalGroups()                                             

    # ... and then, remove 'boundary' group
    # NOTE: this is necessary for better reading task of JIVE's 
    #       module 'GmshInputModule'
    del testCell.groups['boundary']
    testCell.physicalGroups.pop()

    # define Gmsh PhysicalGroup without names
    # NOTE: in the API's function 'testCell.createGmshModel()', it is done
    #       by means of 'testCell.addPhysicalGroupsToGmshModel()'; however,
    #       'boundary' group

    for physGrp in testCell.physicalGroups:
        grpDim=physGrp["dimension"] 
        grpName=physGrp["group"] 
        grpNumber=physGrp["physicalNumber"]
        grpEntIDs=testCell.getIDsFromTags(testCell.groups[grpName])

        #print(grpDim, grpEntIDs, grpNumber, grpName)
        testCell.gmshAPI.addPhysicalGroup(grpDim, grpEntIDs, grpNumber)

    testCell.setupPeriodicity()

    # Gmsh mesh creation

    meshingParameters={                                                             # save all possible parameters in one dict to facilitate the method call
        "threads": None,                                                            # do not activate parallel meshing by default
        "refinementOptions": {"maxMeshSize": mesh_size,                                # automatically calculate maximum mesh size with built-in method
                              "inclusionRefinement": True,                          # flag to indicate active refinement of inclusions
                              "interInclusionRefinement": True,                     # flag to indicate active refinement of space between inclusions (inter-inclusion refinement)
                              "elementsPerCircumference": 22,                       # use 18 elements per inclusion circumference for inclusion refinement
                              "elementsBetweenInclusions": 3,                       # ensure 3 elements between close inclusions for inter-inclusion refinement
                              "inclusionRefinementWidth": 3,                        # use a relative (to inclusion radius) refinement width of 3 for inclusion refinement
                              "transitionElements": "auto",                         # automatically calculate number of transitioning elements (elements in which tanh function jumps from h_min to h_max) for inter-inclusion refinement
                              "aspectRatio": 1.5                                    # aspect ratio for inter-inclusion refinement: ratio of refinement in inclusion distance and perpendicular directions
        }
    }
    testCell.createMesh(**meshingParameters)

    # Save resulting mesh to file


    testCell.saveMesh(f"rve_vf{volFrac}.msh2")


    # Show resulting mesh
    # To check the generated mesh, the result can also be visualized using built-in
    # methods.
    #testCell.visualizeMesh()


    # Close Gmsh model
    # For a proper closing of the Gmsh-Python-API, the API has to be finalized. This
    # can be achieved by calling the close() method of the model
    testCell.close()
