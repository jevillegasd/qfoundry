from scipy.constants import c, mu_0, epsilon_0
import numpy as np

import shapely
from meshwell.model import Model
from meshwell.polysurface import PolySurface
from femwell.coulomb import solve_coulomb
from femwell.visualization import plot_domains as plot_d
from femwell.visualization import plot_subdomain_boundaries as plot_sb
from skfem import Basis, ElementDG, ElementTriP0, Functional, adaptive_theta
from skfem.helpers import dot
from skfem.io.meshio import from_meshio
import matplotlib.pyplot as plt
# Think of migrating to use only SCIKITFEM (skfem) 

'''
Simulation of capacitances using the FEMWELL Module 
'''
units_mm = 1e-3
units_um = 1e-6
units_nm = 1e-9

class coplanar_capacitor:
    '''
    A coplanar capacitor
    Simulates a coplanar capacitor cross section. The final capacitance is calculated
    as the capacitance per length times the total length of the capacitor.
    '''
    def __init__(self, 
                 width:float,
                 spacing:float,
                 thickness:float = 1,
                 substrate_heigth:float = 550,
                 mesh_size:float=0.5,
                 padding:float = 50.,
                 dV:float=1,
                 dielectric_epsilon=11.7,
                 units = units_um
                 ):   
        self.w = width
        self.s = spacing
        self.t = thickness
        self.h = substrate_heigth
        self.mesh_size = mesh_size
        self.dV = dV
        self.epsr = dielectric_epsilon
        self.basis_u = None
        self.u = None 
        self.basis_epsilon = None
        self.epsilon = None
        self.units = units
        self.padding= padding

        mesh, polygons = __coplanar_capacitor_mesh__(
            width = width,
            separation=spacing,
            thickness=thickness,
            substrate_heigth = substrate_heigth,
            mesh_size = self.mesh_size,
            padding = self.padding
        )
        self.polygons = polygons
        self.mesh = mesh
    
    def refine(self):
        self.mesh = self.mesh.refined(1)

    def capacitance(self):
        from scipy.constants import epsilon_0 as eps0
        if self.basis_u == None:
            self.basis_u, self.u, self.basis_epsilon, self.epsilon = self.potential()

        @Functional(dtype=complex)
        def W(w):
            return 0.5 * w["epsilon"] * dot(w["u"].grad, w["u"].grad)
    
        C = (2* W.assemble(
                self.basis_u,
                epsilon=self.basis_epsilon.interpolate(self.epsilon),
                u=self.basis_u.interpolate(self.u),
            )
            / self.dV**2 * eps0 *self.units  # Returns in F/m
        )
        return C

    def potential(self):
        basis_epsilon = Basis(self.mesh, ElementTriP0())
        epsilon = basis_epsilon.ones()

        epsilon[basis_epsilon.get_dofs(elements=("dielectric"))] = self.epsr

        basis_u, u = solve_coulomb(
            basis_epsilon,
            epsilon,
            {
                "left_plate___dielectric": self.dV/2,
                "left_plate___air": self.dV/2,
                "rigth_plate___dielectric": -self.dV/2,
                "rigth_plate___air": -self.dV/2,
            },
        )
        return basis_u, u, basis_epsilon, epsilon

    def plot_potential(self, ax = None): #This is generating very large images
        if ax == None:
            fig, ax = plt.subplots()
        self.basis_u, self.u, self.basis_epsilon, self.epsilon = self.potential()
        for subdomain in self.basis_epsilon.mesh.subdomains.keys() - {"gmsh:bounding_entities"}:
            self.basis_epsilon.mesh.restrict(subdomain).draw(ax=ax, boundaries_only=True)
        self.basis_u.plot(self.u, ax=ax, shading="flat", colorbar=True)
        return plt.show()

    def plot_polygons(self):
        pass

    def plot_mesh(self):
        self.mesh.draw().show()

    def plot_domains(self):
        return plot_d(self.mesh)
    
    def plot_subdomain_boundaries(self):
        return plot_sb(self.mesh)



# Auxiliary Functions
def __coplanar_capacitor_mesh__(
    width:float = 20,
    separation:float=20,
    thickness:float=1,
    substrate_heigth:float = 550,
    mesh_size:float = 0.5,
    padding:float = 50.
):
    left_plate_polygon = shapely.geometry.box(
        -width-separation/2, 0, -separation/2, thickness
    )
    right_plate_polygon = shapely.geometry.box(
        width+separation/2, 0, separation/2, thickness
    )
    intra_plate_polygon = shapely.geometry.box(
        -separation/2, 0, separation/2, thickness
    )

    substrate_polygon = shapely.geometry.box(
        -max(width*10,padding+separation/2+width), 0, max(padding+separation/2+width,padding), -substrate_heigth
    )
    capacitor_polygon = shapely.unary_union(
        [left_plate_polygon, intra_plate_polygon, right_plate_polygon]
    )
    metals_polygon = shapely.unary_union(
        [left_plate_polygon, right_plate_polygon]
    )
    outbound = capacitor_polygon.buffer(max((width/2+separation),padding),
                                        resolution=8)

    dielectric_polygon = shapely.intersection(substrate_polygon,outbound)
    air_polygon = outbound.difference(substrate_polygon).difference(metals_polygon)
    
    model = Model()

    left_plate = PolySurface(
        polygons=left_plate_polygon,
        model=model,
        physical_name="left_plate",
        mesh_bool=False,
        resolution={"resolution": 0.1, "DistMax": 2},
        mesh_order=1,
    )
    right_plate = PolySurface(
        polygons=right_plate_polygon,
        model=model,
        physical_name="rigth_plate",
        mesh_bool=False,
        resolution={"resolution": 0.1, "DistMax": 2},
        mesh_order=2,
    )
    dielectric = PolySurface(
        polygons=dielectric_polygon,
        model=model,
        physical_name="dielectric",
        mesh_bool=True,
        mesh_order=3,
    )
    air = PolySurface(
        polygons=air_polygon,
        model=model,
        physical_name="air",
        mesh_bool=True,
        mesh_order=4,
    )

    mesh = from_meshio(
        model.mesh(
            entities_list=[left_plate, right_plate, dielectric, air],
            filename="mesh.msh",
            default_characteristic_length= mesh_size,
            #progress_bars=None,
        )
    )

    return mesh, [left_plate, right_plate, dielectric, air]

