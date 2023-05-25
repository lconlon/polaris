import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf

from polaris import Step
from polaris.ocean.vertical import init_vertical_coord


class Init(Step):
    """
    A step for an initial condition for for the geostrophic test case
    """
    def __init__(self, test_case, mesh_name):
        """
        Create the step

        Parameters
        ----------
        test_case : polaris.ocean.tests.global_convergence.geostrophic.Geostrophic  # noqa: E501
            The test case this step belongs to

        mesh_name : str
            The name of the mesh
        """

        super().__init__(test_case=test_case,
                         name=f'{mesh_name}_init',
                         subdir=f'{mesh_name}/init')

        self.add_input_file(filename='mesh.nc', target='../mesh/mesh.nc')

        self.add_input_file(filename='graph.info', target='../mesh/graph.info')

        self.add_output_file(filename='initial_state.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config

        section = config['geostrophic']
        temperature = section.getfloat('temperature')
        salinity = section.getfloat('salinity')
        alpha = section.getfloat('alpha')
        u_0 = section.getfloat('u_0')
        h_0 = section.getfloat('h_0')

        a = constant['SHR_CONST_REARTH']
        g = constant['SHR_CONST_G']
        omega = 2 * np.pi / constant['SHR_CONST_SDAY']

        section = config['vertical_grid']
        bottom_depth = section.getfloat('bottom_depth')

        ds_mesh = xr.open_dataset('mesh.nc')
        angleEdge = ds_mesh.angleEdge
        latCell = ds_mesh.latCell
        latEdge = ds_mesh.latEdge
        lonCell = ds_mesh.lonCell
        sphere_radius = ds_mesh.sphere_radius

        ds = ds_mesh.copy()

        ds['bottomDepth'] = bottom_depth * xr.ones_like(latCell)
        ds['ssh'] = xr.zeros_like(latCell)

        init_vertical_coord(config, ds)

        theta=ds.latCell
        lamb=ds.lonCell

        h = h_0 - 1 / g * (a*omega*u_0+u_0**2/2)*(-np.cos(lamb)*np.cos(theta)*np.sin(alpha)+np.sin(theta)*np.cos(alpha))**2

        theta=ds.latEdge
        lamb=ds.lonEdge
        u = u_0 * (np.cos(theta)*np.cos(alpha)+np.cos(lamb)*np.sin(theta)*np.sin(alpha))
        v = -u_0*np.sin(lamb)*np.sin(alpha)

        ds['bottomDepth'] = bottom_depth * xr.ones_like(latCell)
        ds['ssh'] = -ds.bottomDepth + h

        init_vertical_coord(config, ds)

        # Xyler will do velocity projection

        ds['fCell'] = _coriolis(ds.lonCell,latCell,alpha,omega)
        ds['fEdge'] = _coriolis(ds.lonEdge,latEdge,alpha,omega)
        ds['fVertex'] = _coriolis(ds.lonVertex,latVertex,alpha,omega)


        write_netcdf(ds, 'initial_state.nc')

def _coriolis(lon,lat,alpha,omega):
    f = 2 * omega * (-np.cos(lon)*np.cos(lat)
                                    *np.sin(alpha)+np.sin(lat)*np.cos(alpha))
    return f
