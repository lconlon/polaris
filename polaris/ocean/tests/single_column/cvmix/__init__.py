import os

from polaris import TestCase
from polaris.ocean.tests.single_column.forward import Forward
from polaris.ocean.tests.single_column.initial_state import InitialState
from polaris.ocean.tests.single_column.viz import Viz
from polaris.validate import compare_variables


class CVMix(TestCase):
    """
    The default test case for the single column test group simply creates
    the mesh and initial condition, then performs a short forward run on 4
    cores.
    """
    def __init__(self, test_group, resolution):
        """
        Create the test case
        Parameters
        ----------
        test_group : polaris.ocean.tests.single_column.SingleColumn
            The test group that this test case belongs to
        """
        name = 'cvmix'
        self.resolution = resolution
        if resolution >= 1.:
            res_str = f'{resolution:g}km'
        else:
            res_str = f'{resolution * 1000.:g}m'
        subdir = os.path.join(res_str, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)
        self.add_step(
            InitialState(test_case=self, resolution=resolution))

        self.add_step(
            Forward(test_case=self, ntasks=1, min_tasks=1,
                    openmp_threads=1))

        self.add_step(
            Viz(test_case=self))

    def validate(self):
        """
        Compare ``temperature``, ``salinity``, ``layerThickness`` and
        ``normalVelocity`` in the ``forward`` step with a baseline if one was
        provided.
        """
        super().validate()
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')
