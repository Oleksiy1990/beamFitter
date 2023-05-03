import pytest
from lmfit import Parameters
import numpy as np

import beam_fitter.beam_fitter_cli.mathmodels as mm


@pytest.fixture
def parameters_G1D_correct():
    """
    Generate a correct parameters object for testing
    residual_G1D function
    """
    pars_correct = Parameters()
    pars_correct.add("peak_height", value=1)
    pars_correct.add("peak_width", value=1)
    pars_correct.add("peak_position", value=1)
    pars_correct.add("background", value=1)
    return pars_correct


@pytest.fixture
def parameters_G1D_wrong():
    """
    Generate a wrong parameters object for testing
    residual_G1D function
    """
    pars_wrong = Parameters()
    pars_wrong.add("p_height")  # this does not exist
    pars_wrong.add("p_width")  # this does not exist
    pars_wrong.add("peak_position")  # this is correct
    # pars_correct.add("background") # background missing
    return pars_wrong

# there is a problem that pytest cannot directly import the fixture output
# into a parametrized test, therefore we need to put the name of the fixture
# as a string, and then use request.getfixturevalue


@pytest.mark.parametrize("pars_test_string, x_test, data_test, eps_test",
                         [
                             ("parameters_G1D_correct",
                              np.array([1, 2, 3]), None, None),
                             ("parameters_G1D_wrong", np.array(
                                 [1, 2, 3]), None, None),
                             ("parameters_G1D_wrong", np.array(
                                 [1, 2, 3]), np.array(
                                 [1, 2, 3]), None),	
                             ("parameters_G1D_wrong", np.array(
                                 [1, 2, 3]), 
                                 np.array([1,2,3]), 
                                 np.array([1,2,3]))
                         ]
                         )
def test_residual_G1D(pars_test_string, x_test,
                      data_test,
                      eps_test,
                      request):  # the request is there to get the fixture value
    pars_test = request.getfixturevalue(pars_test_string)  # this must give
    # lmfit.Parameters object
    if (pars_test_string == "parameters_G1D_wrong"):
        with pytest.raises(KeyError):
            mm.residual_G1D(pars_test, x_test, data=data_test, eps=eps_test)
    else:
        assert (isinstance(mm.residual_G1D(pars_test, x_test, data_test), np.ndarray),
                "The function must return a numpy array")
