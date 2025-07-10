#is a copy of service_duration from EMSplex code version v2
#distribution of integers (mins), which is defined as the sum of first-order driving approximation + on-site-time + .63 handovertime. Sets this as on-site duration and replaces travel times and handover time to 0 


from scipy.stats import lognorm
import numpy as np

NUMBER_OF_MESH_POINTS_PER_MINUTE = 4  # sets internal mesh used for convolution of continuous distributions and shifts

# from SSRN id=4874479
# distributions in minutes
MEAN_PARAMETER_ON_SITE = 3.61
STDDEV_PARAMETER_ON_SITE = 0.38
LEFT_END_POINT_ON_SITE = -10.01
RIGHT_END_POINT_ON_SITE = 88
MEAN_PARAMETER_HOSPITAL = 3.58
STDDEV_PARAMETER_HOSPITAL = 0.39
LEFT_END_POINT_HOSPITAL = -8.25
RIGHT_END_POINT_HOSPITAL = 88
HOSPITAL_PROBABILITY = 0.63

def convolve_pmfs(pmf1, pmf2):
    """
    Convolves two PMFs.

    Args:
        pmf1 (dict): The first PMF as a dictionary.
        pmf2 (dict): The second PMF as a dictionary.

    Returns:
        dict: The resulting convolved PMF.
    """
    result = {}
    for value1, prob1 in pmf1.items():
        for value2, prob2 in pmf2.items():
            combined_value = value1 + value2
            combined_prob = prob1 * prob2
            if combined_value in result:
                result[combined_value] += combined_prob
            else:
                result[combined_value] = combined_prob
    return result


def mixture(pmf1, pmf2, mixture_prob):
    """
    Combines two PMFs into a mixture PMF using a given mixture probability.

    Args:
        pmf1 (dict): The first PMF as a dictionary.
        pmf2 (dict): The second PMF as a dictionary.
        mixture_prob (float): The weight for the first PMF (0 <= mixture_prob <= 1).

    Returns:
        dict: The resulting mixture PMF.
    """
    all_keys = sorted(set(pmf1) | set(pmf2))
    result = {}
    for key in all_keys:
        result[key] = pmf1.get(key, 0) * mixture_prob + pmf2.get(key, 0) * (1 - mixture_prob)
    return result


def collapse_and_truncate_pmf(pmf, mesh_points_per_minute, tolerance=None):
    """
    Collapses a PMF by summing every mesh_points_per_minute consecutive keys.
    Optionally truncates when cumulative mass reaches 1-tolerance.
    Renormalizes only if truncation occurs.

    Args:
        pmf (dict): The original PMF
        mesh_points_per_minute (int): Number of mesh points to collapse into one
        tolerance (float or None): If set, stop when cumulative mass >= 1-tolerance

    Returns:
        dict: The collapsed (and possibly truncated and renormalized) PMF
    """
    collapsed = {}
    cumulative_mass = 0.0
    truncated = False
    for key, prob in sorted(pmf.items()):
        new_key = (int(key) - 1) // mesh_points_per_minute + 1
        collapsed[new_key] = collapsed.get(new_key, 0) + prob
        cumulative_mass += prob
        if tolerance is not None and cumulative_mass >= 1 - tolerance:
            truncated = True
            break
    if truncated:
        total_mass = sum(collapsed.values())
        for k in collapsed:
            collapsed[k] /= total_mass
    return collapsed

def make_lognorm_pmf(mean, stddev, left, right, mesh_size, shift=0):
    """
    Creates a discretized lognormal probability mass function (PMF) over a specified interval.

    Args:
        mean (float): Mean parameter of the lognormal distribution (in log-space).
        stddev (float): Standard deviation parameter of the lognormal distribution (in log-space).
        left (float): Left endpoint (minimum value) of the distribution support.
        right (float): Right endpoint (maximum value) of the distribution support.
        mesh_size (float): The size of each discretization interval (e.g., in minutes).
        shift (int, optional): Number of mesh points to shift the PMF to the right. Default is 0.

    Returns:
        dict: A dictionary mapping discretized time points (shifted by `shift`) to their probabilities.
    """
    mesh_points = mesh_size * np.arange(1, np.ceil(right / mesh_size) + 1)
    cdf_on_mesh = lognorm.cdf(mesh_points, stddev, loc=left, scale=np.exp(mean))
    probs = np.diff(cdf_on_mesh, prepend=0)
    probs /= probs.sum()
    return {value + shift: float(probs[value - 1]) for value in range(1, len(probs) + 1)}


def create_distribution(number_of_periods_per_hour, shift=0, tolerance=None):
    """
    Returns the service duration distribution as a PMF.

    Args:
        number_of_periods_per_hour (int): Number of periods in one hour.
        shift (float, optional): Amount to shift the distribution, in hours. Rounded to the nearest internal mesh point. Default is 0.
        tolerance (float or None, optional): If set, truncates the PMF when cumulative mass reaches 1 - tolerance.

    Returns:
        dict: The service duration distribution as a dictionary.
    """
    mesh_size = 60 / number_of_periods_per_hour / NUMBER_OF_MESH_POINTS_PER_MINUTE

    on_site_duration_pmf = make_lognorm_pmf(
        MEAN_PARAMETER_ON_SITE, STDDEV_PARAMETER_ON_SITE,
        LEFT_END_POINT_ON_SITE, RIGHT_END_POINT_ON_SITE,
        mesh_size, shift=int(round(shift * 60 / mesh_size))
    )

    hospital_duration_pmf = make_lognorm_pmf(
        MEAN_PARAMETER_HOSPITAL, STDDEV_PARAMETER_HOSPITAL,
        LEFT_END_POINT_HOSPITAL, RIGHT_END_POINT_HOSPITAL,
        mesh_size
    )

    service_duration_pmf = mixture(
        convolve_pmfs(on_site_duration_pmf, hospital_duration_pmf),
        on_site_duration_pmf,
        HOSPITAL_PROBABILITY
    )

    return collapse_and_truncate_pmf(service_duration_pmf, NUMBER_OF_MESH_POINTS_PER_MINUTE, tolerance=tolerance)
