from datetime import datetime

from numpy.typing import NDArray
import numpy as np
from scipy.linalg import norm
from astropy.coordinates import ITRS
from lunarsky import MCMF
from astropy import units as u


ARC_SECONDS_TO_RADIANS = np.pi / 648000
EARTH_ROTATION_DERIVATIVE = np.pi * 1.00273781191135448 / 43200 
DERIVATIVE_MATRIX = np.array([
    [0.0, -EARTH_ROTATION_DERIVATIVE, 0.0], 
    [EARTH_ROTATION_DERIVATIVE, 0.0, 0.0], 
    [0.0, 0.0, 0.0]
])


def eci_to_ecef(
    eci_point: NDArray[np.float64],
    utc_time: datetime,
    eci_velocity: None | NDArray[np.float64] = None
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ECI point/velocity to ECEF point/velocity.

    Args:
        eci_point (NDArray[np.float64]): (3,) 1-d vector describing ECI point [X, Y, Z]
        utc_time (datetime): Observed time of position and/or velocity
        eci_velocity (NDArray[np.float64]): (3,) 1-d vector describing ECI velocity [Vx, Vy, Vz]

    Returns:
        ecef_point (NDArray[np.float64]): (3,) 1-d vector describing ECEF point [X, Y, Z]
        ecef_velocity (NDArray[np.float64]): (3,) 1-d vector describing ECEF velocity [Vx, Vy, Vz]

    Note:
        The velocity is only returned if a velocity is supplied
    """
    # Convert the utc time to julian day, then to century
    julian_day = utc_time_to_julian_date(utc_time)
    julian_century = (julian_day - 2451545.0) / 36525.0 # Eq. 5.2

    # Get the rotation matrix
    rotation_eci_to_ecef = rotation_matrix_ecef_to_eci(julian_century).T

    # Rotate the position
    ecef_point = rotation_eci_to_ecef @ eci_point

    # Rotate the velocity if it is supplied
    if eci_velocity is not None:
        ecef_velocity = (
            rotation_eci_to_ecef @ eci_velocity 
            - (DERIVATIVE_MATRIX @ rotation_eci_to_ecef) @ eci_point
        )
        return ecef_point, ecef_velocity

    return ecef_point


def ecef_to_eci(
    ecef_point: NDArray[np.float64],
    utc_time: datetime,
    ecef_velocity: NDArray[np.float64] = None
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert ECEF point/velocity to ECI point/velocity.
    
    Args:
        ecef_point (NDArray[np.float64]): (3,) 1-d vector describing ECEF point [X, Y, Z]
        utc_time (datetime): Observed time of position and/or velocity
        ecef_velocity (NDArray[np.float64]): (3,) 1-d vector describing ECEF velocity [Vx, Vy, Vz]

    Returns:
        eci_point (NDArray[np.float64]): (3,) 1-d vector describing ECI point [X, Y, Z]
        eci_velocity (NDArray[np.float64]): (3,) 1-d vector describing ECI velocity [Vx, Vy, Vz]

    Note:
        The velocity is only returned if a velocity is supplied
    """
    # Convert the utc time to julian day, then to century
    julian_day = utc_time_to_julian_date(utc_time)
    julian_century = (julian_day - 2451545.0) / 36525.0 # Eq. 5.2

    # Get the rotation matrix
    rotation_ecef_to_eci = rotation_matrix_ecef_to_eci(julian_century)

    # Rotate the position
    eci_point = rotation_ecef_to_eci @ ecef_point

    # Rotate the velocity if it is supplied
    if ecef_velocity is not None:
        eci_velocity = (
            rotation_ecef_to_eci @ ecef_velocity 
            + (rotation_ecef_to_eci @ DERIVATIVE_MATRIX) @ ecef_point
        )
        return eci_point, eci_velocity

    return eci_point


def rotation_matrix_ecef_to_eci(
    julian_century: float
) -> NDArray[np.float64]:
    """Return ecef to eci rotation matrix for a given time.

    Rotation is applied to the vector components.

    P_eci = R(t) @ P_ecef

    Args:
        julian_date (float): Time in Julian centuries.

    Returns:
        rotation_matrx (NDArray[float]): 3x3 Matrix to rotate ECEF to ECI
    """
    # Angular motion of the earth
    earth_rotation_angle = 2 * np.pi * (
        0.7790572732640 + 1.00273781191135448 * 36525.0 * julian_century
    )
    earth_matrix = np.eye(3)
    earth_matrix[0, 0] = earth_matrix[1, 1] = np.cos(earth_rotation_angle)
    earth_matrix[1, 0] = np.sin(earth_rotation_angle)
    earth_matrix[0, 1] = -1 * earth_matrix[1, 0]

    # Precession / Nutation rotation matrix (Eq. 5.10)
    gcrs_x, gcrs_y = compute_celestial_positions(julian_century)
    a = 0.5 + 0.125 * (gcrs_x*gcrs_x + gcrs_y*gcrs_y)
    
    pn_matrix = np.array([
        [1-a*gcrs_x*gcrs_x,  -a*gcrs_x*gcrs_y, gcrs_x],
        [ -a*gcrs_x*gcrs_y, 1-gcrs_y*gcrs_y  , gcrs_y],
        [ -gcrs_x         ,  -gcrs_y         , 1-a*(gcrs_x*gcrs_x + gcrs_y*gcrs_y)]
    ])

    # Return the rotation
    return pn_matrix @ earth_matrix


## The monstrosity that is the nutation / precession
def compute_celestial_positions(
    julian_century: float
)  -> tuple[float, float]:
    """Compute the x-y components of the celestial pole in earth reference frame.

    Args:
        julian_date (float): Time in Julian centuries.

    Returns:
        celestial_x (float): x-component of the pole vector in radians
        celestial_y (float): y-component of the pole vector in radians

    Notes:
        See Equation 5.16 with Table 5.2a / 5.2b. Supplemental material has
        all 2000 parameters or so. Download zip file from website
    """
    celestial_x = precession_x(julian_century)
    celestial_y = precession_y(julian_century)

    # Update for nutation (Coeffients are micro arc-seconds)
    omega = moon_ascension(julian_century) * ARC_SECONDS_TO_RADIANS
    D =  moon_elongation(julian_century) * ARC_SECONDS_TO_RADIANS
    F = moon_longitude(julian_century) * ARC_SECONDS_TO_RADIANS
    l_prime = sun_anomoly(julian_century) * ARC_SECONDS_TO_RADIANS

    # Precompute reoccuring argument
    f_omega_d = 2 * (F + omega - D)

    celestial_x += 1e-6 * ((
        -6844318.44 * np.sin(omega) - 523908.04 * np.sin(f_omega_d) 
        - 90552.22 * np.sin(2*(F+omega)) + 82168.76 * np.sin(2*omega)
        + 58707.02 * np.sin(l_prime)
    ) + julian_century * (
        205833.11 * np.cos(omega) + 12814.01  * np.cos(f_omega_d)
    ))

    celestial_y += 1e-6 * ((
        9205236.26 * np.cos(omega) + 573033.42 * np.cos(f_omega_d) 
        + 97846.69 * np.cos(2*(F+omega)) - 89618.24 * np.cos(2*omega)
        + 22438.42 * np.cos(l_prime-f_omega_d)
    ) + julian_century * (
        153041.79 * np.sin(omega) + 11714.49  * np.sin(f_omega_d)
    ))

    return celestial_x * ARC_SECONDS_TO_RADIANS, celestial_y * ARC_SECONDS_TO_RADIANS


def utc_time_to_julian_date(
    utc_time: datetime
) -> float:
    """Convert UTC time to Julian date.

    This calculation is only valid for days after March 1900.

    Args:
        utc_time (datetime): The observation time as a datetime object

    Returns:
        julian_date (float): The observation time as a julian date.
    """
    year, month, day = utc_time.year, utc_time.month, utc_time.day
    julian_date = (
        367 * year - 7 * (year + (month + 9) // 12) // 4 + 275 * month // 9 + day + 1721013.5
    )

    # update with the frational day
    julian_date +=  (
        utc_time.hour + utc_time.minute / 60 
        + (utc_time.second + 1e-6 * utc_time.microsecond) / 3600 
    ) / 24

    return julian_date


## Precession Polynomials (Arc-Seconds)
# Equation 5.16
precession_x = np.polynomial.polynomial.Polynomial(
    [-0.016617, 2004.191898, -0.4297829, -0.19861834]
)
precession_y = np.polynomial.polynomial.Polynomial(
    [-0.006951,  -0.025896, -22.4072747, 0.00190059]
)

# Nutation Polynomials (Arc-Seconds)
# Equation 5.43

# Mean_anomaly of the moon (l)
moon_anomoly = np.polynomial.polynomial.Polynomial(
    [485868.249036, 1717915923.217800, 31.879200, 0.05163500]
)

# Mean_anomaly of the sun (l-prime)
sun_anomoly = np.polynomial.polynomial.Polynomial(
    [1287104.793048, 129596581.048100, -0.55320]
)

# Moon thing 1 (F)
moon_longitude = np.polynomial.polynomial.Polynomial(
    [335779.526232, 1739527262.8478, -12.7512]
)

# Moon elongation from sun (D)
moon_elongation = np.polynomial.polynomial.Polynomial(
    [1072260.703692, 1602961601.209000, -6.3706]
)

# Moon ascension node (Omega)
moon_ascension = np.polynomial.polynomial.Polynomial(
    [450160.398036, -6962890.5431, 7.4722]
)

def tangential_dist_vector(r_sat: np.ndarray, r_rover: np.ndarray, R_central: float):
  r_ab = r_rover - r_sat

  # Projection of a on vector between a and b
  angle = np.arccos( np.dot(-r_sat, r_ab)/norm(r_sat)/norm(r_ab) )
  
  return norm(r_sat) * np.sin(angle) - R_central

def fixed_to_lla(r_m, body_rad_m):
    

    def lla_single(r):
      x,y,z = r
      return np.array([
        np.degrees(np.arctan2(z, np.sqrt(x*x + y*y))), # lat
        np.degrees(np.arctan2(y, x)), # lon
        norm(r) - body_rad_m
      ])
    
    return np.apply_along_axis(lla_single, -1, r_m)


def ecef_to_mcmf_m(v, t):
    v = np.atleast_2d(v)
    
    x = v[:,0]*u.m
    y = v[:,1]*u.m
    z = v[:,2]*u.m
    
    itrs = ITRS(x=x, y=y, z=z, obstime=t)
    mcmf = itrs.transform_to(MCMF(obstime=t))
    
    out = np.column_stack([mcmf.x.to(u.m).value,
                           mcmf.y.to(u.m).value,
                           mcmf.z.to(u.m).value])
    return out.squeeze()  # handles single vector

def unit(vec):
    return vec / norm(vec)

# Long function for calculating vectors from satellite to 
def moon_sat_vec(moon_ecef_m: np.ndarray,
                 rover_ecef_m: np.ndarray,
                 alt_m: float,
                 mask_deg: float,
                 num_radial: float,
                 num_angular: float):
    
    r = rover_ecef_m - moon_ecef_m # Vector from center of moon to rover position
    h_vec = unit(np.cross(r, [0,0,1])) # horizontal vector from rover pos
    v_vec = unit(np.cross(h_vec, r)) # vertical vector from rover pos

    rho = alt_m + norm(r)
 
    theta_horizon = np.arccos(norm(r) / (alt_m + norm(r)))
    # theta_mask = np.radians(90 - mask_deg)
    # Crazy formula
    theta_mask = np.pi/2 - np.deg2rad(mask_deg) - np.arcsin(
        (norm(r))/(norm(r) + alt_m) * np.sin(np.pi/2 + np.deg2rad(mask_deg))
    )
    # print(theta_mask * pnt.DEG)
    max_angle = min(theta_mask, theta_horizon)

    center_angles = np.linspace(0, max_angle, num_radial + 1)[1:]
    az_angles = np.linspace(0, 2*np.pi, num_angular + 1)[:-1]

    print("norm(r) =", norm(r))
    print("h_vec norm =", norm(h_vec))
    print("v_vec norm =", norm(v_vec))

    # Check orthogonality
    print("dot(r,h) =", np.dot(unit(r), h_vec))
    print("dot(r,v) =", np.dot(unit(r), v_vec))
    print("dot(h,v) =", np.dot(h_vec, v_vec))

    sat_vecs = [moon_ecef_m + rho * unit(r)] # center satellite
    alts = np.linalg.norm(sat_vecs - moon_ecef_m, axis=1) - np.linalg.norm(r)
    print(f"ALTS: {alts}")
    for az in az_angles:
        for center_angle in center_angles:
            direction = (
                np.cos(center_angle) * unit(r) +
                np.sin(center_angle) * (
                    np.cos(az) * h_vec +
                    np.sin(az) * v_vec
                )
            )
            direction = direction / np.linalg.norm(direction)

            sat_vecs.append(moon_ecef_m + rho * direction)

    sat_vecs = np.array(sat_vecs)

    print(f"Calculated {len(sat_vecs)} satellites!")

    return sat_vecs

def get_horizon_circle(rover_mcmf_m: np.ndarray, alt_m: float, mask_deg: float):
  
    
    r = rover_mcmf_m
    h_vec = unit(np.cross(r, [0,0,1])) # horizontal vector from rover pos
    v_vec = unit(np.cross(h_vec, r)) # vertical vector from rover pos

    rho = alt_m + norm(r)
 
    theta_horizon = np.arccos(norm(r) / (alt_m + norm(r)))

    # Crazy formula
    theta_mask = np.pi/2 - np.deg2rad(mask_deg) - np.arcsin(
        (norm(r))/(norm(r) + alt_m) * np.sin(np.pi/2 + np.deg2rad(mask_deg))
    )
    # print(theta_mask * pnt.DEG)
    max_angle = min(theta_mask, theta_horizon)

    az_angles = np.linspace(-np.pi, np.pi, 1000)[:-1]


    points = []
    for az in az_angles:
      points.append(
        rho * (
          np.cos(max_angle) * unit(r) +
          np.sin(max_angle) * (np.cos(az) * h_vec + np.sin(az) * v_vec)
        )
      )

    return np.array(points)

def horizon_lla(alt, mask_deg, rover_mcmf_m, R_body):
  horizon_circle_mcmf = get_horizon_circle(rover_mcmf_m[0], alt, mask_deg)
  lla = fixed_to_lla(horizon_circle_mcmf, R_body)
  return lla




