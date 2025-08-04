import jax.numpy as np
import jax

"""
In this file we will define some quaternion operations for utility
"""

def quat_conjugate(quats: np.ndarray):
    """
    Compute the conjugate of an array of quaternions of shape (batch_sz, 4)

    Args:
        quats: array of quaternions, shape = (batch_sz, 4)

    Returns:
        conj: array of quaternion conjugates, shape = (batch_sz, 4)
    """
    conj = np.copy(quats) * -1
    conj.at[:, 0].multiply(-1)
    return conj


def quat_mul_batch(q1, q2):
    """
    Multiply two batches of quaternions.

    Args:
        q1: numpy array of shape (batch_sz, 4), where each row is a quaternion [w, x, y, z].
        q2: numpy array of shape (batch_sz, 4), where each row is a quaternion [w, x, y, z].

    Returns:
        numpy array of shape (batch_sz, 4), where each row is the resulting quaternion.
    """
    # Extract scalar and vector parts
    s1 = q1[:, 0]  # Shape: (batch_sz,)
    v1 = q1[:, 1:]  # Shape: (batch_sz, 3)
    
    s2 = q2[:, 0]  # Shape: (batch_sz,)
    v2 = q2[:, 1:]  # Shape: (batch_sz, 3)
    
    # Compute the scalar part of the result
    scalar_part = s1 * s2 - np.sum(v1 * v2, axis=1)  # Shape: (batch_sz,)
    
    # Compute the vector part of the result
    vector_part = (
        s1[:, np.newaxis] * v2 +  # s1 * v2
        s2[:, np.newaxis] * v1 +  # s2 * v1
        np.cross(v1, v2)          # v1 x v2
    )  # Shape: (batch_sz, 3)
    
    # Combine scalar and vector parts into the result
    result = np.hstack((scalar_part[:, np.newaxis], vector_part))  # Shape: (batch_sz, 4)
    
    return result


def normalize(vector_arr: np.ndarray):
    """
    Normalize array of vectors to unit length

    Args:
        vectors: numpy array of shape (batch_sz, n), where each row is a vector.

    Returns:
        numpy array of shape (batch_sz, n), where each row is a unit vector.
    """
    # Compute the norm of each vector and keep the dimensionality for broadcasting
    norms = np.linalg.norm(vector_arr, axis=1, keepdims=True)

    # avoid dividing by 0
    norms.at[norms == 0].set(1)

    unit_vecs = vector_arr / norms

    return unit_vecs

def quat_inv(quats: np.ndarray):
    """
    Args:
        quats: Array of quaternions, shape = (batch_sz, 4)

    Returns:
        q_invs: Array of quaternions that are the inverse of inputs, shape = (batch_sz, 4)
    """
    return quat_conjugate(normalize(quats))


def quat_as_matrix(quats):
    quat_norm = np.linalg.norm(quats, axis=1, keepdims=True)
    q_unit = quats / quat_norm

    w = q_unit[:, 0:1]
    x = q_unit[:, 1:2]
    y = q_unit[:, 2:3]
    z = q_unit[:, 3:4]

    # First row of the rotation matrix
    r00 = 2 * (w * w + x * x) - 1
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r0 = np.stack([r00, r01, r02], axis=2)

    # Second row of the rotation matrix
    r10 = 2 * (x * y + w * z)
    r11 = 2 * (w * w + y * y) - 1
    r12 = 2 * (y * z - w * x)
    r1 = np.stack([r10, r11, r12], axis=2)

    # Third row of the rotation matrix
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 2 * (w * w + z * z) - 1
    r2 = np.stack([r20, r21, r22], axis=2)

    rot_mat_tensor = np.concat([r0, r1, r2], axis=1)

    return rot_mat_tensor



def quat_exp(quats:np.ndarray):
    """
    Takes an array of quaternions and exponentiates them

    Args:
        quats: array of quaternions, shape = (batch_sz, 4)

    Returns:
        exp_quats: array of exponentiated quaternions, shape = (batch_sz, 4)
    """

    s, v = quats[:, 0], quats[:, 1:]

    norm_v = np.clip(np.linalg.norm(v, axis=1), a_min=1e-8, a_max=np.inf)

    # Compute the scalar part of the result
    scalar_part = np.exp(s) * np.cos(norm_v)  # Shape: (batch_sz,)
    
    # Compute the vector part of the result
    vector_part = (
        np.exp(s)[:, np.newaxis] *  # e^s
        (np.sin(norm_v) / norm_v)[:, np.newaxis] *  # sin(||v||) / ||v||
        v  # v
    )  # Shape: (batch_sz, 3)

    # Combine scalar and vector parts into the result
    result = np.hstack((scalar_part[:, np.newaxis], vector_part))  # Shape: (batch_sz, 4)
    
    return result


def quat_from_endpts(p1, p2) -> np.ndarray:
    '''
    Takes in two endpoints as lists and computes the quaternion rotation
     - Rotation is w.r.t. principal axis z = (0, 0, 1)
     - NOTE: his function is NOT for batch processing

    Do this by computing rotation axis as cross product of z and normalized unit vector b/w endpoints
    '''
    v = p2 - p1
    v_norm = np.linalg.norm(v)

    # Endpoints shouldn't be the same
    assert v_norm != 0

    # Normalize v
    v = v / v_norm
    z = np.array([0,0,1])

    # Rotation axis:
    r = np.cross(z, v)
    r_norm = np.linalg.norm(r)
    theta = np.arccos(np.dot(z, v))

    # Edge cases: (r is parallel to z, no rotation or flipped)
    if r_norm == 0:
        if r == z: return np.array([1,0,0,0]) # Unit quaternion
        else: return np.array([0, 1, 0, 0])

    # Normalize r if r_norm neq to 0
    r = r / r_norm
    w = np.cos(theta / 2)
    xyz = r * np.sin(theta / 2)

    return np.hstack([np.array([w]), xyz])





