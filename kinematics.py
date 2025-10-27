import numpy as np

# System constants
q1 = np.array([0,0])
q2 = np.array([0,0])
q3 = np.array([0,0])
q4 = np.array([0,0])

b1 = np.array([0,1.0175])
b2 = np.array([0.975,1.0175])
b3 = np.array([0.975,0])
b4 = np.array([0,0])


def inverseKinematics(position, orientation):
    p = position
    theta = orientation

    R = np.array([[np.cos(theta), -np.sin(theta)],
                 [-np.sin(theta), np.cos(theta)]])

    l1 = p + R @ q1 - b1
    l2 = p + R @ q2 - b2
    l3 = p + R @ q3 - b3
    l4 = p + R @ q4 - b4

    return np.linalg.norm(l1), np.linalg.norm(l2), np.linalg.norm(l3), np.linalg.norm(l4)



p = [0.5,0.5]
theta = 0

l1,l2,l3,l4 = inverseKinematics(p,theta)
print(l1, l2, l3, l4)