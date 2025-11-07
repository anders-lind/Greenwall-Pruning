import numpy as np
import scipy


def forward_kinematics():
    q1 = np.array([-0.0425, 0.02])
    q2 = np.array([0.0425, 0.02])
    q3 = np.array([0.0425, -0.02])
    q4 = np.array([-0.0425, -0.02])
    q = [q1, q2, q3, q4]

    B1 = np.array([0, 1.0175])
    B2 = np.array([0.975, 1.0175])
    B3 = np.array([0.975, 0])
    B4 = np.array([0, 0])
    B = [B1, B2, B3, B4]


    def test(input):
        x = input[0]
        y = input[1]
        theta = input[2]

        l = [0.55000, 0.60283, 0.76216, 0.72111]

        z_rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        q_world = np.zeros((4,2))
        d_sq = np.zeros(4)
        d = np.zeros(4)
        for i in range(4):
            q_world[i] = (z_rot @ q[i] + [x, y])
            d_sq[i] = (B[i][0] - q_world[i][0])**2 + (B[i][1] - q_world[i][1])**2
            d[i] = np.sqrt(d_sq[i])

        MSE = 0
        for i in range(4):
            MSE += (d[i]-l[i])**2
        MSE = MSE/4

        return MSE


    print(scipy.optimize.minimize(test, [0.0, 0.0, 0.0]))




if __name__ == "__main__":
    forward_kinematics()