import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

border_w = 1
border_h = 1

fig = plt.figure()
ax = fig.add_subplot()

rec = plt.Rectangle([0,0], 0, 0)

def inverseKinematicPythagoras(botLeft:tuple[float,float], topLeft:tuple[float,float], topRight:tuple[float,float], botRight:tuple[float,float]) -> tuple[float, float, float, float]:
    botLeftLength = np.sqrt(np.pow(botLeft[0], 2) + np.pow(botLeft[1], 2))
    topLeftLength = np.sqrt(np.pow(topLeft[0], 2) + np.pow(topLeft[1]-border_h, 2))
    topRighLength = np.sqrt(np.pow(topRight[0]-border_w, 2) + np.pow(topRight[1]-border_h, 2))
    topLeftLength = np.sqrt(np.pow(topRight[0]-border_w, 2) + np.pow(topRight[1]-border_h, 2))
    
    return botLeftLength, topLeftLength, topRighLength, topLeftLength


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

    R = np.array([np.cos(theta), -np.sin(theta)],
                 [-np.sin(theta), np.cos(theta)])

    l1 = p + R @ q1 - b1
    l2 = p + R @ q2 - b2
    l3 = p + R @ q3 - b3
    l4 = p + R @ q4 - b4

    return l1, l2, l3, l4
    



def update(frame):
    ax.clear()

    # Draw rectangle
    rect_w = 0.5
    rect_h = 0.5
    rect_x = 1.0-0.015*frame
    rect_y = 1.0-0.015*frame
    rec.set_bounds([rect_x, rect_y, rect_w, rect_h])
    ax.add_patch(rec)

    # Draw lines
    l1x, l1y = [0, rect_x], [0, rect_y]
    l2x, l2y = [border_h, rect_x+rect_h], [0, rect_y]
    l3x, l3y = [border_h, rect_x+rect_h], [border_w, rect_y+rect_w]
    l4x, l4y = [0, rect_x], [border_w, rect_y+rect_w]
    ax.plot(l1x, l1y)
    ax.plot(l2x, l2y)
    ax.plot(l3x, l3y)
    ax.plot(l4x, l4y)

    ax.set_xlim(0, border_w)
    ax.set_ylim(0, border_h)


ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=30)
plt.show()

ani.save("motor_animation.gif")  # Save instead of showing
plt.close(fig)