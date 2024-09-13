import glfw
import numpy as np
from OpenGL.GL import *

tx, tr = 0., 0.

def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()

    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0., 0.]))
    glVertex2fv(np.array([1., 0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0., 0.]))
    glVertex2fv(np.array([0., 1.]))
    glEnd()

    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv((T @ np.array([.0, .5, 1.]))[:-1])
    glVertex2fv((T @ np.array([.0, .0, 1.]))[:-1])
    glVertex2fv((T @ np.array([.5, .0, 1.]))[:-1])
    glEnd()

def key_callback(window, key, scancode, action, mods):
    global tx, tr

    if action == glfw.PRESS:
        if key == glfw.KEY_Q: tx -= .1
        elif key == glfw.KEY_E: tx += .1
        elif key == glfw.KEY_A: tr += 10
        elif key == glfw.KEY_D: tr -= 10
        elif key == glfw.KEY_1: tx, tr = 0., 0.

def main():
    if not glfw.init():
        return

    window = glfw.create_window(480, 480, "2020021949-3-1", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.set_key_callback(window, key_callback)

    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        t = glfw.get_time()

        T = np.array([[1., 0., tx],
                        [0., 1., 0],
                        [0., 0., 1.]])
        
        th = np.radians(tr)
        
        R = np.array([[np.cos(th), -np.sin(th), 0.],
                      [np.sin(th), np.cos(th), 0.],
                      [0., 0., 1.]])
        
        render(T @ R)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__": main()