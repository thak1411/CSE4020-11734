import glfw
import numpy as np
from OpenGL.GL import *
mode = GL_LINE_LOOP

def render():
    global mode

    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    glBegin(mode)

    points = 12

    angles = np.linspace(0, 2 * np.pi, points, endpoint = False)
    x = np.cos(angles)
    y = np.sin(angles)
    for i in range(points):
        glVertex2f(x[i], y[i])
    glEnd()

def key_callback(window, key, scancode, action, mods):
    global mode

    if key == glfw.KEY_0: mode = GL_POLYGON
    elif key == glfw.KEY_1: mode = GL_POINTS
    elif key == glfw.KEY_2: mode = GL_LINES
    elif key == glfw.KEY_3: mode = GL_LINE_STRIP
    elif key == glfw.KEY_4: mode = GL_LINE_LOOP
    elif key == glfw.KEY_5: mode = GL_TRIANGLES
    elif key == glfw.KEY_6: mode = GL_TRIANGLE_STRIP
    elif key == glfw.KEY_7: mode = GL_TRIANGLE_FAN
    elif key == glfw.KEY_8: mode = GL_QUADS
    elif key == glfw.KEY_9: mode = GL_QUAD_STRIP

def main():
    if not glfw.init():
        return

    window = glfw.create_window(480, 480, "2020021949-2-1", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.set_key_callback(window, key_callback)

    glfw.make_context_current(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        render()
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__": main()