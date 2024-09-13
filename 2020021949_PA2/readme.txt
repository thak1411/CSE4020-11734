PA2 Documents

Author:
 2020021949 Gunmo KU (구건모)

Description:
 This documents for computer graphics PA2.

---

This assignment is about implementing moving cow with smooth curve.

This assignment's main challenge is to implement the following features

1. Implement vertical drag movement
2. Implement smooth curve equation
3. Implement capturing the cow's movement
4. Implement animating the cow's move with captured six points

---

First, we need to implement vertical drag. before implementation of vertical drag, we need to understand the skeleton code's horizontal drag.

Given skeleton code implements horizontal drag with two event handlers. (cursor click event, cursor moving event)

if user clicks the window then cursor click event catch the information of the cursor's position & action.
so, user do left click then event handler can catch the user's left click with positions.

if user click in first time, handler change the "isDrag" value to H_DRAG (it means horizontal drag mode)
after set isDrag's value to H_DRAG, then cursor moving handler processing the moving cow like following mechanism. (assume that user clicked cow)

1. calculate the ray's intersection point with plane (plane is defined by up vector & clicked cow's poisition) and user cursor
2. get difference between clicked cow's position and intersection point (calculated in 1)
3. make translation matrix with difference (calculated in 2)
4. apply the translation matrix to the cow's position matrix


Given skeleton's horizontal drag implemented with following code.
    ray=screenCoordToRay(window, x, y);
    pp=pickInfo;
    p=Plane(np.array((0,1,0)), pp.cowPickPosition);
    c=ray.intersectsPlane(p);
    currentPos=ray.getPoint(c[1])
    T=np.eye(4)
    setTranslation(T, currentPos-pp.cowPickPosition)
    cow2wld=T@pp.cowPickConfiguration;

so, we can implement vertical drag with similar mechanism with horizontal drag.

1. calculate the ray's intersection point with plane (plane is defined by horizontal vector ("up vector" innerproduct "horizontal vector" = 0) & clicked cow's position) and user cursor
2. get difference between clicked cow's position and intersection point (calculated in 1)
3. discard difference without up vector's direction (up vector is y-axis & horizontal vector is z-axis then discard x's difference)
4. make translation matrix with difference (calculated in 3)
5. apply the translation matrix to the cow's position matrix


we can implement vertical drag like following code.
    ray=screenCoordToRay(window, x, y);
    pp=pickInfo;
    p=Plane(np.array((0,0,1)), pp.cowPickPosition);
    c=ray.intersectsPlane(p);
    currentPos=ray.getPoint(c[1])
    T=np.eye(4)
    setTranslation(T, vector3(0, (currentPos-pp.cowPickPosition)[1], 0))
    cow2wld=T@pp.cowPickConfiguration;

---

Second, we need to implement smooth curve equation.

In this assignment, i used two different curve equation. (catmull-rom spline, bspline)

1. catmull-rom spline

catmull-rom spline is the curve equation that interpolate the points with smooth curve.


catmull_rom(t, a, b, c, d) is the catmull-rom spline's equation.
 t: position parameter (0 <= t <= 1)
 a: previous point
 b: current point
 c: next point
 d: next next point

catmull-rom spline's equation make two tangent vector and calculate the point with hermite's equation.

hermite(t, a, b, u, v) is the hermite's equation.
 t: position parameter (0 <= t <= 1)
 a: current point
 b: next point
 u: tangent vector of a
 v: tangent vector of b

hermite's equation make the curve with two tangent vector.

so, catmull-rom spline's equation make the curve with four points. It can compute with following equation.

hermite(t, a, b, u, v) = s ** 2 * (1 + 2 * t) * a + t ** 2 * (1 + 2 * s) * b + s ** 2 * t * u - s * t ** 2 * v
 ; s = (1 - t)

catmull_rom(t, a, b, c, d) = hermite(t, b, c, u, v)
 ; u = (c - a) / 2
 ; v = (d - b) / 2

2. bspline

bspline is the curve equation that interpolate the points with smooth curve.
It looks like similar with catmull-rom spline but bspline is more smooth than catmull-rom spline.

bspline(t, a, b, c, d) is the bspline's equation.
 t: position parameter (0 <= t <= 1)
 a: previous point
 b: current point
 c: next point
 d: next next point

bspline's equation make the curve with four points. It can compute with following equation.

bspline(t, a, b, c, d) = b1 * a + b2 * b + b3 * c + b4 * d
 ; b1 = (-t^3 + 3t^2 - 3t + 1) / 6
 ; b2 = (3t^3 - 6t^2 + 4) / 6
 ; b3 = (-3t^3 + 3t^2 + 3t + 1) / 6
 ; b4 = t^3 / 6

---

Third, we need to implement capturing the cow's movement.

In this assignment, I have to capture the 6 points of cow's movement.

I decide the method of capturing the cow's movement with following mechanism.

1. first, user do left click the cow then start horizontal dragging mode.
2. if user press down the left click then start vertical dragging mode.
3. if user release the left click then stop the vertical dragging mode.
4. save the cow's position & get back to horizontal dragging mode.
5. repeat the 2~4 steps until capturing the 6 points.
6. if user captured the 6 points then stop the capturing mode & start animation mode.


so, i need to change the event handler (cursor click event, cursor moving event)

first, we need to make variables for captured cow's information & current state

current state is represent the state of capture process. (it called rn_state in my code)

1. didn't start capture process yet. (rn_state = -1)
2. capture the nth point (rn_state = n (= 0 ~ 5))
3. finish the capture process (rn_state = 6)
4. in animation process (rn_state = 7)

so, cursor click event handler changes "isDrag" value according to the current state(rn_state). and capture the cow's position if rn_state is 0 ~ 5. and ignore the event if rn_state > 5

cursor moving event handler changes the cow's position according to "isDrag" values & save the cow's position for captureing the cow.

You can see the implementation of capturing the cow's movement in my code. (SimpleScene.py - onMouseButton, onMouseDrag)

---

Fourth, we need to implement animating the cow's move with captured six points.

if rn_state is 6 then current state mean that we captured the 6 points of cow's movement. so, we can start the animation process.

in display function of SimpleScene.py, we detect the rn_state is 6 then start the animation process. (set rn_state to 7)
when start animation process, we need to save the start time of animation (it called animStartTime in my code)

After animStartTime, animation mode is continued for 6 * 3 seconds.

in each seconds, we interpolate the cow's position with catmull-rom spline or bspline. (you can change the curve equation in display function in my code)
 0~1 second: interpolate the cow's position from 1st point to 2nd point
 1~2 second: interpolate the cow's position from 2nd point to 3rd point
 2~3 second: interpolate the cow's position from 3rd point to 4th point
 3~4 second: interpolate the cow's position from 4th point to 5th point
 4~5 second: interpolate the cow's position from 5th point to 6th point
 5~6 second: interpolate the cow's position from 6th point to 1st point

and repeat the 3 times.

This way, the cow moves along the curve with 6 points.
However, the direction the cow is facing does not change.
Therefore, in order for the cow to face the curve it moves, it must be rotated.

In order to rotate the cow, we need to calculate the tangent vector of the curve at the current position of the cow.
The tangent vector can be calculated by spline(t + EPS) - spline(t) where EPS is a small value. The tangent vector is the direction the cow should face.

so, we can rotate the cow with tangent vector.
default cow's facing direction is x-axis of cow's local coordinate. so, we need to rotate the cow's local coordinate from x-axis to tangent vector.

rotate process can be implemented with following mechanism.

1. rotate the y-axis (up vector) to projection vector from tangent vector to xz-plane. - (1)
2. rotate the z-axis (forward vector) to tangent vector. - (2)

we have to compute the theta for rotate (1), (2).

1. arctan(tangent vector's x / tangent vector's z) = theta1
2. arctan(tangent vector's y / len of projection vector from tangent vector to xz-plane) = theta2

so, we can rotate the cow with theta1, theta2.


finally, cow moves along the curve with 6 points and face the curve with tangent vector.

you can see the result of my implementation in the video file. (bspline.mp4, catmull_rom.mp4)

