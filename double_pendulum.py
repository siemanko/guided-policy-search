# Double pendulum formula translated from the C code at
# http://www.physics.usyd.edu.au/~wheat/dpend_html/solve_dpend.c

from numpy import sin, cos, pi, array
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


class DoublePendulum(object):
    @staticmethod
    def example_pendulum():
        params = {
            'g_ms2': 9.8, # acceleration due to gravity, in m/s^2
            'l1_m': 1.0, # length of pendulum 1 in m
            'l2_m': 1.0, # length of pendulum 2 in m
            'm1_kg': 1.0, # mass of pendulum 1 in kg
            'm2_kg': 1.0, # mass of pendulum 2 in kg
        }
        return DoublePendulum(params)

    def __init__(self, params):
        self.P = params

    def derivs(self, state, t):
        G = self.P['g_ms2']
        L1 = self.P['l1_m']
        L2 = self.P['l2_m']
        M1 = self.P['m1_kg']
        M2 = self.P['m2_kg']
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        del_ = state[2]-state[0]
        den1 = (M1+M2)*L1 - M2*L1*cos(del_)*cos(del_)
        dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_)
                   + M2*G*sin(state[2])*cos(del_) + M2*L2*state[3]*state[3]*sin(del_)
                   - (M1+M2)*G*sin(state[0]))/den1

        dydx[2] = state[3]

        den2 = (L2/L1)*den1
        dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_)
                   + (M1+M2)*G*sin(state[0])*cos(del_)
                   - (M1+M2)*L1*state[1]*state[1]*sin(del_)
                   - (M1+M2)*G*sin(state[2]))/den2

        return dydx


    def simulate(self, intial_state, time_range, visualize=False):
        # th1 and th2 are the initial angles (degrees)
        # w10 and w20 are the initial angular velocities (degrees per second)

        # integrate your ODE using scipy.integrate.
        res = integrate.odeint(self.derivs, initial_state, time_range)

        if visualize:
            self.visualize(res, time_range)

    def visualize(self, y, time_range):
        x1 = self.P['l1_m']*sin(y[:,0])
        y1 = -self.P['l1_m']*cos(y[:,0])

        x2 = self.P['l2_m']*sin(y[:,2]) + x1
        y2 = -self.P['l2_m']*cos(y[:,2]) + y1

        pen_len = self.P['l1_m'] + self.P['l2_m']

        fig = plt.figure()

        halfxlim = pen_len + 0.1

        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-halfxlim, halfxlim), ylim=(-halfxlim, halfxlim))
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]
            dt = time_range[i] - time_range[i-1]
            line.set_data(thisx, thisy)
            time_text.set_text(time_template%(i*dt))
            return line, time_text

        average_dt_s = sum([time_range[i]-time_range[i-1] for i in range(1, len(time_range)) ])/float(len(time_range) -1)
        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
            interval=average_dt_s*1000/2, blit=True, init_func=init)

        #ani.save('double_pendulum.mp4', fps=15)
        plt.show()


if __name__ == '__main__':
    p = DoublePendulum.example_pendulum()

    initial_state = [pi, 0.0, 0, 0.0]
    time_range = np.arange(0.0, 20, 0.01)

    y = p.simulate(initial_state, time_range, visualize=True)
