import numpy as np
import mcubes
import OpenGL.GL as gl
import pangolin
import threading
import time
class Visualizer(threading.Thread):
    def __init__(self):
        super(Visualizer, self).__init__()
        self.points = np.zeros((3,3), np.float32)
        self.colors = np.zeros((3,3), np.float32)
        self.is_run=True
    def run(self):

        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # Define Projection and initial ModelView matrix
        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(handler)
        while not pangolin.ShouldQuit() and self.is_run:

            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            self.dcam.Activate(self.scam)

            # Render OpenGL Cube
            #pangolin.glDrawColouredCube()

            # Draw Point Cloud
            #points = np.random.random((100000, 3)) * 10
            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)

            #colors = map(float, colors.flat)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointerf(self.points)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, self.colors)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.points.shape[0])
            #pangolin.glDrawVertices(points.shape[0], points, gl.GL_TRIANGLES)
            #pangolin.DrawPoints(points)

            pangolin.FinishFrame()
            time.sleep(0.05)
        self.is_run=False
def test():
    X, Y, Z = np.mgrid[:30, :30, :30]
    u = ((X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2)
    print(u.shape)
    vertices, triangles = mcubes.marching_cubes(u, 0)
    colors = (vertices-np.min(vertices)) / (np.max(vertices)-np.min(vertices))
    colors[:, 0] = 1.0-colors[:,0]
    vt = vertices[triangles.flatten()].astype(np.float32)*0.1
    cl = colors[triangles.flatten()].astype(np.float32)
    vis = Visualizer()
    vis.points = vt
    vis.colors = cl
    vis.start()
    #vis.start()
    t = 0.0
    while vis.is_run:#not pangolin.ShouldQuit():
    #    vis.draw()
    #    print("b")
        vis.points = vt +np.sin(t)
        vis.colors = cl
        t+=0.01
        time.sleep(0.01)
    vis.join()
if __name__ == "__main__":
    test()