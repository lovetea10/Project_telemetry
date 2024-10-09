from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MyFigure,self).__init__(self.fig)
        self.axes = self.fig.add_subplot(111)

class MyFigurePolar(FigureCanvas):

    def __init__(self,width=5, height=4, dpi=100, fig = None, ax = None):
        if fig == None:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111, projection='polar')
        else:
            self.fig = fig
            self.axes = ax
        super(MyFigurePolar, self).__init__(self.fig)

class MyFigure3d(FigureCanvas):

    def __init__(self, width=5, height=4, dpi=100, fig = None, ax = None):
        if fig == None:
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.fig = fig
            self.axes = ax
        super(MyFigure3d, self).__init__(self.fig)