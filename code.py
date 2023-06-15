from manim import *
from numpy import linalg as lg
import numpy as numpy
import cmath as math


class ej:
    def __init__(self, A, B):
        self.A = A.real
        self.B = B.real

    def mul(self, ej2):
        return ej(self.A * ej2.A, self.B + ej2.B)

    def show(self):
        print("%.4fe^ %.4fj" % (self.A, self.B))


def axvector(x):
    # if type(x[0]) is complex:
    return [x[0].real - x[1].imag, x[1].real + x[0].imag]


# else:
# return x


def axtoe(x):
    A = math.sqrt(x.real**2 + x.imag**2)
    B = math.atan(x.imag / x.real)
    return ej(A, B)


class Shapes(MovingCameraScene):
    # A few simple shapes
    def construct(self):
        matrix1 = np.mat([[1.2816, -0.2867], [0.6201, 0.5980]])
        matrix2 = np.mat([[-0.3680, 0.7796], [-1.3610, 0.4394]])
        matrix3 = np.mat([[0.0923, -0.6086], [1.7298, -0.7371]])
        matrix4 = np.mat([[0.1097, -0.2900], [1.1287, 1.2616]])

        # a, b = lg.eig(matrix1)

        # print(b)
        # b = np.asarray(b)
        # ej1 = axtoe(a[0])
        # ej1.show()
        # ej2 = axtoe(b[0][0])
        # ej2.show()
        # ej3 = axtoe(b[1][0])
        # ej3.show()

        # for i in range(10):
        #     ej2 = ej2.mul(ej1)
        #     ej3 = ej3.mul(ej1)
        #     ej2.show()
        #     ej3.show()

        matrix = matrix3
        f = [0, 0, 0]
        plen = 30
        vlen = 1
        margin = 4
        line = False
        doubledot = False
        n = 300

        tex1 = Tex(r"$x^{(k+1)}=Bx^{(k)}+f,k=(0,1,2,\cdots)$", font_size=72)
        # tex2 = Tex(
        #     r"$$B=\begin{bmatrix}1.2816&-0.2867\\0.6201&0.5980\end{bmatrix}$$$$f=[0,0]^T$$",
        #     font_size=72,
        # )
        # tex2 = Tex(
        #     r"$$B=\begin{bmatrix}-0.3680&0.7796\\-1.3610&0.4394\end{bmatrix}$$$$f=[0,0]^T$$",
        #     font_size=72,
        # )
        tex2 = Tex(
            r"$$B=\begin{bmatrix}0.0923&-0.6086\\1.7298&-0.7371\end{bmatrix}$$$$f=[0,0]^T$$",
            font_size=72,
        )
        tex3 = Tex(r"$x_0^0=[0,1]^T,x_1^0=[1,0]^T$", font_size=72)
        self.add(tex1)
        self.wait(2)
        self.play(FadeOut(tex1), Transform(tex1, tex2))
        self.add(tex2)
        self.wait(2)
        self.play(FadeOut(tex2), Transform(tex2, tex3))
        self.add(tex3)
        self.wait(2)
        self.play(FadeOut(tex3))
        # t1 = [0.4558 + 0.3292j, 0.8269]
        # t2 = [0.4558 - 0.3292j, 0.8269]
        # t1 = [-0.49455709, 0.86914515]
        # t2 = [0.41153084, -0.91139583]
        t1 = [0, vlen]
        t2 = [vlen, 0]
        v1 = Vector(t1)
        v2 = Vector(t2)
        plane_old = NumberPlane(
            x_range=(-plen, plen, 1),
            y_range=(-plen, plen, 1),
            x_length=2 * plen,
            y_length=2 * plen,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.5,
            },
        ).add_coordinates()
        plane = NumberPlane(
            x_range=(-plen, plen, 1),
            y_range=(-plen, plen, 1),
            x_length=2 * plen,
            y_length=2 * plen,
            background_line_style={
                "stroke_width": 4,
            },
            axis_config={
                "stroke_width": 4,
            },
        )
        maxheight = 2 * max(abs(t1[1]), abs(t2[1]))
        maxwidth = 2 * max(abs(t1[0]), abs(t2[0]))
        foucs = Rectangle(width=maxwidth + 0.01, height=maxheight)
        self.play(FadeIn(plane_old, plane, v1, v2))
        if doubledot:
            self.add(Dot(v1.get_vector(), radius=0.04))
        self.add(Dot(v2.get_vector(), radius=0.04))
        # self.add(foucs)
        self.wait(2)

        list1 = []
        list2 = []
        list1.append(v1.get_vector())
        list2.append(v2.get_vector())
        for i in range(n):
            if i < 5:
                t = 0.5
            else:
                t = 1 / (i - 4) + 0.001
            self.play(
                ApplyMatrix(matrix, plane, run_time=t),
                ApplyMatrix(matrix, v1, run_time=t),
                ApplyMatrix(matrix, v2, run_time=t),
            )
            if n < 20:
                self.wait(0.2 * t)

            vector1 = v1.get_vector()
            vector2 = v2.get_vector()

            if f[0] != 0 or f[1] != 0:
                vector1 += f
                vector2 += f
                temp1 = Vector(vector1)
                temp2 = Vector(vector2)
                self.play(
                    FadeOut(v1, run_time=t),
                    Transform(v1, temp1, run_time=t),
                    FadeOut(v2, run_time=t),
                    Transform(v2, temp2, run_time=t),
                    ApplyPointwiseFunction(lambda x: x + f, plane, run_time=t),
                )
                v1 = temp1
                v2 = temp2
                self.add(v1)
                self.add(v2)

            if doubledot:
                self.add(Dot(v1.get_vector(), radius=0.04))
            self.add(Dot(v2.get_vector(), radius=0.04))
            list1.append(vector1)
            list2.append(vector2)
            if n < 20:
                self.wait(0.01 * t)

            maxheight = max(maxheight, max(2 * abs(vector1[1]), 2 * abs(vector2[1])))
            maxwidth = max(maxwidth, max(2 * abs(vector1[0]), 2 * abs(vector2[0])))
            if foucs.width < maxwidth or foucs.height < maxheight:
                # self.remove(foucs)
                foucs = Rectangle(height=maxheight, width=maxwidth)
                # self.add(foucs)
                self.play(
                    self.camera.auto_zoom(
                        foucs, margin=margin, animate=True, run_time=t
                    )
                )
                if n < 20:
                    self.wait(0.01 * t)

        self.play(FadeOut(plane, v1, v2))
        if line:
            d = VGroup()
            l = VGroup()
            self.add(d)
            self.add(l)
            for i in range(n + 1):
                t = 1 / (i + 2) + 0.1
                if i != 0:
                    if doubledot:
                        l.add(Line(list1[i - 1], list1[i]))
                    l.add(Line(list2[i - 1], list2[i]))
                self.wait(t)
        self.wait(10)
