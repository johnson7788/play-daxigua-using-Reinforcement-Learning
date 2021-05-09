import pygame as pg


def create_fruit(type, x, y):
    """
    创建一个水果，类型是1到11， 1：葡萄，2：樱桃，3：橘子，4：柠檬，5：猕猴桃，6：西红柿，7：桃子，8：菠萝，9：柚子，10：西瓜，11：大西瓜
    :param type: 1-11
    :type type: int
    :param x: react的大小x
    :type x:
    :param y: react的大小y
    :type y:
    :return: 创建的的水果的类
    :rtype:
    """
    fruit = None
    if type == 1:
        fruit = PT(x, y)
    elif type == 2:
        fruit = YT(x, y)
    elif type == 3:
        fruit = JZ(x, y)
    elif type == 4:
        fruit = NM(x, y)
    elif type == 5:
        fruit = MHT(x, y)
    elif type == 6:
        fruit = XHS(x, y)
    elif type == 7:
        fruit = TZ(x, y)
    elif type == 8:
        fruit = BL(x, y)
    elif type == 9:
        fruit = YZ(x, y)
    elif type == 10:
        fruit = XG(x, y)
    elif type == 11:
        fruit = DXG(x, y)
    return fruit


class Fruit():
    def __init__(self, x, y):
        """
        水果类，是父类
        :param x: react的x的大小
        :type x:
        :param y:
        :type y:
        """
        self.load_images()
        # 初始化一个pygame正方形
        self.rect = self.image.get_rect()
        # react的x和y，确定它的大小
        self.rect.x = x
        self.rect.y = y
        self.angle_degree = 0

    def load_images(self):
        pass

    def update_position(self, x, y, angle_degree=0):
        self.rect.x = x - self.r
        self.rect.y = y - self.r
        self.angle_degree = angle_degree
        # self.image = pg.transform.rotate(self.image, self.angle_degree)

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class PT(Fruit):
    def __init__(self, x, y):
        """
        葡萄
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 10
        self.type = 1
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        """
        加载普通图片
        :return:
        :rtype:
        """
        self.image = pg.image.load('res/01.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class YT(Fruit):
    def __init__(self, x, y):
        """
        樱桃
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 15
        self.type = 2
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/02.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class JZ(Fruit):
    def __init__(self, x, y):
        """
        橘子
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 21
        self.type = 3
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/03.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class NM(Fruit):
    def __init__(self, x, y):
        """
        柠檬
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 23
        self.type = 4
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/04.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class MHT(Fruit):
    def __init__(self, x, y):
        """
        猕猴桃
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 29
        self.type = 5
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/05.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class XHS(Fruit):
    def __init__(self, x, y):
        """
        西红柿
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 35
        self.type = 6
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/06.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class TZ(Fruit):
    def __init__(self, x, y):
        """
        桃子
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 37
        self.type = 7
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/07.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class BL(Fruit):
    def __init__(self, x, y):
        """
        菠萝
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 50
        self.type = 8
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/08.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class YZ(Fruit):

    def __init__(self, x, y):
        """
        柚子
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 59
        self.type = 9
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/09.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class XG(Fruit):

    def __init__(self, x, y):
        """
        西瓜
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 60
        self.type = 10
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/10.png')
        self.image = pg.transform.smoothscale(self.image, self.size)


class DXG(Fruit):

    def __init__(self, x, y):
        """
        大西瓜
        :param x:
        :type x:
        :param y:
        :type y:
        """
        self.r = 2 * 78
        self.type = 11
        self.size = (self.r*2, self.r*2)
        Fruit.__init__(self, x - self.r, y - self.r)

    def load_images(self):
        self.image = pg.image.load('res/11.png')
        self.image = pg.transform.smoothscale(self.image, self.size)
