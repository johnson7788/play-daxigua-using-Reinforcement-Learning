import pygame as pg
from random import randrange
from Fruit import create_fruit
from Game import GameBoard


class Board(GameBoard):
    def __init__(self):
        self.create_time = 2
        self.gravity = (0, 800)
        GameBoard.__init__(self, self.create_time, self.gravity)
        # 初始化游戏的框架边界
        self.init_segment()
        self.setup_collision_handler()
        
    def next_frame(self):
        try:
            if not self.waiting:
                self.count += 1
            self.surface.fill(pg.Color('black'))

            self.space.step(1 / self.FPS)
            self.space.debug_draw(self.draw_options)
            if self.count % (self.FPS * self.create_time) == 0:
                # 随机创建一个水果，1：葡萄，2：樱桃，3：橘子，4：柠檬，都是比较小的水果，
                self.i = randrange(1, 5)
                # 水果的位置在中间创建，等待用户鼠标点击
                self.current_fruit = create_fruit(
                    self.i, int(self.WIDTH/2), self.init_y - 10)
                self.count = 1
                self.waiting = True
            # pygame返回的事件处理，如果用户点击了鼠标，那么在用户点击的位置创建上面的水果种类，就相当于移动了水果的位置
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
                elif event.type == pg.MOUSEBUTTONUP and self.i and self.waiting:
                    #获取用户鼠标点击的位置
                    x, _ = pg.mouse.get_pos()
                    #创建一个水果，self.i代表水果的类别，x是鼠标点击的位置，y的位置是我们初始化的位置
                    fruit = create_fruit(self.i, x, self.init_y)
                    self.fruits.append(fruit)
                    ball = self.create_ball(
                        self.space, x, self.init_y, m=fruit.r//10, r=fruit.r-fruit.r % 5, i=self.i)
                    self.balls.append(ball)
                    self.current_fruit = None
                    self.i = None
                    self.waiting = False

            if not self.lock:
                for i, ball in enumerate(self.balls):
                    if ball:
                        angle = ball.body.angle
                        x, y = (int(ball.body.position[0]), int(
                            ball.body.position[1]))
                        self.fruits[i].update_position(x, y, angle)
                        self.fruits[i].draw(self.surface)

            if self.current_fruit:
                self.current_fruit.draw(self.surface)
            pg.draw.aaline(self.surface, (0, 200, 0),
                           (0, self.init_y), (self.WIDTH, self.init_y), 5)
            #更新分数
            self.show_score()
            if self.check_fail():
                self.score = 0
                self.last_score = 0
                self.reset()
            pg.display.flip()
            self.clock.tick(self.FPS)

        except Exception as e:
            print(e)
            if len(self.fruits) > len(self.balls):
                seg = len(self.fruits) - len(self.balls)
                self.fruits = self.fruits[:-seg]
            elif len(self.balls) > len(self.fruits):
                seg = len(self.balls) - len(self.fruits)
                self.balls = self.balls[:-seg]

    def run(self):
        # 游戏主体程序
        while True:
            self.next_frame()


if __name__ == '__main__':

    game = Board()
    game.run()
