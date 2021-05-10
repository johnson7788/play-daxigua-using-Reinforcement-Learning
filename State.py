import pygame as pg
from random import randrange
from Fruit import create_fruit
from Game import GameBoard


class AI_Board(GameBoard):
    def __init__(self):
        # 每隔0.5秒，创建一个新的水果
        self.create_time = 0.5
        # 重力，可以算出水果下落的速度
        self.gravity = (0, 4000)
        GameBoard.__init__(self, self.create_time, self.gravity)
        #动作的数量
        self.action_num = 16
        # 初始化画布
        self.init_segment()
        self.setup_collision_handler()

    def decode_action(self, action):
        # 根据不同的动作信号，选择不同的放置位置
        seg = (self.WIDTH - 40) // self.action_num
        x = action * seg + 20
        print(f"根据选择的动作:{action},这次水果放置到的位置是，横坐标是:{x},")
        # 返回要放置的位置
        return x

    def next_frame(self, action=None):
        """
        输入一个动作，对游戏进行交互, 这里是游戏的部分, 其它时间不用输入动作，等待水果自然落下，所以action就是None了
        :param action:  eg: 0
        :type action: int
        :return: image: 形状(400, 800, 3), self.score:int, reward:int, self.alive:bool
        :rtype:
        """
        try:
            # 这一帧的奖励初始化
            reward = 0
            if not self.waiting:
                self.count += 1
            self.surface.fill(pg.Color('black'))

            self.space.step(1 / self.FPS)
            self.space.debug_draw(self.draw_options)
            #  每隔固定的帧，我们的新的水果才出现，因为需要等待水果落地后与其它水果进行合并
            if self.count % (self.FPS * self.create_time) == 0:
                self.i = randrange(1, 5)
                self.current_fruit = create_fruit(
                    self.i, int(self.WIDTH/2), self.init_y - 10)
                self.count = 1
                self.waiting = True

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    exit()
            if not action is None and self.i and self.waiting:
                # 根据选择不同的动作，放置不同的位置
                x = self.decode_action(action)
                # 创建一个水果
                fruit = create_fruit(self.i, x, self.init_y)
                # 记录所有的水果
                self.fruits.append(fruit)
                # 水果开始下落， x, self.init_y水果下落的初始位置
                ball = self.create_ball(
                    self.space, x, self.init_y, m=fruit.r//10, r=fruit.r-fruit.r % 5, i=self.i)
                # 记录
                self.balls.append(ball)
                # 把当前的水果类清空，因为一会要放下一个水果
                self.current_fruit = None
                # 随机的初始的水果需要清空
                self.i = None
                # 处理完成了，不需要等待了
                self.waiting = False
            # 计算奖励
            reward = self.score - self.last_score
            if reward > 0:
                self.last_score = self.score
            # 更新水果合并的的信息
            if not self.lock:
                for i, ball in enumerate(self.balls):
                    if ball:
                        angle = ball.body.angle
                        x, y = (int(ball.body.position[0]), int(
                            ball.body.position[1]))
                        self.fruits[i].update_position(x, y, angle)
                        self.fruits[i].draw(self.surface)
            # 如果下一个要出现的水果已经生成，那么画到图像上
            if self.current_fruit:
                self.current_fruit.draw(self.surface)

            pg.draw.aaline(self.surface, (0, 200, 0),
                           (0, self.init_y), (self.WIDTH, self.init_y), 5)
            #更新下最新的分数
            self.show_score()
            # 检查游戏是否结束
            if self.check_fail():
                self.score = 0
                self.last_score = 0
                self.reset()
            # 开始绘制下落的一帧
            pg.display.flip()
            self.clock.tick(self.FPS)
            # 截图，获取图像信息
            image = pg.surfarray.array3d(pg.display.get_surface())

        except Exception as e:
            print(e)
            if len(self.fruits) > len(self.balls):
                seg = len(self.fruits) - len(self.balls)
                self.fruits = self.fruits[:-seg]
            elif len(self.balls) > len(self.fruits):
                seg = len(self.balls) - len(self.fruits)
                self.balls = self.balls[:-seg]
        # image: shape: (400, 800, 3)
        return image, self.score, reward, self.alive

    def next(self, action=None):
        """
        根据输入的动作，返回状态奖励，存活状态等
        :param action: eg: 0
        :type action: int
        :return:
        :rtype:
        """
        # 选择一个动作
        _, _, reward, _ = self.next_frame(action=action)
        # 需要经历3*self.FPS，等待水果落下，合并后，才能给出最终奖励
        for _ in range(self.FPS * 3):
            _, _, nreward, _ = self.next_frame()
            reward += nreward
        # 把下一个要出现的水果画到游戏画面上，这时应该不会有任何奖励
        image, _, nreward, _ = self.next_frame()
        # 累加所有奖励
        reward += nreward
        if reward == 0:
            #如果没有得分，那么我们让奖励为负值，这样有益于训练
            reward = -self.i
        # 获取最终的奖励
        return image, self.score, reward, self.alive

    def run(self):
        # 游戏主体程序
        while True:
            action = randrange(0, self.action_num)
            print('action:', action)
            _, score, reward, alive = self.next(action=action)
            print('score:{} reward:{} alive:{}'.format(score, reward, alive))


if __name__ == '__main__':

    game = AI_Board()
    game.run()
