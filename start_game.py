import flappy_bird.game.wrapped_flappy_bird as game
import pygame
import numpy as np


# 我们有两种可选的行为，一种就是不动，小鸟就自己下落
# 另一种就是我们点击鼠标左键，小鸟向上飞一下
ACTIONS = 2


def play_game():
    # 启动游戏，通过类创建一个实例对象
    game_state = game.GameState()

    # while true 死循环
    while "flappy bird" != "angry bird":
        # 下面一行相当于创建一个长度为2的，全为0的数组
        # array([0., 0.])
        a_t = np.zeros([ACTIONS])
        a_t[0] = 1  # array([1., 0.]) 对应的就是下降，相当于do nothing

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                a_t = np.zeros([ACTIONS])
                a_t[1] = 1  # array([0., 1.]) 对应的就是小鸟上升
            else:
                pass

        # 这个frame_step函数就是进行游戏的下一帧
        _, _, terminal = game_state.frame_step(a_t)

        # 如果crash，终止循环，当前程序会执行完，游戏等于就是结束了
        # if terminal:
        #     break


def main():
    play_game()


if __name__ == '__main__':
    main()





