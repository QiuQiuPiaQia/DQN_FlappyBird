import flappy_bird.game.wrapped_flappy_bird as game
import numpy as np
from agent import Agent

ACTIONS = 2


def playGame():
    # 初始化游戏环境
    game_state = game.GameState()

    # 构建智能体
    my_agent = Agent()

    # 智能体和环境进行交互
    x_t, r_0, terminal = game_state.frame_step(my_agent.do_nothing)

    # 记录走第几步
    t = 0
    while True:
        readout_t, action_index, a_t = my_agent.make_decision(x_t)

        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        # 更新
        x_t = x_t1_colored
        t += 1

        print("TIMESTEP", t, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))


def main():
    playGame()


if __name__ == '__main__':
    main()
