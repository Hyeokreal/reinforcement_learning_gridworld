# -*- coding: utf-8 -*-
import copy
import random

DISCOUNT_FACTOR = 0.9  # 감가율


class ValueIteration:
    def __init__(self, env):
        self.env = env
        # 가치(value)값을 담을 2차원 리스트
        self.values = [[0.00] * env.width for _ in range(env.height)]
        # 계산을 위한 가치(value)리스트의 복사본
        self.valueCopy = self.values.copy()

    # 주어진 숫자 만큼 evaluation 하는 함수입니다
    def calculate_value(self):
        values_copy = copy.deepcopy(self.values)
        for state in self.env.get_all_states():
            values_copy[state[0]][state[1]] = round(self.compute_max_q_value(state), 2)
        self.values = copy.deepcopy(values_copy)

    # 상태(state)와 행동(action)으로 부터 Q값 계산
    def compute_max_q_value(self, state):

        if state == [2, 2]:
            return 0.0

        q_value_list = []

        for action in self.env.possible_actions:
            q_value = 0.0
            next_state = self.env.state_after_action(state, action)
            next_value = self.get_value(next_state)
            q_value += self.env.get_transition_prob(state, action) * next_value
            q_value *= DISCOUNT_FACTOR
            q_value += self.env.get_reward(next_state)
            q_value_list.append(q_value)

        return max(q_value_list)

    def get_action(self, state, random_pick = True):

        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            next_value = DISCOUNT_FACTOR * self.get_value(next_state)
            next_value += self.env.get_reward(next_state)

            if next_value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = next_value
            elif next_value == max_value:
                action_list.append(action)

        if random_pick is True:
            return random.sample(action_list, 1)[0]

        return action_list

    # 전체 가치(value)값 리스트 받아오기
    def get_values(self):
        return copy.deepcopy(self.values)

    # 특정 상태(state)의 가치(value)를 반환하는 함수
    def get_value(self, state):
        return round(self.values[state[0]][state[1]], 2)
