# -*- coding: utf-8 -*-
import copy
import sys
import random

DISCOUNT_FACTOR = 0.9  # 감가율


class PolicyIterationAgent:
    def __init__(self, env):
        self.env = env
        # 가치(value)값을 담을 2차원 리스트
        self.values = [[0.00] * env.width for _ in range(env.height)]

        # 정책(policy)를 담을 리스트 각 상태(state) 에 대한 정책은 각 행동(action)에 대한 확률 값으로 된 리스트입니다.
        self.policies = [[[0.25, 0.25, 0.25, 0.25]] * env.width for _ in range(env.height)]
        self.policies[2][2] = []
        
        # 계산을 위한 가치(value)리스트의 복사본
        self.valueCopy = self.values.copy()

    # 주어진 숫자 만큼 evaluation 하는 함수입니다
    def policy_evaluation(self):
        values_copy = copy.deepcopy(self.values)
        for state in self.env.get_all_states():
            values_copy[state[0]][state[1]] = round(self.compute_value(state), 2)
        self.values = copy.deepcopy(values_copy)

    # 상태(state)와 행동(action)으로 부터 Q값 계산
    def compute_value(self, state):

        value = 0

        for action in self.env.possible_actions:
            temp = 0.0
            next_state = self.env.state_after_action(state, action)
            next_value = self.get_value(next_state)
            temp += self.env.get_transition_prob(state, action) * next_value
            temp *= DISCOUNT_FACTOR
            temp += self.env.get_reward(next_state)
            temp *= self.get_policy(state, action)
            value += temp

        if state == [2, 2]:
            return 0.0

        return value

    # 가치값(values)들로 부터 행동(action)을 산출(compute)
    def compute_policy(self, state):

        if len(self.env.possible_actions) == 0:
            return None

        value = -sys.maxsize
        max_index = []
        result = [0.0, 0.0, 0.0, 0.0]

        for index, action in enumerate(self.env.possible_actions):
            next_state = self.env.state_after_action(state, action)
            temp = self.get_value(next_state)
            temp += self.env.get_reward(next_state)
            if temp == value:
                max_index.append(index)
            elif temp > value:
                value = temp
                max_index.clear()
                max_index.append(index)

        prob = 1/len(max_index)
        for index in max_index:
            result[index] = prob
        return result

    # 모든 상태들에 대해서 정책 (policy)를 업데이트
    def policy_improvement(self):
        
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue
            self.set_policy(state, self.compute_policy(state))

    def get_action(self, state):
        random_pick = random.randrange(100) / 100
        policy = self.get_policy(state)
        policy_sum = 0.0
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return self.env.possible_actions[index]

    # 전체 정책 리스트 받아오기
    def get_policies(self):
        return copy.deepcopy(self.policies)

    # 상태와 행동에 따른 정책 받아오기
    def get_policy(self, state, action=None):

        if action is None:
            return self.policies[state[0]][state[1]]

        if state == [2, 2]:
            return 0.0

        return self.policies[state[0]][state[1]][self.env.possible_actions.index(action)]

    def set_policy(self, state, policy_list):
        self.policies[state[0]][state[1]] = policy_list

    # 전체 가치(value)값 리스트 받아오기
    def get_values(self):
        return copy.deepcopy(self.values)

    # 특정 상태(state)의 가치(value)를 반환하는 함수
    def get_value(self, state):
        return round(self.values[state[0]][state[1]], 2)
