# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Use `pip install pyswagger requests` to install pyswagger and requests
from pyswagger import App
from pyswagger.contrib.client.requests import Client

# Use `pip install numpy pandas` to install numpy and pandas
import numpy as np
import pandas as pd

import argparse
import base64
import hashlib
import os
import random
import time

# Below class QLearningTable is copy from MorvanZhou's tutorials
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/2_Q_Learning_maze/RL_brain.py
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions    # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state_exist(state)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[state, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()    # next state is not terminal
        else:
            q_target = r    # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)    # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )


def action1(gArray):
    number = 28
    if len(gArray) != 0:
        number = gArray[-1]
    return number, number

def action2(gArray):
    number = 28
    if len(gArray) != 0:
        number = gArray[-1]*0.618
    return number, number

def action3(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-5:])
    return number, number

def action4(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-5:])*0.618
    return number, number

def action5(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-10:])
    return number, number

def action6(gArray):
    number = 28
    if len(gArray) != 0:
        number = np.average(gArray[-10:])*0.618
    return number, number

def action7(gArray):
    if len(gArray) == 0:
        return 28, 28
    if len(gArray) == 1:
        return gArray[0], gArray[0]
    number = gArray[-1] / gArray[-2] * gArray[-1]
    if number <= 0:
        number = 0.001
    if number >= 100:
        number = 100*0.618
    return number, number

def action8(gArray):
    if len(gArray) == 0:
        return 28, 28
    number1=50
    number2=50/30*0.618+np.average(gArray[-5:])
    return number1, number2

actions=[]
actions.append(action1)
actions.append(action2)
actions.append(action3)
actions.append(action4)
actions.append(action5)
actions.append(action6)
actions.append(action7)
actions.append(action8)

n_actions = len(actions)
RL = QLearningTable(actions=list(range(n_actions)))

def getState(gArray):
    if len(gArray) == 0 or len(gArray) == 1:
        return '0_0'
    else:
        sub = np.array(gArray[-10:])
        sub1 = sub[:-1]
        sub2 = sub[1:]
        dif = sub1 - sub2
        up = sum(1 for e in dif if e < 0)
        down = sum(1 for e in dif if e > 0)
        return '{}_{}'.format(up, down)

lastState=None
lastAction=None

def GeneratePredictionNumbers(goldenNumberList, lastScore, numberCount):
    global lastState
    global lastAction

    state = getState(goldenNumberList)

    if lastState != None and lastAction != None:
        RL.learn(lastState, lastAction, lastScore, state)

    action = RL.choose_action(state)
    number1, number2 = actions[action](goldenNumberList)

    lastState = state
    lastAction = action

    return number1, number2

# Init swagger client
host = 'https://goldennumber.aiedu.msra.cn/'
jsonpath = '/swagger/v1/swagger.json'
app = App._create_(host + jsonpath)
client = Client()

# Make sure all the parameters have the right value
def perProcess(roomid, userid, usertoken):
    userInfoFile = 'userinfo.txt'
    nickname = None
    tokenFile = 'token.txt'

    # if not specify userid, try read userid locally
    if userid is None:
        if os.path.isfile(userInfoFile):
            with open(userInfoFile) as f:
                userid = f.read().split(',')[0]

    # verify userid is valide or not
    if userid:
        userResp = client.request(
                app.op['User'](
                    uid = userid
                ))
        if userResp.status == 400:
            print(f'Verify user ID failed: {userResp.data.message}')
            userid = None
        else:
            userid = userResp.data.userId
            nickname = userResp.data.nickName
            print(f'Use an exist player: {nickname}, User ID: {userid}')

    # create user if userid is empty or invalide
    if not userid:
        # random nickname
        nickname = 'AI Player ' + str(random.randint(0, 9999))
        userResp = client.request(
            app.op['NewUser'](
                nickName = nickname
            ))
        assert userResp.status == 200
        userid = userResp.data.userId
        nickname = userResp.data.nickName
        print(f'Create a new player: {nickname}, User ID: {userid}')

    # save user information locally
    with open(userInfoFile, "w") as f:
        f.write("%s,%s" % (userid, nickname))

    if roomid is None:
        # Input the roomid if there is no roomid in args
        roomid = input("Input room id: ")
        try:
            roomid = int(roomid)
            print(f'You are using room {roomid}')
        except:
            roomid = 0
            print('Parse room id failed, default use room 0')

    if not usertoken and os.path.isfile(tokenFile):
        with open(tokenFile) as f:
            usertoken = f.read()

    return roomid, userid, usertoken

def main(roomid, userid, usertoken):
    # Make sure all the parameters have the right value
    roomid, userid, usertoken = perProcess(roomid, userid, usertoken)

    while True:
        stateResp = client.request(
            app.op['State'](
                uid = userid,
                roomid = roomid
            ))
        if stateResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        state = stateResp.data

        if state.state == 2:
            print('The game has finished')
            break

        if state.state == 1:
            print('The game has not started, query again after 1 second')
            time.sleep(1)
            continue

        if state.hasSubmitted:
            print('Already submitted this round, wait for next round')
            if state.maxUserCount == 0:
                time.sleep(state.leftTime + 1)
            else:
                # One round can be finished when all players submitted their numbers if the room have set the max count of users, need to check the state every second.
                time.sleep(1)
            continue

        print('\r\nThis is round ' + str(state.finishedRoundCount + 1))

        todayGoldenListResp = client.request(
            app.op['TodayGoldenList'](
                roomid = roomid
            ))
        if todayGoldenListResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        todayGoldenList = todayGoldenListResp.data
        if len(todayGoldenList.goldenNumberList) != 0:
            print('Last golden number is: ' + str(todayGoldenList.goldenNumberList[-1]))

        lastRoundResp = client.request(
            app.op['History'](
                roomid = roomid,
                count = 1
            ))
        if lastRoundResp.status != 200:
            print('Network issue, query again after 1 second')
            time.sleep(1)
            continue
        lastScore = 0
        if len(lastRoundResp.data.rounds) > 0:
            scoreArray = [user for user in lastRoundResp.data.rounds[0].userNumbers if user.userId == userid]
            if len(scoreArray) == 1:
                lastScore = scoreArray[0].score
        print('Last round score: {}'.format(lastScore))

        number1, number2 = GeneratePredictionNumbers(todayGoldenList.goldenNumberList, lastScore, state.numbers)

        computedToken = ''
        if state.enabledToken:
            mergedString = userid + state.roundId + usertoken
            computedToken = base64.b64encode(hashlib.sha256(mergedString.encode('utf-8')).digest()).decode('utf-8')

        if state.numbers == 2:
            submitRsp = client.request(
                app.op['Submit'](
                    uid = userid,
                    rid = state.roundId,
                    n1 = str(number1),
                    n2 = str(number2),
                    token = computedToken
                ))
            if submitRsp.status == 200:
                print('You submit numbers: ' + str(number1) + ', ' + str(number2))
            else:
                print('Error: ' + submitRsp.data.message)
                time.sleep(1)

        else:
            submitRsp = client.request(
                app.op['Submit'](
                    uid = userid,
                    rid = state.roundId,
                    n1 = str(number1),
                    token = computedToken
                ))
            if submitRsp.status == 200:
                print('You submit number: ' + str(number1))
            else:
                print('Error: ' + submitRsp.data.message)
                time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roomid', type=int, help='Room ID', required=False)
    parser.add_argument('--userid', type=str, help='User ID', required=False)
    parser.add_argument('--token', type=str, help='User token', required=False)
    args = parser.parse_args()
    main(args.roomid, args.userid, args.token)