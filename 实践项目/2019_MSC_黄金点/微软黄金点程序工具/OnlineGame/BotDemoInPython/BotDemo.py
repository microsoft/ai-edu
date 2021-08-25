# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Use `pip install pyswagger requests` to install pyswagger and requests
from pyswagger import App
from pyswagger.contrib.client.requests import Client

import argparse
import base64
import hashlib
import os
import random
import time

def GeneratePredictionNumbers(goldenNumberList, numberCount):
    number1 = 0.0
    number2 = 0.0

    if len(goldenNumberList) == 0:
        number1 = 18.0
        if numberCount == 2:
            number2 = 18.0
    else:
        # Use the average of latest 10 rounds golden number as the prediction number for next round
        number1 = sum(goldenNumberList[-10:]) / float(len(goldenNumberList[-10:]))
        if numberCount == 2:
            # Use the latest round golden number as the prediction number for the next round
            number2 = goldenNumberList[-1]

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

        number1, number2 = GeneratePredictionNumbers(todayGoldenList.goldenNumberList, state.numbers)

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