# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Use `pip install pyswagger requests` to install pyswagger and requests
from pyswagger import App
from pyswagger.contrib.client.requests import Client

import random
import time
import argparse

def GeneratePredictionNumbers(goldenNumberList, numberCount):
    number1 = 0.0
    number2 = 0.0

    if len(goldenNumberList) == 0:
        number1 = 18.0
        if numberCount == 2:
            number2 = 18.0
    else:
        # Use the average of latest 10 rounds golden number as the prediction number for next round
        number1 = sum(goldenNumberList[-10:]) / float(len(goldenNumberList))
        if numberCount == 2:
            # Use the latest round golden number as the prediction number for the next round
            number2 = goldenNumberList[-1]

    return number1, number2


def main(roomId):
    host = 'https://goldennumber.aiedu.msra.cn/'
    jsonpath = '/swagger/v1/swagger.json'

    app = App._create_(host + jsonpath)
    client = Client()

    if not roomId:
        # Input the roomid if there is no roomid in args
        roomId = input("Input room id: ")
        try:
            roomId = int(roomId)
        except:
            roomId = 0
            print('Parse room id failed, default join in to room 0')

    userInfoFile = "userinfo.txt"
    userId = None
    nickName = None
    try:
        # Use an exist player
        with open(userInfoFile) as f:
            userId, nickName = f.read().split(',')[:2]
        print('Use an exist player: ' + nickName + '  Id: ' + userId)
    except:
        # Create a new player
        userResp = client.request(
            app.op['NewUser'](
                nickName='AI Player ' + str(random.randint(0, 9999))
            ))
        assert userResp.status == 200
        user = userResp.data
        userId = user.userId
        nickName = user.nickName
        print('Create a new player: ' + nickName + '  Id: ' + userId)

        with open(userInfoFile, "w") as f:
            f.write("%s,%s" % (userId, nickName))

    print('Room id: ' + str(roomId))

    while True:
        stateResp = client.request(
            app.op['State'](
                uid=userId,
                roomid=roomId
            ))
        assert stateResp.status == 200
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
                roomid=roomId
            ))
        assert todayGoldenListResp.status == 200
        todayGoldenList = todayGoldenListResp.data
        if len(todayGoldenList.goldenNumberList) != 0:
            print('Last golden number is: ' + str(todayGoldenList.goldenNumberList[-1]))

        number1, number2 = GeneratePredictionNumbers(todayGoldenList.goldenNumberList, state.numbers)

        if (state.numbers == 2):
            submitRsp = client.request(
                app.op['Submit'](
                    uid=userId,
                    rid=state.roundId,
                    n1=str(number1),
                    n2=str(number2)
                ))
            if submitRsp.status == 200:
                print('You submit numbers: ' + str(number1) + ', ' + str(number2))
            else:
                print('Error: ' + submitRsp.data.message)

        else:
            submitRsp = client.request(
                app.op['Submit'](
                    uid=userId,
                    rid=state.roundId,
                    n1=str(number1)
                ))
            if submitRsp.status == 200:
                print('You submit number: ' + str(number1))
            else:
                print('Error: ' + submitRsp.data.message)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--room', type=int, help='Room ID', required=False)
    args = parser.parse_args()

    main(args.room)