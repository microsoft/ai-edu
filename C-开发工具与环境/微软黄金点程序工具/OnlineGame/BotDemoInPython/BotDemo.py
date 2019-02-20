# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Use `pip install bravado` to install bravado
from bravado.client import SwaggerClient
from bravado.exception import HTTPBadRequest
import random
import time

import warnings
warnings.filterwarnings("ignore")


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


def main():
    host = 'https://goldennumber.azurewebsites.net/'
    jsonpath = '/swagger/v1/swagger.json'

    client = SwaggerClient.from_url(host + jsonpath)

    roomId = input("Input room id: ")
    try:
        roomId = int(roomId)
    except:
        roomId = 0
        print('Parse room id failed, default join in to room 0')

    try:
        user = client.Default.Default_NewUser(nickName='AI Player ' + str(random.randint(0, 9999))).response().result
        userId = user.userId

        print('Player: ' + user.nickName + '  Id: ' + userId)
        print('Room id: ' + str(roomId))

        while True:
            state = client.Default.Default_GetState(uid=userId, roomid=roomId).response().result
    
            if state.state == 2:
                print('The game has finished')
                break

            if state.state == 1:
                print('The game has not started, query again after 1 second')
                time.sleep(1)
                continue

            if state.hasSubmitted:
                print('Already submitted this round, wait for next round')
                time.sleep(state.leftTime + 1)
                continue

            print('This is round ' + str(state.finishedRoundCount + 1))

            todayGoldenList = client.Default.Default_GetTodayGoldenList(roomid=roomId).response().result
            if len(todayGoldenList.goldenNumberList) != 0:
                print('Last golden number is: ' + str(todayGoldenList.goldenNumberList[-1]))

            number1, number2 = GeneratePredictionNumbers(todayGoldenList.goldenNumberList, state.numbers)

            try:
                if (state.numbers == 2):
                    client.Default.Default_Submit(uid=userId, rid=state.roundId, n1=str(number1), n2=str(number2)).response()
                    print('You submit numbers: ' + str(number1) + ', ' + str(number2))
                else:
                    client.Default.Default_Submit(uid=userId, rid=state.roundId, n1=str(number1), n2='0').response()
                    print('You submit number: ' + str(number1))
            except HTTPBadRequest as args:
                print('Error: ' + args.swagger_result.message)
            except Exception as args:
                print('Error: ' + str(args))

    except HTTPBadRequest as args:
        print('Error: ' + args.swagger_result.message)
    except Exception as args:
        print('Error: ' + str(args))


if __name__ == '__main__':
    main()
