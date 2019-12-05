// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using GoldenNumber;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace BotDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            int roomId = 0;

            // Read roomid from args --room
            var index = args.ToList().FindIndex(item => item.Equals("--room", StringComparison.OrdinalIgnoreCase));
            if (index < 0 || args.Length <= index + 1 || !int.TryParse(args[index + 1], out roomId))
            {
                // Input the roomid if there is no roomid in args
                Console.WriteLine("Input room id: ");
                string inputRoomId = Console.ReadLine();
                if (!int.TryParse(inputRoomId, out roomId))
                {
                    Console.WriteLine("Parse room id failed, default join in to room 0");
                }
            }

            RunBot(roomId).Wait();
        }

        static Tuple<double, double> GeneratePredictionNumbers(List<double> goldenNumberList, int numberCount)
        {
            double number1 = 0;
            double number2 = 0;

            if (goldenNumberList == null || goldenNumberList.Count == 0)
            {
                number1 = 18;
                if (numberCount == 2)
                {
                    number2 = 18;
                }
            }
            else
            {
                // Use the average of latest 10 rounds golden number as the prediction number for next round
                number1 = goldenNumberList.AsEnumerable().Reverse().Take(10).Average();
                if (numberCount == 2)
                {
                    // Use the latest round golden number as the prediction number for the next round
                    number2 = goldenNumberList.Last();
                }
            }

            return Tuple.Create(number1, number2);
        }

        static async Task RunBot(int roomId)
        {
            GoldenNumberService service = new GoldenNumberService(new HttpClient());
            service.BaseUrl = "https://goldennumber.aiedu.msra.cn";

            try
            {
                var userInfoFile = "userinfo.txt";
                var userId = string.Empty;
                var nickName = string.Empty;

                try
                {
                    // Use an exist player
                    var userInfo = File.ReadAllText(userInfoFile).Split(',');
                    userId = userInfo[0];
                    nickName = userInfo[1];

                    Console.WriteLine($"Use an exist player: {nickName}  Id: {userId}");
                }
                catch (Exception)
                {
                    // Create a new player
                    var user = await service.NewUserAsync($"AI Player {new Random().Next() % 10000}");
                    userId = user.UserId;
                    nickName = user.NickName;
                    Console.WriteLine($"Create a new player: {nickName}  Id: {userId}");

                    File.WriteAllText(userInfoFile, $"{userId},{nickName}");
                }

                Console.WriteLine($"Room id: {roomId}");

                while (true)
                {
                    // Get the room state
                    var state = await service.StateAsync(userId, roomId);

                    if (state.State == 2)
                    {
                        Console.WriteLine($"The game has finished");
                        break;
                    }

                    if (state.State == 1)
                    {
                        Console.WriteLine($"The game has not started, query again after 1 second");
                        await Task.Delay(1000);
                        continue;
                    }

                    if (state.HasSubmitted)
                    {
                        Console.WriteLine($"Already submitted this round, wait for next round");
                        if (state.MaxUserCount == 0)
                        {
                            await Task.Delay((state.LeftTime + 1) * 1000);
                        }
                        else
                        {
                            // One round can be finished when all players submitted their numbers if the room have set the max count of users, need to check the state every second.
                            await Task.Delay(1000);
                        }
                        continue;
                    }

                    Console.WriteLine($"\r\nThis is round {state.FinishedRoundCount + 1}");

                    // Get history of golden numbers
                    var todayGoldenList = await service.TodayGoldenListAsync(roomId);

                    if (todayGoldenList.GoldenNumberList.Count != 0)
                    {
                        Console.WriteLine($"Last golden number is: {todayGoldenList.GoldenNumberList.Last()}");
                    }

                    // Predict
                    var numbers = GeneratePredictionNumbers(todayGoldenList.GoldenNumberList.ToList(), state.Numbers);
                    double number1 = numbers.Item1;
                    double number2 = numbers.Item2;

                    // Submit prediction number
                    try
                    {
                        if (state.Numbers == 2)
                        {
                            await service.SubmitAsync(userId, state.RoundId.ToString(), number1.ToString(), number2.ToString());
                            Console.WriteLine($"You submit numbers: {number1}，{number2}");
                        }
                        else
                        {
                            await service.SubmitAsync(userId, state.RoundId.ToString(), number1.ToString(), "0");
                            Console.WriteLine($"You submit number: {number1}");
                        }
                    }
                    catch (SwaggerException<BadRequestRspModel> ex)
                    {
                        Console.WriteLine($"Error: {ex.Result.Message}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                    }
                }

            }
            catch (SwaggerException<BadRequestRspModel> ex)
            {
                Console.WriteLine($"Error: {ex.Result.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
        }

    }
}
