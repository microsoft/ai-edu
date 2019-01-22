// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using GoldenNumber;
using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            Console.WriteLine("输入房间号：");
            string inputRoomId = Console.ReadLine();
            int roomId = 0;
            if (!int.TryParse(inputRoomId, out roomId))
            {
                Console.WriteLine("解析房间号失败，默认进入0号房间");
            }

            RunBot(roomId).Wait();
        }

        static Tuple<double, double> GetNumber(List<double> goldenNumberList, int numberCount)
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
                // 用最近十回合的黄金点的均值作为下一回合的预测值
                number1 = goldenNumberList.AsEnumerable().Reverse().Take(10).Average();
                if (numberCount == 2)
                {
                    // 用最近一回合的黄金点值作为下一回合的预测值
                    number2 = goldenNumberList.Last();
                }
            }

            return Tuple.Create(number1, number2);
        }

        static async Task RunBot(int roomId)
        {
            GoldenNumberService service = new GoldenNumberService(new HttpClient());

            try
            {
                // 创建用户
                var user = await service.NewUserAsync($"AI玩家{new Random().Next() % 10000}");
                var userId = user.UserId;
                Console.WriteLine($"玩家：{user.NickName}  Id：{user.UserId}");

                Console.WriteLine($"房间号：{roomId}");

                while (true)
                {
                    // 查询房间状态
                    var state = await service.GetStateAsync(userId, roomId);

                    if (state.State == 2)
                    {
                        Console.WriteLine($"房间已结束，退出");
                        break;
                    }

                    if (state.State == 1)
                    {
                        Console.WriteLine($"房间尚未开始，1秒后轮询");
                        await Task.Delay(1000);
                        continue;
                    }

                    if (state.HasSubmitted)
                    {
                        Console.WriteLine($"已提交数据，等待下一回合");
                        await Task.Delay((state.LeftTime + 1) * 1000);
                        continue;
                    }

                    Console.WriteLine($"进入第{state.FinishedRoundCount + 1}回合");

                    // 查询历史黄金点
                    var todayGoldenList = await service.GetTodayGoldenListAsync(roomId);

                    if (todayGoldenList.GoldenNumberList.Count != 0)
                    {
                        Console.WriteLine($"上一回合黄金点值为：{todayGoldenList.GoldenNumberList.Last()}");
                    }

                    // 计算预测值
                    var numbers = GetNumber(todayGoldenList.GoldenNumberList.ToList(), state.Numbers);
                    double number1 = numbers.Item1;
                    double number2 = numbers.Item2;

                    // 提交预测值
                    try
                    {
                        if (state.Numbers == 2)
                        {
                            await service.SubmitAsync(userId, state.RoundId.ToString(), number1.ToString(), number2.ToString());
                            Console.WriteLine($"本回合提交的预测值为：{number1}，{number2}");
                        }
                        else
                        {
                            await service.SubmitAsync(userId, state.RoundId.ToString(), number1.ToString(), "0");
                            Console.WriteLine($"本回合提交的预测值为：{number1}");
                        }
                    }
                    catch (SwaggerException<BadRequestRspModel> ex)
                    {
                        Console.WriteLine($"Error：{ex.Result.Message}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error：{ex.Message}");
                    }
                }

            }
            catch (SwaggerException<BadRequestRspModel> ex)
            {
                Console.WriteLine($"Error：{ex.Result.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error：{ex.Message}");
            }
        }

    }
}
