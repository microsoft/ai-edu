// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using GoldedNumberClient.Utils;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace GoldedNumberClient.Models
{
    /// <summary>
    /// 控制游戏逻辑的核心类。
    /// </summary>
    public class Game
    {
        // 和游戏逻辑相关的一些远程网址。
        private const string GameEndpointBase = "https://goldennumber.aiedu.msra.cn/api/";
        private const string StateEndpointTemplate = GameEndpointBase + "state?uid={0}&roomid={1}";

        //TODO 使用分页获取历史的接口。
        private const string HistoryEndpointTemplate = GameEndpointBase + "history?roomid={0}";
        private const string SubmitEndpointTemplate = GameEndpointBase + "submit?uid={0}&rid={1}&n1={2}&n2={3}";

        // 总是“人类玩家”模式，一轮60s。
        private const string NewRoomEndpointTemplate = GameEndpointBase + "newroom?numbers={0}";
        private const string NicknameEndpointTemplate = GameEndpointBase + "nickname?uid={0}&nickname={1}";

        // 当前设计下支持的数字模式。只支持提交一个或两个数字。
        private static readonly Dictionary<RoomNumberMode, string> SupportedNumberModes = new Dictionary<RoomNumberMode, string>
        {
            [RoomNumberMode.One] = "1",
            [RoomNumberMode.Two] = "2",
        };

        // HttpClient 的方法全都是线程安全的，可以通过静态实例复用。
        private static readonly HttpClient s_httpClient = new HttpClient();

        /// <summary>
        /// 当前进行游戏的玩家的 ID。
        /// </summary>
        public string UserId { get; private set; }

        /// <summary>
        /// 当前游戏所在的房间号。
        /// </summary>
        public string RoomId { get; private set; }

        /// <summary>
        /// 当前进行游戏的玩家的昵称。
        /// </summary>
        public string Nickname { get; private set; }

        /// <summary>
        /// 当前游戏支持提交的数字个数。
        /// </summary>
        public RoomNumberMode NumberMode { get; private set; }

        /// <summary>
        /// 获取当前 Game 实例是否已被关闭了。
        /// </summary>
        public bool IsClosed => _closed;

        /// <summary>
        /// 在后台线程触发。
        /// 每当我们觉得合适更新倒计时的显示了，这个事件就触发。
        /// 传给事件处理程序的参数是：触发事件的 Game 实例，和该轮提交剩下的时间。
        /// 时间部分用标准的 <see cref="TimeSpan"/>类，既解耦，又清晰。
        /// </summary>
        public event TypedEventHandler<Game, TimeSpan> CountdownTicked;

        /// <summary>
        /// 在后台线程触发。
        /// 提示新的一轮提交开始了。
        /// </summary>
        public event TypedEventHandler<Game, Round> NewRoundEntered;

        /// <summary>
        /// 在后台线程触发。
        /// 提示核心逻辑内有错误发生。第二个参数是错误信息。
        /// </summary>
        public event TypedEventHandler<Game, string> ErrorOccurred;

        // 这两个都是 volatile 变量。并且 C# 保证了这些变量的读写都是原子的（做不到这点的类型不能声明为volatile）。
        // 由于这个类利用了线程池线程，所以通过 volatile 让变量的变化在各个线程立即可见。
        private volatile string _roundId;
        private volatile bool _closed = false;

        /// <summary>
        /// 开启一个新游戏。
        /// </summary>
        /// <param name="roomIdToStart">这个游戏实例要连接到的房间号。若为null或空，则为默认的0号房间。</param>
        /// <param name="userId">参与这个游戏的玩家ID。若为null或空，会生成新ID。</param>
        /// <returns>创建游戏的操作结果。</returns>
        public static async Task<GameOperation<CreateGameResult>> OpenRoomAsync(string roomIdToStart, string userId)
        {
            // ConfigureAwait(false) 可以避免对特定的线程进行调度，比如只能序列化访问的UI线程。
            // 该函数内的操作都不需要 UI 线程。
            var stateOp = await QueryStateAsync(userId, roomIdToStart).ConfigureAwait(false);
            if (!stateOp.Succeeded)
            {
                // 如果查询状态失败了，创建游戏也就失败了。
                return GameOperation<CreateGameResult>.Fail(stateOp.ErrorMessage);
            }

            var state = stateOp.OperationResult;
            if (string.IsNullOrEmpty(state.UserId))
            {
                // User ID 是必要的。
                return GameOperation<CreateGameResult>.Fail("No valid User ID.");
            }

            var mode = ConvertNumberMode(state.Numbers);
            if (mode == RoomNumberMode.Unknown)
            {
                // 不支持的游戏模式。
                return GameOperation<CreateGameResult>.Fail($"Unsupported number mode {mode}.");
            }

            var game = new Game
            {
                UserId = state.UserId,
                RoomId = state.RoomId,
                Nickname = state.NickName,
                NumberMode = mode,
            };

            // 启动游戏主循环，用于倒计时和推进游戏轮数。
            // 由于我们用 async/await的形式包装了主循环，故其返回 Task。不过我们不需要用这个 Task。
            var ignored = game.StartAsync(state.RoundId, state.LeftTime);
            return GameOperation<CreateGameResult>.Succ(new CreateGameResult(game, TimeSpan.FromSeconds(state.LeftTime), ConvertStateToNewRound(state)));
        }

        /// <summary>
        /// 创建一个新的游戏房间，并在其中开启新游戏。
        /// </summary>
        /// <param name="mode"></param>
        /// <param name="userId">可选。要继承的玩家ID。</param>
        /// <returns>创建游戏的操作结果。</returns>
        public static async Task<GameOperation<CreateGameResult>> StartInNewRoomAsync(RoomNumberMode mode, string userId = null)
        {
            if (!SupportedNumberModes.ContainsKey(mode))
            {
                throw new ArgumentOutOfRangeException(nameof(mode));
            }

            // 先创建新房间，再在这个房间里，按正常流程启动游戏。
            var url = string.Format(NewRoomEndpointTemplate, SupportedNumberModes[mode]);
            var newRoomOp = await OperationFromResponseAsync(s_httpClient.GetAsync(url), JsonConvert.DeserializeObject<NewRoom>);
            if (!newRoomOp.Succeeded)
            {
                return GameOperation<CreateGameResult>.Fail(newRoomOp.ErrorMessage);
            }

            var newRoom = newRoomOp.OperationResult;
            return await OpenRoomAsync(roomIdToStart: newRoom.RoomId, userId: userId);
        }

        /// <summary>
        /// 获取当前游戏的提交历史。
        /// </summary>
        /// <returns>获取历史的操作结果。</returns>
        //TODO History类是服务器接口类，需要考虑将其替换为合适的核心类型。
        public async Task<GameOperation<History>> GetHistoryAsync()
        {
            EnsureGameNotClosed();

            var url = string.Format(HistoryEndpointTemplate, RoomId);
            return await OperationFromResponseAsync(
                s_httpClient.GetAsync(url),
                JsonConvert.DeserializeObject<History>);
        }

        /// <summary>
        /// 检查并提交这一轮的黄金点。
        /// </summary>
        /// <param name="candidate">要提交的数。</param>
        /// <param name="candidate2">要提交的第二个数。在不支持两个数的游戏里，必须是null。</param>
        /// <returns>提交黄金点的操作结果。</returns>
        public async Task<GameOperation<bool>> SubmitAsync(double candidate, double? candidate2)
        {
            EnsureGameNotClosed();

            // _roundId 可能由于多线程而变动。我们先复制它到局部变量。
            var roundId = _roundId;
            if (!(0 < candidate && candidate < 100))
            {
                return GameOperation<bool>.Fail("Input must be in (0, 100)");
            }

            if (candidate2.HasValue)
            {
                if (NumberMode != RoomNumberMode.Two)
                {
                    return GameOperation<bool>.Fail("Not 2-number room.");
                }

                if (!(0 < candidate2 && candidate2 < 100))
                {
                    return GameOperation<bool>.Fail("Secondary input must be in (0, 100)");
                }
            }

            string submitUrl = string.Format(
                SubmitEndpointTemplate,
                Uri.EscapeDataString(UserId),
                Uri.EscapeDataString(roundId),
                Uri.EscapeDataString(candidate.ToString()),
                Uri.EscapeDataString(candidate2?.ToString() ?? "")); // 如果游戏模式不是提交两个数，就将第二个数对应的参数设为空。

            var dummyBody = new StringContent("");
            return await OperationFromResponseAsync(
                s_httpClient.PostAsync(submitUrl, dummyBody),
                unused => true);
        }

        /// <summary>
        /// 更改当前玩家的昵称。
        /// </summary>
        /// <param name="newNickname">新昵称</param>
        /// <returns>更改昵称的操作结果。</returns>
        public async Task<GameOperation<string>> ChangeNicknameAsync(string newNickname)
        {
            var url = string.Format(NicknameEndpointTemplate, UserId, Uri.EscapeDataString(newNickname));
            var dummyBody = new StringContent("");
            var op = await OperationFromResponseAsync(
                s_httpClient.PostAsync(url, dummyBody),
                unused => newNickname);
            if (op.Succeeded)
            {
                Nickname = newNickname;
            }

            return op;
        }

        /// <summary>
        /// 关闭当前游戏，当前的 Game 实例将不再可用。
        /// 远程的游戏状态不会改变。
        /// </summary>
        public void Close()
        {
            // 置位信号，表示当前 Game 对象被关闭了。
            //
            // 由于当前实现中使用了多线程，所以其他线程上运行的 Game 类的方法可能还在执行中。
            // 对我们目前的实现来说这没有问题，我们只需在敏感操作前检查并结束那个操作即可。其他正在执行的代码，可能还会继续执行一会儿。
            _closed = true;

            // 清除所有注册的事件处理器。
            //
            // 事件背后的、用于储存所有事件处理器的列表，实际上是一个不可变（immutable）对象。
            // 我们在这里原子地将事件置空，并不影响其他线程上的、在这之前就创建的副本（基本都是在调用 SafeQueueEvent 时做的拷贝）。所以这个置空是安全的。
            //
            // 如果我们不在这之后，继续执行未能解绑的事件处理器，我们需要在那里判断 _closed 的值。
            CountdownTicked = null;
            NewRoundEntered = null;
            ErrorOccurred = null;
        }

        /// <summary>
        /// 游戏主循环。
        /// </summary>
        /// <param name="roundId">当前这一轮的ID。</param>
        /// <param name="countdown">当前这一轮剩余的秒数。</param>
        /// <returns>代表游戏主循环的 Task。</returns>
        private async Task StartAsync(string roundId, int countdown)
        {
            try
            {
                _roundId = roundId;

                while (true)
                {
                    if (_closed)
                    {
                        //TODO 最好通过 CancellationToken 的形式将游戏结束的信号传递到下方的网络请求中，
                        //TODO 可以及时结束不必要的网络请求。
                        return;
                    }

                    // 倒数剩余秒数
                    while (countdown > 0)
                    {
                        // 这里异步地延迟 1s。稍有一些误差。
                        await Task.Delay(1000).ConfigureAwait(false);
                        countdown--;

                        // 提示外部组件更新倒计时。
                        RaiseCountdown(countdown);
                    }

                    // 尝试查询一轮的状态。
                    State thisRoundState;
                    while (true)
                    {
                        if (_closed)
                        {
                            return;
                        }

                        var queryOp = await QueryStateAsync(UserId, RoomId);
                        if (queryOp.Succeeded)
                        {
                            thisRoundState = queryOp.OperationResult;
                            if (thisRoundState != null && thisRoundState.RoundId != roundId)
                            {
                                // 查询成功，并且新一轮的 ID 和当前的不一样，说明我们确实接触到了新一轮游戏。
                                break;
                            }
                        }

                        // 查询失败、查询过早等等，200ms后重试。
                        await Task.Delay(200);
                    }

                    countdown = thisRoundState.LeftTime;
                    roundId = thisRoundState.RoundId;

                    _roundId = roundId;

                    // 通知外部，进入新一轮的提交。
                    SafeQueueEvent(NewRoundEntered, ConvertStateToNewRound(thisRoundState));

                    // 我们希望倒计时和新一轮的通知是正交的。
                    RaiseCountdown(countdown);
                }
            }
            catch (AggregateException ae)
            {
                SafeQueueEvent(ErrorOccurred, ae.InnerException.Message);
            }
            catch (Exception ex)
            {
                SafeQueueEvent(ErrorOccurred, ex.Message);
            }
        }

        private void RaiseCountdown(int cd)
        {
            // 尽管在 Game 类内部我们只用秒做单位，但我们希望对外部解耦这一假设。
            SafeQueueEvent(CountdownTicked, TimeSpan.FromSeconds(cd));
        }

        private void SafeQueueEvent<A>(TypedEventHandler<Game, A> evt, A arg)
        {
            // 发送事件是 Game 类唯一对外界造成的影响。如果已经关闭游戏，我们就不要造成这个影响。
            if (_closed)
            {
                return;
            }

            // 为了不影响主循环的倒计时，我们把可能需要很长时间执行的事件处理器，放到其他线程里去处理。
            evt.SafeInvoke(e => Task.Run(() =>
            {
                // 可能事件已经发出了，但还没有在后台线程上被执行。如果在这之前关闭了游戏，我们就放弃这次执行。
                if (_closed)
                {
                    return;
                }

                e(this, arg);
            }));

        }

        /// <summary>
        /// 有些游戏相关的操作，在游戏关闭后就不能执行。我们通过这个函数来包装这种检查。
        /// </summary>
        private void EnsureGameNotClosed()
        {
            if (_closed)
            {
                throw new InvalidOperationException("This game has been closed.");
            }
        }

        /// <summary>
        /// 查询当前房间的状态。
        /// </summary>
        /// <param name="uid">玩家ID。不提供的话服务器将分配新ID。</param>
        /// <param name="roomIdToStart">要查询的房间号。可选，默认是0号。</param>
        /// <returns>查询房间状态的操作结果。</returns>
        private static async Task<GameOperation<State>> QueryStateAsync(string uid = null, string roomIdToStart = null)
        {
            // ?? 操作符的作用是，对表达式 a ?? b，当 a 的值为 null 时，整个表达式的值就是 b。相当于 if (a == null) 的简化写法。
            var url = string.Format(StateEndpointTemplate, Uri.EscapeDataString(uid ?? ""), Uri.EscapeDataString(roomIdToStart ?? ""));
            return await OperationFromResponseAsync(
                s_httpClient.GetAsync(url),
                JsonConvert.DeserializeObject<State>);
        }

        /// <summary>
        /// 辅助子过程。检查异步的网络请求是否成功、对服务器的操作请求是否实现，并返回实现了的操作的结果，或错误信息。
        /// </summary>
        /// <typeparam name="T">代表结果数据的泛型参数。</typeparam>
        /// <param name="respTask">代表异步的网络请求。</param>
        /// <param name="deserializer">对网络请求结果的反序列化器。</param>
        /// <returns>操作的结果。</returns>
        private static async Task<GameOperation<T>> OperationFromResponseAsync<T>(Task<HttpResponseMessage> respTask, Func<string, T> deserializer)
        {
            string msg = "";
            try
            {
                var resp = await respTask.ConfigureAwait(false);

                // 如果是 404 BadRequest，我们最好看看服务器提供了哪些错误信息。
                if (resp.IsSuccessStatusCode || resp.StatusCode == HttpStatusCode.BadRequest)
                {
                    var respStr = await resp.Content.ReadAsStringAsync();
                    if (resp.IsSuccessStatusCode)
                    {
                        return GameOperation<T>.Succ(deserializer(respStr));
                    }
                    else
                    {
                        // 404。服务器会将错误信息放在 JSON 数据的 Message 字段里。
                        var deserialized = JsonConvert.DeserializeAnonymousType(respStr, new { Message = "" });
                        msg = deserialized.Message;
                    }
                }
            }
            catch (Exception ex)
            {
                msg = ex.Message;
            }

            return GameOperation<T>.Fail(msg);
        }

        private static RoomNumberMode ConvertNumberMode(string mode)
        {
            // 没有找到的话，FirstOrDefault 中的 default 将是 RoomNumberMode.Unknown。
            return SupportedNumberModes.FirstOrDefault(kv => kv.Value == mode).Key;
        }

        private static Round ConvertStateToNewRound(State state)
        {
            return new Round(state.RoundId, ConvertNumberMode(state.Numbers));
        }

        // 我们只能通过指定的静态方法创建 Game 类。
        private Game()
        { }
    }
}