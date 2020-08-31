// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using GoldedNumberClient.Models;
using GoldedNumberClient.Utils;
using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;

namespace GoldedNumberClient
{
    /// <summary>
    /// MainWindow.xaml 的交互逻辑
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            LogTextBox.TextChanged += Log_TextChanged;
            SubmitButton.Click += Submit_Click;

            SwitchRoomButton.Click += SwitchRoomButton_Click;
            NewRoomButton.Click += NewRoomButton_Click;
            NewTwoNumberRoomButton.Click += NewTwoNumberRoomButton_Click;

            SetNicknameButton.Click += SetNicknameButton_Click;

            // 异步创建初始的 Game 对象，连接到默认的0号房间。如果初始化失败，该程序就退出。
            HandleGameCreationAsync(Game.OpenRoomAsync(roomIdToStart: null, userId: null)).ContinueWith(handle =>
            {
                // 需要在 UI 线程上执行。
                Dispatcher.InvokeAsync(() =>
                {
                    if (handle.Status != TaskStatus.RanToCompletion || !handle.Result.Succeeded)
                    {
                        string msg = "";
                        if (handle.Status != TaskStatus.RanToCompletion)
                        {
                            // 因为异步执行出错了。如未捕获的异常。
                            // 先看看异常是不是 AggregateException，是的话就取其 InnerException 的错误信息，否则就直接取异常的错误信息。
                            msg = ((handle.Exception as AggregateException)?.InnerException ?? handle.Exception).Message;
                        }
                        else
                        {
                            // 某个 GameOperation 报告了错误……
                            msg = handle.Result.ErrorMessage;
                        }

                        MessageBox.Show(
                            "Game failed to start! Try restart this app. " + msg,
                            "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                        Application.Current.Shutdown();
                        return;
                    }
                });
            });
        }


        /// <summary>
        /// 当前进行中的游戏对象。
        /// </summary>
        private Game _game;

        /// <summary>
        /// 提取出 <see cref="Game">对象创建（加入房间或创建新房间）成功后需要执行的操作。创建成功的话就初始化界面。
        /// 并将创建结果向后传递，以方便 <see cref="Game"/>的调用逻辑进行后续处理。
        /// </summary>
        /// <param name="creationTask">异步的游戏创建操作。</param>
        /// <returns>游戏创建操作的结果。</returns>
        private async Task<GameOperation<CreateGameResult>> HandleGameCreationAsync(Task<GameOperation<CreateGameResult>> creationTask)
        {
            Dispatcher.VerifyAccess(); // We has reached UI thread.

            var creation = await creationTask;
            if (creation.Succeeded)
            {
                OnGameCreationSucceeded(creation.OperationResult);
            }

            return creation;
        }

        /// <summary>
        /// <see cref="Game"/>对象创建成功后，进行初始化或修改界面的操作。
        /// </summary>
        /// <param name="creation">游戏创建操作的结果。</param>
        private async void OnGameCreationSucceeded(CreateGameResult creation)
        {
            // 如果有在进行游戏，需要关闭它。以防止其继续推送游戏事件或造成对象泄漏。
            _game?.Close();
            _game = creation.Game;

            //TODO 对这里的一些细节操作，我们可以再加抽象。需要注意的是变量 g，这个 g 和 _game 字段是不可混用的。
            //TODO 因为 g 真正表示了在过去时间点上，发送了事件的 Game 对象。而通过 _game，我们总是取得当前正在进行的 Game 对象。
            //TODO 我们在这里显式地、重复地使用 g，就很容易不小心混用了。
            //TODO 我们可以把 `_game.SomeEvent += aHandler` 操作抽象成 Action，并统一地管理变量 g 的传递，这样我们就能防患于未然。
            //
            // Game 类的事件不一定在 UI 线程触发，我们需要包装所有涉及 UI 的操作，使其在UI线程进行。
            // 否则对UI控件的操作会抛异常（这是 WPF 作为界面框架的一种特性）。
            // PostAsync 会将操作转移到 UI 线程执行，这个执行是异步的。
            _game.CountdownTicked += (g, cd) => PostAsync(g, () => UpdateCountdown(cd.Seconds));
            _game.NewRoundEntered += (g, a) => PostAsync(g, () => OnNewRound(g, a));
            _game.ErrorOccurred += (g, s) => PostAsync(g, () => WriteLog(s + " Please restart game."));

            // 将界面调整到运行状态。
            MainGrid.IsEnabled = true;
            Blocker.Visibility = Visibility.Collapsed;
            WriteLog($"=== Entered room {creation.Game.RoomId} ({creation.Game.NumberMode}-number) ===");
            WriteLog($"Current user ID: {_game.UserId}");
            WriteLog($"Current round: {creation.InitialRound.RoundId}");
            UpdateCountdown(creation.InitialCountdown.Seconds);

            bool twoNumbers = creation.Game.NumberMode == RoomNumberMode.Two;
            Number2InputTextBox.Visibility = twoNumbers ? Visibility.Visible : Visibility.Collapsed;
            NumberInputTextBox.Width = twoNumbers ? 40 : 100;

            var nickname = creation.Game.Nickname;
            if (nickname != null)
            {
                NicknameTextBox.Text = nickname;
            }

            await ShowHistoryAsync();
        }

        private async void OnNewRound(Game game, Round arg)
        {
            // 判断是不是当前正在进行的游戏。以防止在 PostAsync 将消息发送到 UI 线程过程中导致历史重载。
            if (game == _game)
            {
                WriteLog($"New round: {arg.RoundId}");
                await ShowHistoryAsync();
            }
        }

        // 在表格上显示各种历史分数。
        private async Task ShowHistoryAsync()
        {
            HistoryMask.Visibility = Visibility.Visible;
            var historyOp = await _game.GetHistoryAsync();

            if (historyOp.Succeeded)
            {
                var history = historyOp.OperationResult;

                // 这里的 ? 操作符可以在前方的变量为 null 时，跳过后续的流程，防止产生 NullReferenceException。
                HistoryView.ItemsSource = history.Rounds?.Select(round => new
                {
                    // 这些匿名对象的字段名，需要和 XAML 文件中设定的数据绑定保持一致。
                    Time = $"{round.Time:MM/dd HH:mm:ss}",
                    GoldenNumber = round.GoldenNumber
                });

                // 最多显示五轮的结果。
                LastRoundScoreView.ItemsSource = history.Rounds?.Take(5).SelectMany(round => round?.UserNumbers?.Select(num => new
                {
                    RoundTime = $"{round.Time:MM/dd HH:mm:ss}",
                    Nickname = history.NickNames[num.UserId],
                    UserId = num.UserId,
                    Number1 = num.MasterNumber,
                    Number2 = num.SlaveNumber,
                    Score = num.Score,
                }));

                HistoryMask.Visibility = Visibility.Collapsed;
            }
        }

        private void UpdateCountdown(int second)
        {
            CountdownLabel.Text = $"{second}s";
        }

        /// <summary>
        /// 确保操作在 UI 线程执行。
        /// </summary>
        /// <param name="sender">
        ///   作为事件源头的 <see cref="Game"/> 对象。
        ///   如果该对象已经被关闭（<see cref="Game.IsClosed"/>）了，<paramref name="action"/>就不会被执行。</param>
        /// <param name="action">要执行的操作。</param>
        /// <returns>调度的结果。</returns>
        private DispatcherOperation PostAsync(Game sender, Action action)
        {
            return Dispatcher.InvokeAsync(() =>
            {
                // 到这里，我们已经到达 UI 线程了，并且已经发生了的执行调度都是异步的。
                // 因为是异步的，所以作为消息源头的 Game 对象可能已经关闭了。这时候，我们就放弃自它而生的相关操作。
                if (!sender.IsClosed)
                {
                    action();
                }
            });
        }

        private async void Submit_Click(object sender, RoutedEventArgs e)
        {
            string msg = "";

            do // 通过 do while (false) + break 的形式，在出错时跳转到最后，只进行错误提示。
            {
                if (!double.TryParse(NumberInputTextBox.Text, out double candidate))
                {
                    msg = $"Input must be number! {NumberInputTextBox.Text}";
                    break;
                }

                double? candidate2 = null;
                if (_game.NumberMode == RoomNumberMode.Two)
                {
                    if (double.TryParse(Number2InputTextBox.Text, out double n2))
                    {
                        candidate2 = n2;
                    }
                    else
                    {
                        msg = $"Secondary input must be number! {Number2InputTextBox.Text}";
                        break;
                    }
                }

                var result = await _game.SubmitAsync(candidate, candidate2);
                if (result.Succeeded)
                {
                    string submitted = $"Submitted {candidate}";
                    if (candidate2.HasValue)
                    {
                        submitted += $" and {candidate2.Value}";
                    }

                    WriteLog(submitted);
                    NumberInputTextBox.Text = "";
                    Number2InputTextBox.Text = "";
                    return;
                }
                else
                {
                    msg = $"Failed to submit: {result.ErrorMessage}";
                    break;
                }
            } while (false);

            MessageBox.Show(msg, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }

        private async void SwitchRoomButton_Click(object sender, RoutedEventArgs e)
        {
            if (RoomIdTextBox.Text == _game.RoomId)
            {
                MessageBox.Show($"You are already in room {_game.RoomId}!", "Switch stopped");
                return;
            }

            if (!string.IsNullOrEmpty(RoomIdTextBox.Text))
            {
                string msg = "";
                bool failed = true;
                try
                {
                    // 切换房间可能因为房间不存在而失败。我们不为这种失败结束程序。
                    var op = await HandleGameCreationAsync(Game.OpenRoomAsync(RoomIdTextBox.Text, _game.UserId));
                    failed = !op.Succeeded;
                    if (failed)
                    {
                        msg = op.ErrorMessage;
                    }
                }
                catch (Exception ex)
                {
                    msg = ((ex as AggregateException)?.InnerException ?? ex).Message;
                }

                if (failed)
                {
                    MessageBox.Show(
                        "Failed to switch room! " + msg,
                        "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }

            RoomIdTextBox.Text = "";
        }

        private async void NewRoomButton_Click(object sender, RoutedEventArgs e)
        {
            await HandleGameCreationAsync(Game.StartInNewRoomAsync(RoomNumberMode.One, _game.UserId));
        }

        private async void NewTwoNumberRoomButton_Click(object sender, RoutedEventArgs e)
        {
            await HandleGameCreationAsync(Game.StartInNewRoomAsync(RoomNumberMode.Two, _game.UserId));
        }

        private async void SetNicknameButton_Click(object sender, RoutedEventArgs e)
        {
            var operation = await _game.ChangeNicknameAsync(NicknameTextBox.Text);
            if (operation.Succeeded)
            {
                WriteLog($"Nickname changed to {NicknameTextBox.Text}");
            }
            else
            {
                WriteLog($"Failed to change nickname: {operation.ErrorMessage}");
            }
        }

        private void Log_TextChanged(object sender, TextChangedEventArgs e)
        {
            LogTextBox.ScrollToEnd();
        }

        private void WriteLog(string log)
        {
            if (!log.EndsWith("\n"))
            {
                log += '\n';
            }

            log = $"[{DateTime.Now:HH:mm:ss}] {log}";

            int newLineCount = log.Count(c => c == '\n');
            string existingLog = LogTextBox.Text;

            int toLineCount = newLineCount + LogTextBox.LineCount;
            if (toLineCount > LogTextBox.MaxLines)
            {
                int startIdx = 0;
                for (int i = 0; i < 10; i++)
                {
                    // Line count to remove should be lesser than `MaxLines`.
                    // Even if \n is not found, `IndexOf` returns -1, so `startIdx` becomes 0, that's good.
                    startIdx = existingLog.IndexOf('\n', startIdx) + 1;
                }

                existingLog = existingLog.Substring(startIdx);
            }

            existingLog += log;
            LogTextBox.Text = existingLog;
        }
    }
}