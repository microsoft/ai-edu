// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BotRun;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace LocalManager
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();

            Reset();
        }

        // 参加比赛的Bot列表
        private List<Bot> botList = new List<Bot>();

        // 比赛数据表，每一行是一回合的数据，每行的第一列是回合编号，第二列是该回合的黄金点值，后面依次是每个Bot的两个预测值
        DataTable dtHistory;

        // Bot得分表
        DataTable dtScoreHistory;
        // 排序后的Bot得分
        DataView dvScoreHistory;

        // 比赛中已经进行的回合数
        int roundNum = 0;

        // 控制比赛中强制停止比赛
        bool bStop = false;

        // 重置比赛
        private void Reset()
        {
            botList.Clear();

            historyView.DataSource = null;
            scoreView.DataSource = null;
            chartScore.DataSource = null;
            chartScore.Series.Clear();
            chartGoldenNumber.DataSource = null;
            chartGoldenNumber.Series.Clear();

            dtHistory = null;
            dvScoreHistory = null;
            dtScoreHistory = null;

            roundNum = 0;
            tbLog.Clear();

            btnPlayMultiRound.Enabled = false;
            btnStop.Enabled = false;
            btnExport.Enabled = false;
        }

        // 选择Bots所在的目录，并初始化相关数据
        private void btnSelectBots_Click(object sender, EventArgs e)
        {
            // 默认选中内置的三个测试Bot
            var selectedPath = Application.StartupPath + "\\SampleBots";

            FolderBrowserDialog dialog = new FolderBrowserDialog();
            dialog.SelectedPath = selectedPath;
            var dialogResult = dialog.ShowDialog();
            if (!dialogResult.Equals(DialogResult.OK))
            {
                return;
            }

            if (string.IsNullOrEmpty(dialog.SelectedPath))
            {
                return;
            }

            Reset();

            selectedPath = dialog.SelectedPath;

            // Bot从1开始编号，找到有效的Bot，加入到Bot列表中
            int index = 1;
            var dirs = Directory.GetDirectories(selectedPath);
            foreach (var dir in dirs)
            {
                bool isSupportPython = Bot.IsSupportPython();
                bool isSupportJulia = Bot.IsSupportJulia();

                var exeFiles = Directory.GetFiles(dir, "*.exe");
                if (exeFiles.Length == 0)
                {
                    var getNumbersPyFile = Path.Combine(dir, "get_numbers.py");
                    var getNumbersJuliaFile = Path.Combine(dir, "get_numbers.jl");
                    if (isSupportPython && File.Exists(getNumbersPyFile))
                    {
                        botList.Add(new Bot(getNumbersPyFile, index++) { ActionLog = Log });
                    }
                    else if (isSupportJulia && File.Exists(getNumbersJuliaFile))
                    {
                        botList.Add(new Bot(getNumbersJuliaFile, index++) { ActionLog = Log });
                    }
                    else
                    {
                        Log($"警告：跳过 \"{dir}\" 没有找到exe{(isSupportPython ? "或者get_numbers.py" : "")}{(isSupportJulia ? "或者get_numbers.jl" : "")}");
                    }
                    continue;
                }
                else if (exeFiles.Length == 1)
                {
                    botList.Add(new Bot(exeFiles.First(), index++) { ActionLog = Log });
                    continue;
                }
                else if (exeFiles.Length > 1)
                {
                    Log($"警告：跳过 \"{dir}\" 找到多个exe");
                    continue;
                }
            }

            // 将Bot编号及Bot路径输出在日志窗口
            var botListString = string.Join("\r\n", botList.Select(bot => $"{bot.Index} --> {bot.FilePath}"));
            Log("找到如下Bot：\r\n" + botListString);

            // 初始化比赛数据表
            dtHistory = new DataTable("history");
            dtHistory.Columns.Add(new DataColumn("回合编号", typeof(int)));
            dtHistory.Columns.Add(new DataColumn("黄金点", typeof(double)));
            for (int i = 1; i <= botList.Count; i++)
            {
                dtHistory.Columns.Add(new DataColumn($"Bot{i.ToString("D3")}_1", typeof(double)));
                dtHistory.Columns.Add(new DataColumn($"Bot{i.ToString("D3")}_2", typeof(double)));
            }
            historyView.DataSource = dtHistory.DefaultView;

            // 初始化得分表
            dtScoreHistory = new DataTable("scoreHistory");
            dtScoreHistory.Columns.Add(new DataColumn("Bot编号", typeof(int)));
            dtScoreHistory.Columns.Add(new DataColumn("得分", typeof(int)));
            for (int i = 0; i < botList.Count; i++)
            {
                var dr = dtScoreHistory.NewRow();
                dr["Bot编号"] = botList[i].Index;
                dr["得分"] = 0;
                dtScoreHistory.Rows.Add(dr);
            }
            dvScoreHistory = new DataView(dtScoreHistory);
            dvScoreHistory.Sort = "得分 DESC";
            scoreView.DataSource = dvScoreHistory;

            // 得分图
            chartScore.DataSource = dtScoreHistory;
            chartScore.Series.Add("得分");
            chartScore.Series[0].XValueMember = "Bot编号";
            chartScore.Series[0].YValueMembers = "得分";
            chartScore.ChartAreas[0].AxisX.Title = "Bot编号";
            chartScore.ChartAreas[0].AxisY.Title = "得分";
            chartScore.DataBind();

            // 黄金点走势图
            chartGoldenNumber.DataSource = dtHistory;
            chartGoldenNumber.Series.Add("黄金点");
            chartGoldenNumber.Series[0].ChartType = SeriesChartType.Spline;
            chartGoldenNumber.Series[0].XValueMember = "回合编号";
            chartGoldenNumber.Series[0].YValueMembers = "黄金点";
            chartGoldenNumber.ChartAreas[0].AxisX.Title = "回合编号";
            chartGoldenNumber.ChartAreas[0].AxisY.Title = "黄金点";
            chartGoldenNumber.DataBind();

            btnPlayMultiRound.Enabled = true;
            btnStop.Enabled = true;
            btnExport.Enabled = true;
        }

        readonly static object botIndexLocker = new object();
        private int nextBotIndex = 0;

        private async Task RunBotsAsync(string historyString, int roundNum, ConcurrentBag<BotRunResult> results)
        {
            while (nextBotIndex < botList.Count)
            {
                Bot bot = null;
                lock (botIndexLocker)
                {
                    if (nextBotIndex < botList.Count)
                    {
                        bot = botList[nextBotIndex++];
                    }
                }

                if (bot == null)
                {
                    return;
                }

                var result = await bot.RunAsync(historyString, roundNum);
                results.Add(result);
            }
        }

        // 运行一回合
        private async Task RunOneRoundAsync()
        {
            if (botList.Count == 0)
            {
                return;
            }

            // 当前回合编号
            int curRoundNum = ++roundNum;

            // 从比赛数据中按约定格式生成输入数据
            var historyString = FormatBotInput(dtHistory);

            // Bot执行结果列表
            ConcurrentBag<BotRunResult> botRunResultList = new ConcurrentBag<BotRunResult>();

            List<Task> tasks = new List<Task>();
            nextBotIndex = 0;
            for (int i = 0; i < Environment.ProcessorCount; i++)
            {
                tasks.Add(RunBotsAsync(historyString, curRoundNum, botRunResultList));
            }

            // 等待所有Bot执行完成
            await Task.WhenAll(tasks);

            // Bot输出的预测值列表
            var numberList = new List<double>();
            foreach (var result in botRunResultList.OrderBy(result => result.Bot.Index))
            {
                numberList.Add(result.MasterValue);
                numberList.Add(result.SlaveValue);
            }

            // 排除无效值，然后再计算本回合黄金点值
            var validNumberList = numberList.Where(value => value > 0).ToList();
            var average = validNumberList.Average();
            var goldenNumber = average * 0.618;

            // 当前回合提交有效数据的Bot个数
            int validBotCount = validNumberList.Count / 2;

            // 将数据写入比赛数据表中
            List<object> data = new List<object>();
            data.Add(curRoundNum);
            data.Add(goldenNumber);
            data.AddRange(numberList.OfType<object>());
            DataRow dr = dtHistory.NewRow();
            dr.ItemArray = data.ToArray();
            dtHistory.Rows.Add(dr);
            historyView.FirstDisplayedScrollingRowIndex = historyView.RowCount - 1;

            Log($"第{curRoundNum}回合黄金点为 {goldenNumber} ，各Bot输出： " + String.Join(" ", numberList));

            // 计算得分

            // 将有效预测值按照到黄金点的距离分组并按距离排序
            var scoreResult = botRunResultList.Where(result => result.MasterValue != 0)
                .Select(result => new KeyValuePair<double, Bot>(Math.Abs((double)result.MasterValue - goldenNumber), result.Bot))
                .ToList();
            scoreResult.AddRange(botRunResultList.Where(result => result.SlaveValue != 0)
                .Select(result => new KeyValuePair<double, Bot>(Math.Abs((double)result.SlaveValue - goldenNumber), result.Bot))
                .ToList());
            var scoreResultGroup = scoreResult.GroupBy(item => item.Key)
                .OrderBy(item => item.Key)
                .ToList();

            if (scoreResultGroup.Count != 0)
            {
                // 找到最近的和最远的，分别计分
                var winnerGroup = scoreResultGroup.First().Select(item => item.Value).Distinct().ToList();
                var loserGroup = scoreResultGroup.Last().Select(item => item.Value).Distinct().ToList();

                foreach (var bot in botList)
                {
                    int score = 0;
                    if (winnerGroup.Contains(bot))
                    {
                        score += validBotCount;
                    }
                    if (loserGroup.Contains(bot))
                    {
                        score -= 2;
                    }
                    bot.ScoreHistory.Add(score);

                    dtScoreHistory.Rows[bot.Index - 1]["得分"] = bot.ScoreHistory.Sum();
                }
            }

            chartScore.DataBind();
            chartGoldenNumber.DataBind();

            // 等待UI刷新
            await Task.Delay(1);
        }

        // 为Bot数据加上meta头，指明行列数
        private string FormatBotInput(DataTable dt)
        {
            int rowCount = dt?.Rows?.Count ?? 0;
            int columnCount = 2 * botList.Count + 1;

            return $"{rowCount}\t{columnCount}\r\n{FormatHistoryDataTable(dt)}";
        }

        // 格式化比赛数据
        private string FormatHistoryDataTable(DataTable dt)
        {
            string tableString = string.Empty;

            if (dt?.Rows?.Count == 0)
            {
                return tableString;
            }

            for (int i = 0; i < dt.Rows.Count; i++)
            {
                string rowString = FormatHistoryDataRow(dt.Rows[i]);
                tableString = $"{tableString}{rowString}\r\n";
            }

            return tableString;
        }

        // 格式化每一回合的比赛数据
        private string FormatHistoryDataRow(DataRow dr)
        {
            string rowString = string.Empty;

            if (dr?.ItemArray?.Length == 0)
            {
                return rowString;
            }

            return String.Join("\t", dr.ItemArray.OfType<double>());
        }

        // 比赛指定回合数
        private async void btnPlayMultiRound_Click(object sender, EventArgs e)
        {
            btnSelectBots.Enabled = false;
            btnPlayMultiRound.Enabled = false;
            bStop = false;
            int rounds = 0;
            int.TryParse(tbRounds.Text, out rounds);

            if (rounds > 0)
            {
                for (int i = 0; i < rounds; i++)
                {
                    if (!bStop)
                    {
                        await RunOneRoundAsync();
                    }
                }
            }
            btnPlayMultiRound.Enabled = true;
            btnSelectBots.Enabled = true;
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            bStop = true;
        }

        // 导出结果数据
        private void btnExport_Click(object sender, EventArgs e)
        {
            try
            {
                var historyString = FormatBotInput(dtHistory);

                SaveFileDialog saveFileDialog = new SaveFileDialog();
                saveFileDialog.Filter = "Text|*.txt";
                saveFileDialog.Title = "导出比赛结果";
                saveFileDialog.ShowDialog();

                if (saveFileDialog.FileName != "")
                {
                    File.WriteAllText(saveFileDialog.FileName, historyString);

                    try
                    {
                        File.WriteAllText(saveFileDialog.FileName + ".log", tbLog.Text);

                        var scoreString = string.Join(
                            "\r\n",
                            botList.Select(bot => $"{bot.Index}\t{bot.ScoreHistory.Sum()}\t{bot.FilePath}").ToList());
                        File.WriteAllText(saveFileDialog.FileName + ".score.log", "Bot编号\t得分\tBot路径\r\n" + scoreString);
                    }
                    catch (Exception)
                    {
                    }
                }
            }
            catch (Exception)
            {
            }
        }

        private void Log(string message)
        {
            var msg = $"[{DateTime.Now.ToString("HH:mm:ss:fff")}] {message}\r\n";

            Action log = () =>
            {
                tbLog.AppendText(msg);
            };

            RunInUI(log);
        }

        private void RunInUI(Action action)
        {
            if (action == null)
            {
                return;
            }

            try
            {
                if (InvokeRequired)
                {
                    MethodInvoker method = new MethodInvoker(action);
                    Invoke(action, null);
                }
                else
                {
                    action();
                }
            }
            catch (Exception)
            {
            }
        }
    }
}