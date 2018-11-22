// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BotRun
{
    public class Bot
    {
        public Bot(string filePath, int index)
        {
            FilePath = filePath;
            Index = index;
            ScoreHistory = new List<int>();
        }

        // Bot exe或py文件的路径
        public string FilePath { get; }

        // Bot编号
        public int Index { get; }

        // 得分历史
        public List<int> ScoreHistory { get; set; }

        // 用来写日志输出
        public Action<string> ActionLog;

        // 限制每个Bot运行5秒
        private const int TIMEOUT = 5000;

        /// <summary>
        /// 运行一次Bot
        /// </summary>
        /// <param name="historyString">历史数据</param>
        /// <param name="roundNum">当前回合编号，仅用于日志输出</param>
        /// <returns></returns>
        public async Task<BotRunResult> RunAsync(string historyString, int roundNum)
        {
            return await Task.Run(() =>
            {
                // 运行Bot
                var f = new Process();
                if (Path.GetExtension(FilePath).Equals(".py", StringComparison.OrdinalIgnoreCase))
                {
                    f.StartInfo.FileName = "cmd.exe";
                    f.StartInfo.Arguments = $"/C python \"{FilePath}\"";
                }
                else if (Path.GetExtension(FilePath).Equals(".jl", StringComparison.OrdinalIgnoreCase))
                {
                    f.StartInfo.FileName = "cmd.exe";
                    f.StartInfo.Arguments = $"/C julia \"{FilePath}\"";
                }
                else
                {
                    f.StartInfo.FileName = FilePath;
                }
                f.StartInfo.WorkingDirectory = Path.GetDirectoryName(FilePath);
                f.StartInfo.UseShellExecute = false;
                f.StartInfo.RedirectStandardError = true;
                f.StartInfo.RedirectStandardInput = true;
                f.StartInfo.RedirectStandardOutput = true;
                f.EnableRaisingEvents = true;
                f.StartInfo.CreateNoWindow = true;

                LogRunInfo("start", roundNum);
                f.Start();

                // 在标准输入上输入历史数据
                try
                {
                    f.StandardInput.Write(historyString);
                    f.StandardInput.Close();
                }
                catch (Exception)
                {
                }

                if (!f.WaitForExit(TIMEOUT))
                {
                    // Bot执行超时将被杀掉，并认为输出了无效值
                    try
                    {
                        LogRunInfo("kill", roundNum);
                        f.Kill();
                    }
                    catch (Exception)
                    {
                    }
                    return new BotRunResult(this, 0, 0);
                }
                else
                {
                    // Bot执行结束时，在标准输出上读取制表符分割的两个数
                    LogRunInfo("exit", roundNum);
                    var line = f.StandardOutput.ReadLine();

                    var error = f.StandardError.ReadLine();
                    if (!string.IsNullOrEmpty(error))
                    {
                        LogRunInfo($"[error] {error}", roundNum);
                    }

                    LogRunInfo($"输出 {line}", roundNum);

                    return new BotRunResult(this, ParseDoubleValues(line));
                }
            }).ConfigureAwait(false);
        }

        // 从输出字符串中解析出两个数
        private Tuple<double, double> ParseDoubleValues(string str)
        {
            if (string.IsNullOrWhiteSpace(str))
            {
                return new Tuple<double, double>(0, 0);
            }

            str = str.Trim(new char[] { ' ', '\r', '\n', '\t' });
            string[] strValues = str.Split('\t');
            double masterValue = 0;
            double slaveValue = 0;

            if (strValues.Length >= 1)
            {
                masterValue = ConvertToDoubleBetween0And100(strValues[0]);
            }

            if (strValues.Length >= 2)
            {
                slaveValue = ConvertToDoubleBetween0And100(strValues[1]);
            }

            // 如果任一输出值无效，认为两个输出值都无效
            if (masterValue == 0 || slaveValue == 0)
            {
                masterValue = 0;
                slaveValue = 0;
            }

            return new Tuple<double, double>(masterValue, slaveValue);
        }

        // 将字符串转换到0到100之间的数
        private double ConvertToDoubleBetween0And100(string str)
        {
            double value = 0;
            try
            {
                value = double.Parse(str);
            }
            catch (Exception ex)
            {
                Log($"Convert \"{str}\" to double failed: {ex.Message}");
                return 0;
            }

            if (double.IsNaN(value))
            {
                return 0;
            }
            else if (value <= 0 || value >= 100)
            {
                return 0;
            }
            else
            {
                return value;
            }
        }

        private void LogRunInfo(string message, int roundNum)
        {
            Log($"[round {roundNum}] [bot {Index}] {message}");
        }

        private void Log(string message)
        {
            if (ActionLog != null)
            {
                ActionLog(message);
            }
        }

        public static bool IsSupportPython()
        {
            StringBuilder sb = new StringBuilder("python.exe", MAX_PATH);
            return PathFindOnPath(sb, null);
        }

        public static bool IsSupportJulia()
        {
            StringBuilder sb = new StringBuilder("julia.exe", MAX_PATH);
            return PathFindOnPath(sb, null);
        }

        private const int MAX_PATH = 260;

        [DllImport("shlwapi.dll", CharSet = CharSet.Auto, SetLastError = false)]
        static extern bool PathFindOnPath([In, Out] StringBuilder pszFile, [In] String[] ppszOtherDirs);

    }
}