// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BotRun
{
    // Bot执行结果
    public class BotRunResult
    {
        public BotRunResult(Bot bot, double masterValue, double slaveValue)
        {
            Bot = bot;
            MasterValue = masterValue;
            SlaveValue = slaveValue;
        }

        public BotRunResult(Bot bot, Tuple<double, double> values)
        {
            Bot = bot;
            MasterValue = values.Item1;
            SlaveValue = values.Item2;
        }

        public Bot Bot { get; }

        // Bot输出的第一个预测值
        public double MasterValue { get; }
        
        // Bot输出的第二个预测值
        public double SlaveValue { get; }
    }
}