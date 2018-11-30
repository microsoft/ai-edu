// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 这里是服务器接口相关的数据结构，主要是用于反序列化JSON串。我们尽量不将这些结构暴露给服务器接口的调用者（即 Game 类）以外的代码。
namespace GoldedNumberClient.Models
{
    public class State
    {
        public string UserId { get; set; }

        public string NickName { get; set; }

        public string RoomId { get; set; }

        public string Numbers { get; set; }

        public string RoundId { get; set; }

        public int LeftTime { get; set; }
    }

    public class History
    {
        public List<RoundNumbers> Rounds { get; set; }

        public class RoundNumbers
        {
            public string RoundId { get; }

            public double GoldenNumber { get; set; }

            public DateTime Time { get; set; }

            public List<UserNumber> UserNumbers { get; set; }
        }

        public class UserNumber
        {
            public string UserId { get; set; }

            public double MasterNumber { get; set; }

            public double SlaveNumber { get; set; }

            public int Score { get; set; }
        }

        public Dictionary<string, string> NickNames { get; set; }
    }

    public class NewRoom
    {
        public string RoomId { get; set; }
    }
}