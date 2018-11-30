// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bot
{
    public class NumberHelper
    {
        // !!! 需要提供实现的函数。
        // 请注意，该函数会在后台线程执行。
        public static Tuple<double, double> GetNumber(string input)
        {
            return Tuple.Create(42.0, 1.5);
        }
    }
}