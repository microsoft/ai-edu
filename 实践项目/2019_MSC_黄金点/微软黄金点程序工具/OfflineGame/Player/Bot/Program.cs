// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Bot
{
    class Program
    {
        static void Main(string[] args)
        {
            StringBuilder inputBuilder = new StringBuilder();

            // Read and parse input information
            string infoLine = Console.In.ReadLine();
            string[] infoArray = infoLine.Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            int rowCount = int.Parse(infoArray[0]);
            inputBuilder.AppendLine(infoLine);

            while (rowCount-- > 0)
            {
                // Each line is the history for one round
                inputBuilder.AppendLine(Console.In.ReadLine());
            }

            Tuple<double, double> numbers = NumberHelper.GetNumber(inputBuilder.ToString());

            Console.Out.Write(numbers.Item1);
            Console.Out.Write('\t');
            Console.Out.Write(numbers.Item2);
        }
    }
}