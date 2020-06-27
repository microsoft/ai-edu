// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using static System.Math;

namespace OfflineScoreboard.Utils
{
    class ScoreHelper
    {
        public ScoreHelper(string[] lines, char delimiter)
        {
            _lines = lines.Clone() as string[];
            _delimiter = delimiter;
        }

        private readonly string[] _lines;
        private readonly char _delimiter;

        /// <summary>
        /// 计算某一回合里，所有人（包括当前玩家）的得分。
        /// </summary>
        /// <param name="roundIdx">回合序号，从 0 开始。</param>
        /// <param name="submitted">不为 null 时，必须包含有效数字。为 null 时，表示当前玩家没有提交合法的数。</param>
        /// <returns>这个回合的分数计算结果。</returns>
        public ScoreCalculationResult CalculateScore(int roundIdx, double[] submitted)
        {
            var historicalNumbers = _lines[roundIdx + 1].Split(_delimiter)
                .Skip(1) // 在计算当前轮的结果时，去掉历史数据里的黄金点，并重新计算。
                .Select(double.Parse)
                .ToList();

            var indexedValidSubmittedSets = historicalNumbers
                .Select((n, numIdx) => new { Number = n, PlayerIndex = numIdx / 2 }) // 每人提交2个数，我们可以通过整数除法来得到对应的玩家序号。
                .GroupBy(idxedNum => idxedNum.PlayerIndex)
                .Select(group => new
                {
                    PlayerIndex = group.Key,
                    Submitted = group.Select(idxedNum => idxedNum.Number).ToArray()
                }).Where(idxedSubmittedSet => idxedSubmittedSet.Submitted.All(num => num > 0))
                .ToList();
            int bonus = indexedValidSubmittedSets.Count; // 计算加分时，不计当前玩家。

            if (submitted != null)
            {
                indexedValidSubmittedSets.Add(new
                {
                    PlayerIndex = historicalNumbers.Count / 2, // 当前玩家被加在最后。
                    Submitted = submitted
                });
            }

            int allPlayerCount = historicalNumbers.Count / 2 + 1;
            int validPlayerCount = indexedValidSubmittedSets.Count;

            var validNumbers = indexedValidSubmittedSets
                .SelectMany(idxedSet => idxedSet.Submitted)
                .ToList();

            // 黄金点的计算，不涉及非法的数字，在历史数据中即为 0。
            var newG = validNumbers.Average() * 0.618;

            Func<double, double> getDistance = n => Abs(n - newG);

            var sortedDistances = validNumbers
                .Select(getDistance)
                .Distinct()
                .OrderBy(d => d)
                .ToList();

            // 以数组存储，方便进行集合运算。
            var smallestDis = new[] { sortedDistances.First() };
            var biggestDis = new[] { sortedDistances.Last() };

            int?[] scores = new int?[allPlayerCount]; // 通过额外的 null 状态来辨别不合法数据，即 0。
            bool nearest = false;
            bool farthest = false;

            foreach (var idxedNumSet in indexedValidSubmittedSets)
            {
                // 当前的参与者提交的数据放在最后。
                int playerIdx = idxedNumSet.PlayerIndex;
                bool isRealPlayer = playerIdx == allPlayerCount - 1;

                int thisRoundScore = 0;

                // 提交的数可能在黄金点左右都有。会产生一样的距离。
                // Except和Intersect都是集合运算，此处先去重，以满足集合特性。
                var distinctDis = idxedNumSet.Submitted.Select(getDistance).Distinct().ToArray();

                var notSmallestDis = distinctDis.Except(smallestDis).ToArray();
                if (notSmallestDis.Length < distinctDis.Length)
                {
                    // 存在某个提交的数，属于距黄金点最近的一组。并且最多只得一次分。
                    thisRoundScore += bonus;

                    // 必须是当前游戏者我们才将 HasNearest 设为 true。
                    nearest |= isRealPlayer;
                }

                if (notSmallestDis.Intersect(biggestDis).Any())
                {
                    // 除了最近的数以外，还存在最远的数。并且最多只扣一次分。
                    // 如果所有数都一样，从上面可知，算是得分。
                    thisRoundScore -= 2;

                    farthest |= isRealPlayer;
                }

                scores[playerIdx] = thisRoundScore;
            }

            return new ScoreCalculationResult(scores, nearest: nearest, farthest: farthest);
        }

        public class ScoreCalculationResult
        {
            public ScoreCalculationResult(int?[] scores, bool nearest, bool farthest)
            {
                Scores = scores;
                HasNearest = nearest;
                HasFarthest = farthest;
            }

            /// <summary>
            /// 获取所有人一个回合里的得分。得分顺序
            /// </summary>
            public IReadOnlyList<int?> Scores { get; }

            /// <summary>
            /// 获取当前玩家提交的数里，是否有最靠近黄金点的数。
            /// </summary>
            public bool HasNearest { get; }

            /// <summary>
            /// 获取当前玩家提交的数里，是否有最远离黄金点的数。
            /// </summary>
            public bool HasFarthest { get; }
        }
    }
}