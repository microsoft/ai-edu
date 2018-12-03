// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace QAClient
{
    public class Answers
    {
        public Answer[] answers { get; set; }
    }

    public class Answer
    {
        public string[] questions { get; set; }
        public string answer { get; set; }
        public float score { get; set; }
        public int id { get; set; }
        public string source { get; set; }
        public object[] metadata { get; set; }

        public override string ToString()
        {
            return string.Format("Answer: {0}, Score:{1}", answer, score);
        }
    }
    /*
    {
  "answers": [
    {
      "questions": [
        "谁是院长",
        "院长是谁",
        "老大"
      ],
      "answer": "洪小文",
      "score": 50.43,
      "id": 2,
      "source": "Editorial",
      "metadata": []
}
  ]*/
}
