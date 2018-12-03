// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CartoonTranslate.OcrResult
{
    public class Rootobject
    {
        public string language { get; set; }
        public string orientation { get; set; }
        public float textAngle { get; set; }
        public Region[] regions { get; set; }
    }

    public class Region
    {
        public string boundingBox { get; set; }
        public Line[] lines { get; set; }
    }

    public class Line
    {
        public string boundingBox { get; set; }
        public Word[] words { get; set; }

        public int[] BB { get; set; }
        public string TEXT { get; set; }


        public bool Convert()
        {
            CombineWordToSentence();
            return ConvertBBFromString2Int();
        }

        private bool ConvertBBFromString2Int()
        {
            string[] tmp = boundingBox.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            if (tmp.Length == 4)
            {
                BB = new int[4];
                for (int i = 0; i < 4; i++)
                {
                    int.TryParse(tmp[i], out BB[i]);
                }
                return true;
            }
            return false;
        }

        private void CombineWordToSentence()
        {
            StringBuilder sb = new StringBuilder();
            foreach (Word word in words)
            {
                sb.Append(word.text);
            }
            this.TEXT = sb.ToString();
        }

    }

    public class Word
    {
        public string boundingBox { get; set; }
        public string text { get; set; }
    }

}