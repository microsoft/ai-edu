// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Microsoft.ML.Scoring;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace Model
{
    public partial class Bear
    {
        const string modelName = "Bear";
        private ModelManager manager;

        private static List<string> inferInputNames = new List<string> { "data" };
        private static List<string> inferOutputNames = new List<string> { "classLabel" };

        /// <summary>
        /// Returns an instance of Bear model.
        /// </summary>
        public Bear()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string dllpath = Uri.UnescapeDataString(uri.Path);
            string modelpath = Path.Combine(Path.GetDirectoryName(dllpath), "Bear");
            string path = Path.Combine(modelpath, "00000001");
            manager = new ModelManager(path, true);
            manager.InitModel(modelName, int.MaxValue);
        }

        /// <summary>
        /// Returns instance of Bear model instantiated from exported model path.
        /// </summary>
        /// <param name="path">Exported model directory.</param>
        public Bear(string path)
        {
            manager = new ModelManager(path, true);
            manager.InitModel(modelName, int.MaxValue);
        }

        /// <summary>
        /// Runs inference on Bear model for a batch of inputs.
        /// The shape of each input is the same as that for the non-batch case above.
        /// </summary>
        public IEnumerable<IEnumerable<string>> Infer(IEnumerable<IEnumerable<float>> dataBatch)
        {
            List<float> dataCombined = new List<float>();
            long batchCount = 0;
            foreach (var input in dataBatch)
            {
                dataCombined.AddRange(input);
                ++batchCount;
            }

            List<Tensor> result = manager.RunModel(
                modelName,
                int.MaxValue,
                inferInputNames,
                new List<Tensor> { new Tensor(dataCombined, new List<long> { batchCount, 3, 227, 227 }) },
                inferOutputNames
            );

            int classLabelBatchNum = (int)result[0].GetShape()[0];
            int classLabelBatchSize = (int)result[0].GetShape().Aggregate((a, x) => a * x) / classLabelBatchNum;
            for (int batchNum = 0, offset = 0; batchNum < classLabelBatchNum; batchNum++, offset += classLabelBatchSize)
            {
                List<string> tmp = new List<string>();
                result[0].CopyTo(tmp, offset, classLabelBatchSize);
                yield return tmp;
            }
        }
    } // END OF CLASS
} // END OF NAMESPACE
