// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.AI.MachineLearning;
namespace OnnxDemo
{
    
    public sealed class mnistInput
    {
        public TensorFloat port; // shape(784)
    }
    
    public sealed class mnistOutput
    {
        public TensorFloat dense3port; // shape(1,10)
    }
    
    public sealed class mnistModel
    {
        private LearningModel model;
        private LearningModelSession session;
        private LearningModelBinding binding;
        public static async Task<mnistModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
        {
            mnistModel learningModel = new mnistModel();
            learningModel.model = await LearningModel.LoadFromStreamAsync(stream);
            learningModel.session = new LearningModelSession(learningModel.model);
            learningModel.binding = new LearningModelBinding(learningModel.session);
            return learningModel;
        }
        public async Task<mnistOutput> EvaluateAsync(mnistInput input)
        {
            binding.Bind("port", input.port);
            var result = await session.EvaluateAsync(binding, "0");
            var output = new mnistOutput();
            output.dense3port = result.Outputs["dense3port"] as TensorFloat;
            return output;
        }
    }
}
