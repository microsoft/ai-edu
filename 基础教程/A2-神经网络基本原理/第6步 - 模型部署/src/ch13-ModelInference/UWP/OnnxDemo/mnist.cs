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
        public TensorFloat fc1x; // shape(1,784)
    }
    
    public sealed class mnistOutput
    {
        public TensorFloat activation3y; // shape(1,10)
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
            binding.Bind("fc1x", input.fc1x);
            var result = await session.EvaluateAsync(binding, "0");
            var output = new mnistOutput();
            output.activation3y = result.Outputs["activation3y"] as TensorFloat;
            return output;
        }
    }
}
