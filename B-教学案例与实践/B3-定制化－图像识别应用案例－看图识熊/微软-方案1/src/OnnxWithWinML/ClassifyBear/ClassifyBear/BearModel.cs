// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.AI.MachineLearning;
namespace ClassifyBear
{
    
    public sealed class BearModelInput
    {
        public ImageFeatureValue data; // BitmapPixelFormat: Bgra8, BitmapAlphaMode: Premultiplied, width: 227, height: 227
    }
    
    public sealed class BearModelOutput
    {
        public TensorString classLabel; // shape(-1,1)
        public IList<Dictionary<string,float>> loss;
    }
    
    public sealed class BearModelModel
    {
        private LearningModel model;
        private LearningModelSession session;
        private LearningModelBinding binding;
        public static async Task<BearModelModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
        {
            BearModelModel learningModel = new BearModelModel();
            learningModel.model = await LearningModel.LoadFromStreamAsync(stream);
            learningModel.session = new LearningModelSession(learningModel.model);
            learningModel.binding = new LearningModelBinding(learningModel.session);
            return learningModel;
        }
        public async Task<BearModelOutput> EvaluateAsync(BearModelInput input)
        {
            binding.Bind("data", input.data);
            var result = await session.EvaluateAsync(binding, "0");
            var output = new BearModelOutput();
            output.classLabel = result.Outputs["classLabel"] as TensorString;
            output.loss = result.Outputs["loss"] as IList<Dictionary<string,float>>;
            return output;
        }
    }
}
