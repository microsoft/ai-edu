// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.IO;
using System.Threading.Tasks;

namespace CognitiveMiddlewareService.CognitiveServices
{
    public interface IVisionService
    {
        Task<Landmark> RecognizeLandmarkAsync(Stream imgStream);

        Task<Celebrity> RecognizeCelebrityAsync(Stream imgStream);
    }
}
