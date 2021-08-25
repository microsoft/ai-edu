// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

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