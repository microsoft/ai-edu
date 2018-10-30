// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using CognitiveMiddlewareService.MiddlewareService;

namespace CognitiveMiddlewareService.Processors
{
    public class AggregatedResult
    {
        public LandmarkResult Landmark { get; set; }

        public CelebrityResult Celebrity { get; set; }
    }
}