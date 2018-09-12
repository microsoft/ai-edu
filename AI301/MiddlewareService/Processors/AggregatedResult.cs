// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using CognitiveMiddlewareService.MiddlewareService;

namespace CognitiveMiddlewareService.Processors
{
    public class AggregatedResult
    {
        public LandmarkResult Landmark { get; set; }

        public CelebrityResult Celebrity { get; set; }
    }
}
